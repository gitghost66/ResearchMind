"""
Tests for Alembic migration definitions and metadata consistency.
"""

import os
from pathlib import Path

import pytest
from alembic.config import Config
from alembic.script import ScriptDirectory

from app.models import Base

EXPECTED_TABLES = {
    "research_runs",
    "sub_questions",
    "sources",
    "passages",
    "claims",
    "claim_evidence",
    "contradictions",
    "contradiction_claims",
    "reports",
    "run_events",
    "llm_usage",
}


def test_metadata_contains_all_11_tables():
    """Verify that SQLAlchemy metadata registers all 11 required tables."""
    registered = set(Base.metadata.tables.keys())
    assert EXPECTED_TABLES.issubset(registered), f"Missing tables: {EXPECTED_TABLES - registered}"
    assert len(registered) == 11


def test_alembic_migration_script_validity():
    """Verify that Alembic locates the migration chain cleanly."""
    backend_dir = Path(__file__).parent.parent
    alembic_ini_path = backend_dir / "alembic.ini"
    assert alembic_ini_path.exists(), "alembic.ini missing"

    cfg = Config(str(alembic_ini_path))
    cfg.set_main_option("script_location", str(backend_dir / "alembic"))

    script_dir = ScriptDirectory.from_config(cfg)
    head_rev = script_dir.get_current_head()

    assert head_rev == "0003_run_event_seq_and_run_options"
    chain = (
        "0001_initial_schema",
        "0002_passage_embedding_model",
        "0003_run_event_seq_and_run_options",
    )
    for revision_id in chain:
        rev = script_dir.get_revision(revision_id)
        assert rev is not None
        assert rev.module is not None
        assert hasattr(rev.module, "upgrade")
        assert hasattr(rev.module, "downgrade")

    for parent, child in zip(chain, chain[1:], strict=False):
        rev = script_dir.get_revision(child)
        assert rev is not None
        assert rev.down_revision == parent


def _render_offline_sql(revision_id: str, direction: str = "upgrade") -> str:
    """
    Render one migration step as PostgreSQL SQL without a database.

    This catches dialect-level breakage in the migration scripts themselves
    (pgvector column types, JSONB defaults) in CI, where no Postgres is available.
    """
    from alembic.migration import MigrationContext
    from alembic.operations import Operations

    backend_dir = Path(__file__).parent.parent
    cfg = Config(str(backend_dir / "alembic.ini"))
    cfg.set_main_option("script_location", str(backend_dir / "alembic"))

    script_dir = ScriptDirectory.from_config(cfg)
    revision = script_dir.get_revision(revision_id)
    assert revision is not None

    statements: list[str] = []
    context = MigrationContext.configure(
        dialect_name="postgresql",
        opts={"as_sql": True, "output_buffer": _Collector(statements)},
    )
    # Operations.context binds the `alembic.op` proxy the migration module uses.
    with context.begin_transaction(), Operations.context(context):
        getattr(revision.module, direction)()

    return "\n".join(statements)


def test_offline_sql_generation_targets_postgresql():
    sql = _render_offline_sql("0001_initial_schema")
    assert "CREATE EXTENSION IF NOT EXISTS vector" in sql
    assert "CREATE TABLE research_runs" in sql
    assert "vector(768)" in sql.lower()
    assert "USING hnsw" in sql


def test_0002_adds_embedding_model_and_drops_the_backfill_default():
    """Existing rows are backfilled; new rows must name their model."""
    sql = _render_offline_sql("0002_passage_embedding_model")

    assert "ALTER TABLE passages ADD COLUMN embedding_model VARCHAR(100)" in sql
    assert "DEFAULT 'text-embedding-004' NOT NULL" in sql
    assert "ALTER TABLE passages ALTER COLUMN embedding_model DROP DEFAULT" in sql


def test_0002_downgrade_drops_the_column():
    sql = _render_offline_sql("0002_passage_embedding_model", direction="downgrade")
    assert "ALTER TABLE passages DROP COLUMN embedding_model" in sql


def test_0003_numbers_existing_events_then_enforces_order():
    """Existing events are numbered per run before seq becomes NOT NULL and unique."""
    sql = _render_offline_sql("0003_run_event_seq_and_run_options")

    assert "ALTER TABLE research_runs ADD COLUMN options JSONB DEFAULT '{}' NOT NULL" in sql
    assert "ALTER TABLE run_events ADD COLUMN seq INTEGER" in sql
    assert "row_number() OVER (PARTITION BY run_id ORDER BY ts, id)" in sql
    assert sql.index("row_number()") < sql.index("ALTER COLUMN seq SET NOT NULL")
    assert "CREATE UNIQUE INDEX ix_run_events_run_id_seq ON run_events (run_id, seq)" in sql


def test_0003_downgrade_drops_both_columns():
    sql = _render_offline_sql("0003_run_event_seq_and_run_options", direction="downgrade")
    assert "DROP INDEX ix_run_events_run_id_seq" in sql
    assert "ALTER TABLE run_events DROP COLUMN seq" in sql
    assert "ALTER TABLE research_runs DROP COLUMN options" in sql


def test_models_match_migration_0003():
    seq = Base.metadata.tables["run_events"].c.seq
    assert seq.nullable is False
    indexes = {ix.name: ix for ix in Base.metadata.tables["run_events"].indexes}
    assert indexes["ix_run_events_run_id_seq"].unique
    assert [c.name for c in indexes["ix_run_events_run_id_seq"].columns] == ["run_id", "seq"]
    assert Base.metadata.tables["research_runs"].c.options.nullable is False


def test_passage_model_matches_migration_0002():
    column = Base.metadata.tables["passages"].c.embedding_model
    assert column.nullable is False
    assert column.type.length == 100  # type: ignore[attr-defined]


class _Collector:
    """Minimal write-only buffer for Alembic's offline SQL output."""

    def __init__(self, sink: list[str]) -> None:
        self._sink = sink

    def write(self, text: str) -> None:
        if text.strip():
            self._sink.append(text)

    def flush(self) -> None:
        pass


@pytest.mark.integration
@pytest.mark.skipif(
    not os.getenv("DATABASE_URL"),
    reason="requires a real PostgreSQL+pgvector database (set DATABASE_URL)",
)
def test_migration_applies_to_a_real_database():
    """
    M1 acceptance: the migration applies cleanly to an empty database.

    Optional and clearly marked (AGENTS.md rule 7) — it needs a live Postgres
    with pgvector available, so CI skips it.
    """
    from alembic import command

    backend_dir = Path(__file__).parent.parent
    cfg = Config(str(backend_dir / "alembic.ini"))
    cfg.set_main_option("script_location", str(backend_dir / "alembic"))

    command.upgrade(cfg, "head")
    command.downgrade(cfg, "base")
