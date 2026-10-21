"""
Tests for Alembic migration definitions and metadata consistency.
"""

from pathlib import Path

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
    """Verify that Alembic locates the initial migration revision cleanly."""
    backend_dir = Path(__file__).parent.parent
    alembic_ini_path = backend_dir / "alembic.ini"
    assert alembic_ini_path.exists(), "alembic.ini missing"

    cfg = Config(str(alembic_ini_path))
    cfg.set_main_option("script_location", str(backend_dir / "alembic"))

    script_dir = ScriptDirectory.from_config(cfg)
    head_rev = script_dir.get_current_head()

    assert head_rev == "0001_initial_schema"
    rev = script_dir.get_revision("0001_initial_schema")
    assert rev is not None
    assert rev.module is not None
    assert hasattr(rev.module, "upgrade")
    assert hasattr(rev.module, "downgrade")
