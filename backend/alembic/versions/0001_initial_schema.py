"""Initial schema with all 11 tables and pgvector HNSW index

Revision ID: 0001_initial_schema
Revises:
Create Date: 2026-10-21 20:00:00.000000

"""
from collections.abc import Sequence

import sqlalchemy as sa
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0001_initial_schema"
down_revision: str | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    # 0. Enable pgvector extension
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # 1. research_runs
    op.create_table(
        "research_runs",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("question", sa.Text(), nullable=False),
        sa.Column("mode", sa.String(length=50), nullable=False, server_default="quick"),
        sa.Column("status", sa.String(length=50), nullable=False, server_default="queued"),
        sa.Column("user_context", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default="{}"),
        sa.Column("plan", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default="{}"),
        sa.Column("iteration", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("finished_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("error", sa.Text(), nullable=True),
    )
    op.create_index("ix_research_runs_status", "research_runs", ["status"])

    # 2. sub_questions
    op.create_table(
        "sub_questions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("run_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("research_runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("domain", sa.String(length=100), nullable=False, server_default="web"),
        sa.Column("status", sa.String(length=50), nullable=False, server_default="pending"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_sub_questions_run_id", "sub_questions", ["run_id"])

    # 3. sources
    op.create_table(
        "sources",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("run_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("research_runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("url", sa.Text(), nullable=False),
        sa.Column("title", sa.Text(), nullable=False, server_default=""),
        sa.Column("published_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("domain_type", sa.String(length=50), nullable=False, server_default="unknown"),
        sa.Column("reliability_score", sa.Float(), nullable=False, server_default="0.5"),
        sa.Column("content_hash", sa.String(length=64), nullable=False, server_default=""),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_sources_run_id", "sources", ["run_id"])
    op.create_index("ix_sources_content_hash", "sources", ["content_hash"])

    # 4. passages
    op.create_table(
        "passages",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("source_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("sources.id", ondelete="CASCADE"), nullable=False),
        sa.Column("sub_question_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("sub_questions.id", ondelete="SET NULL"), nullable=True),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(768), nullable=True),
        sa.Column("chunk_index", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_passages_source_id", "passages", ["source_id"])
    op.create_index("ix_passages_sub_question_id", "passages", ["sub_question_id"])
    # HNSW vector index for fast cosine distance search (Section 7 PRD requirement)
    op.execute(
        "CREATE INDEX IF NOT EXISTS ix_passages_embedding_hnsw "
        "ON passages USING hnsw (embedding vector_cosine_ops)"
    )

    # 5. claims
    op.create_table(
        "claims",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("run_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("research_runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("sub_question_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("sub_questions.id", ondelete="SET NULL"), nullable=True),
        sa.Column("text", sa.Text(), nullable=False),
        sa.Column("verdict", sa.String(length=50), nullable=False, server_default="unverified"),
        sa.Column("confidence", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_claims_run_id", "claims", ["run_id"])

    # 6. claim_evidence
    op.create_table(
        "claim_evidence",
        sa.Column("claim_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("claims.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("passage_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("passages.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("stance", sa.String(length=50), nullable=False, server_default="supports"),
    )

    # 7. contradictions
    op.create_table(
        "contradictions",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("run_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("research_runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("explanation", sa.Text(), nullable=False),
        sa.Column("likely_cause", sa.Text(), nullable=False, server_default=""),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_contradictions_run_id", "contradictions", ["run_id"])

    # 8. contradiction_claims
    op.create_table(
        "contradiction_claims",
        sa.Column("contradiction_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("contradictions.id", ondelete="CASCADE"), primary_key=True),
        sa.Column("claim_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("claims.id", ondelete="CASCADE"), primary_key=True),
    )

    # 9. reports
    op.create_table(
        "reports",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("run_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("research_runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("content", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default="{}"),
        sa.Column("markdown", sa.Text(), nullable=False),
        sa.Column("overall_confidence", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("version", sa.Integer(), nullable=False, server_default="1"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_reports_run_id", "reports", ["run_id"])

    # 10. run_events
    op.create_table(
        "run_events",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("run_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("research_runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("ts", sa.DateTime(timezone=True), nullable=False),
        sa.Column("agent", sa.String(length=100), nullable=False),
        sa.Column("type", sa.String(length=100), nullable=False),
        sa.Column("payload", postgresql.JSONB(astext_type=sa.Text()), nullable=False, server_default="{}"),
    )
    op.create_index("ix_run_events_run_id", "run_events", ["run_id"])

    # 11. llm_usage
    op.create_table(
        "llm_usage",
        sa.Column("id", postgresql.UUID(as_uuid=True), primary_key=True),
        sa.Column("run_id", postgresql.UUID(as_uuid=True), sa.ForeignKey("research_runs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("agent", sa.String(length=100), nullable=False),
        sa.Column("model", sa.String(length=100), nullable=False),
        sa.Column("tokens_in", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("tokens_out", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("cost_usd", sa.Float(), nullable=False, server_default="0.0"),
        sa.Column("latency_ms", sa.Integer(), nullable=False, server_default="0"),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
    )
    op.create_index("ix_llm_usage_run_id", "llm_usage", ["run_id"])


def downgrade() -> None:
    op.drop_table("llm_usage")
    op.drop_table("run_events")
    op.drop_table("reports")
    op.drop_table("contradiction_claims")
    op.drop_table("contradictions")
    op.drop_table("claim_evidence")
    op.drop_table("claims")
    op.execute("DROP INDEX IF EXISTS ix_passages_embedding_hnsw")
    op.drop_table("passages")
    op.drop_table("sources")
    op.drop_table("sub_questions")
    op.drop_table("research_runs")
    op.execute("DROP EXTENSION IF EXISTS vector")
