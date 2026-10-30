"""Worker leases on runs, and LangGraph checkpoint tables

Revision ID: 0004_run_leases_and_checkpoints
Revises: 0003_run_event_seq_and_options
Create Date: 2026-10-28 12:00:00.000000

* ``research_runs.worker_id`` / ``lease_expires_at`` / ``attempts``: runs are
  claimed from the database by worker processes, which renew a lease while
  they work. A run whose lease lapses is resumed by another worker. Runs that
  already started count as one attempt.
* ``graph_checkpoints``, ``graph_checkpoint_blobs``, ``graph_checkpoint_writes``:
  the LangGraph checkpoints a resumed run continues from
  (``app.services.checkpointer.SqlCheckpointSaver``).
"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0004_run_leases_and_checkpoints"
down_revision: str | None = "0003_run_event_seq_and_options"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column("research_runs", sa.Column("worker_id", sa.String(length=200), nullable=True))
    op.add_column(
        "research_runs",
        sa.Column("lease_expires_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.add_column(
        "research_runs",
        sa.Column("attempts", sa.Integer(), nullable=False, server_default="0"),
    )
    op.execute("UPDATE research_runs SET attempts = 1 WHERE started_at IS NOT NULL")

    op.create_table(
        "graph_checkpoints",
        sa.Column("thread_id", sa.String(length=255), nullable=False),
        sa.Column("checkpoint_ns", sa.String(length=255), nullable=False, server_default=""),
        sa.Column("checkpoint_id", sa.String(length=64), nullable=False),
        sa.Column("parent_checkpoint_id", sa.String(length=64), nullable=True),
        sa.Column("checkpoint_type", sa.String(length=32), nullable=False),
        sa.Column("checkpoint", sa.LargeBinary(), nullable=False),
        sa.Column("metadata_type", sa.String(length=32), nullable=False),
        sa.Column("metadata_blob", sa.LargeBinary(), nullable=False),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            nullable=False,
            server_default=sa.text("now()"),
        ),
        sa.PrimaryKeyConstraint("thread_id", "checkpoint_ns", "checkpoint_id"),
    )
    op.create_table(
        "graph_checkpoint_blobs",
        sa.Column("thread_id", sa.String(length=255), nullable=False),
        sa.Column("checkpoint_ns", sa.String(length=255), nullable=False, server_default=""),
        sa.Column("channel", sa.String(length=255), nullable=False),
        sa.Column("version", sa.String(length=64), nullable=False),
        sa.Column("type", sa.String(length=32), nullable=False),
        sa.Column("blob", sa.LargeBinary(), nullable=False),
        sa.PrimaryKeyConstraint("thread_id", "checkpoint_ns", "channel", "version"),
    )
    op.create_table(
        "graph_checkpoint_writes",
        sa.Column("thread_id", sa.String(length=255), nullable=False),
        sa.Column("checkpoint_ns", sa.String(length=255), nullable=False, server_default=""),
        sa.Column("checkpoint_id", sa.String(length=64), nullable=False),
        sa.Column("task_id", sa.String(length=255), nullable=False),
        sa.Column("idx", sa.Integer(), nullable=False),
        sa.Column("channel", sa.String(length=255), nullable=False),
        sa.Column("type", sa.String(length=32), nullable=False),
        sa.Column("blob", sa.LargeBinary(), nullable=False),
        sa.Column("task_path", sa.Text(), nullable=False, server_default=""),
        sa.PrimaryKeyConstraint("thread_id", "checkpoint_ns", "checkpoint_id", "task_id", "idx"),
    )


def downgrade() -> None:
    op.drop_table("graph_checkpoint_writes")
    op.drop_table("graph_checkpoint_blobs")
    op.drop_table("graph_checkpoints")
    op.drop_column("research_runs", "attempts")
    op.drop_column("research_runs", "lease_expires_at")
    op.drop_column("research_runs", "worker_id")
