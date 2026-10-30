"""Order run events per run, and store per-run options

Revision ID: 0003_run_event_seq_and_options
Revises: 0002_passage_embedding_model
Create Date: 2026-10-27 12:00:00.000000

* ``run_events.seq`` numbers a run's events 1, 2, 3, ... in commit order. The
  SSE endpoint uses it as the event id, so a client reconnecting with
  ``Last-Event-ID`` gets exactly the events it missed. Timestamps cannot do
  this: concurrent branches can commit out of timestamp order. Existing rows
  are numbered by ``(ts, id)``.
* ``research_runs.options`` holds per-run settings from the API request (for
  now ``max_sources``), kept apart from the client's free-form ``user_context``.
"""
from collections.abc import Sequence

import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0003_run_event_seq_and_options"
down_revision: str | None = "0002_passage_embedding_model"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.add_column(
        "research_runs",
        sa.Column(
            "options",
            postgresql.JSONB(astext_type=sa.Text()),
            nullable=False,
            server_default="{}",
        ),
    )

    op.add_column("run_events", sa.Column("seq", sa.Integer(), nullable=True))
    op.execute(
        """
        UPDATE run_events AS e
        SET seq = numbered.rn
        FROM (
            SELECT id, row_number() OVER (PARTITION BY run_id ORDER BY ts, id) AS rn
            FROM run_events
        ) AS numbered
        WHERE e.id = numbered.id
        """
    )
    op.alter_column("run_events", "seq", nullable=False)
    op.create_index("ix_run_events_run_id_seq", "run_events", ["run_id", "seq"], unique=True)


def downgrade() -> None:
    op.drop_index("ix_run_events_run_id_seq", table_name="run_events")
    op.drop_column("run_events", "seq")
    op.drop_column("research_runs", "options")
