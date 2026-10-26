"""Record which embedding model produced each passage vector

Revision ID: 0002_passage_embedding_model
Revises: 0001_initial_schema
Create Date: 2026-10-24 12:00:00.000000

Vectors from different embedding models are not comparable, so each passage
records the model that embedded it. text-embedding-004 was shut down on
2026-01-14 and the default moved to gemini-embedding-001.
"""
from collections.abc import Sequence

import sqlalchemy as sa

from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0002_passage_embedding_model"
down_revision: str | None = "0001_initial_schema"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

# Rows written under 0001 were embedded with text-embedding-004, the default at
# the time, so the backfill records that model rather than the current one.
_LEGACY_EMBEDDING_MODEL = "text-embedding-004"


def upgrade() -> None:
    op.add_column(
        "passages",
        sa.Column(
            "embedding_model",
            sa.String(length=100),
            nullable=False,
            server_default=_LEGACY_EMBEDDING_MODEL,
        ),
    )
    # The default only exists to backfill existing rows; new rows must state
    # their model explicitly.
    op.alter_column("passages", "embedding_model", server_default=None)


def downgrade() -> None:
    op.drop_column("passages", "embedding_model")
