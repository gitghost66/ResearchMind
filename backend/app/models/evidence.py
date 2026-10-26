"""
Evidence models: sources and passages with pgvector embeddings.
"""

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.core.config import get_settings
from app.models.base import Base, TimestampMixin, UniversalUUID

if TYPE_CHECKING:
    from app.models.claims import ClaimEvidence
    from app.models.research import ResearchRun, SubQuestion


# Compilation rule for SQLite compatibility during testing
@compiles(Vector, "sqlite")
def _compile_vector_sqlite(type_, compiler, **kw):  # type: ignore[no-untyped-def]
    return "TEXT"


def _current_embedding_model() -> str:
    """Default for ``Passage.embedding_model``: the configured model."""
    return get_settings().embedding_model


class Source(Base, TimestampMixin):
    """
    Source discovered by web/academic/technical research agents.
    """

    __tablename__ = "sources"

    id: Mapped[uuid.UUID] = mapped_column(
        UniversalUUID,
        primary_key=True,
        default=uuid.uuid4,
    )
    run_id: Mapped[uuid.UUID] = mapped_column(
        UniversalUUID,
        ForeignKey("research_runs.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    url: Mapped[str] = mapped_column(Text, nullable=False)
    title: Mapped[str] = mapped_column(Text, nullable=False, default="")
    published_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    domain_type: Mapped[str] = mapped_column(String(50), default="unknown", nullable=False)
    reliability_score: Mapped[float] = mapped_column(Float, default=0.5, nullable=False)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, default="", index=True)

    run: Mapped["ResearchRun"] = relationship("ResearchRun", back_populates="sources")
    passages: Mapped[list["Passage"]] = relationship(
        "Passage",
        back_populates="source",
        cascade="all, delete-orphan",
    )


class Passage(Base, TimestampMixin):
    """
    Extracted text chunk from a source with vector embedding.
    """

    __tablename__ = "passages"

    id: Mapped[uuid.UUID] = mapped_column(
        UniversalUUID,
        primary_key=True,
        default=uuid.uuid4,
    )
    source_id: Mapped[uuid.UUID] = mapped_column(
        UniversalUUID,
        ForeignKey("sources.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    sub_question_id: Mapped[uuid.UUID | None] = mapped_column(
        UniversalUUID,
        ForeignKey("sub_questions.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    text: Mapped[str] = mapped_column(Text, nullable=False)
    # Width must equal Settings.embedding_dimension (768); changing it needs a
    # new migration.
    embedding = mapped_column(Vector(768), nullable=True)
    # Vectors from different models are not comparable, so each passage records
    # the model that embedded it (migration 0002).
    embedding_model: Mapped[str] = mapped_column(
        String(100),
        nullable=False,
        default=_current_embedding_model,
    )
    chunk_index: Mapped[int] = mapped_column(Integer, default=0, nullable=False)

    source: Mapped["Source"] = relationship("Source", back_populates="passages")
    sub_question: Mapped["SubQuestion"] = relationship("SubQuestion", back_populates="passages")
    evidence_links: Mapped[list["ClaimEvidence"]] = relationship(
        "ClaimEvidence",
        back_populates="passage",
        cascade="all, delete-orphan",
    )
