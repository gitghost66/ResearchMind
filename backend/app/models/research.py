"""
Research models: research_runs, sub_questions, reports, run_events, and llm_usage.
"""

import uuid
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from sqlalchemy import DateTime, Float, ForeignKey, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import JSON

from app.models.base import Base, TimestampMixin, UniversalUUID

if TYPE_CHECKING:
    from app.models.claims import Claim, Contradiction
    from app.models.evidence import Passage, Source

# Universal JSON type: JSONB on PostgreSQL, JSON on SQLite
UniversalJSON = JSON().with_variant(JSONB, "postgresql")


class ResearchRun(Base, TimestampMixin):
    """
    Core research run record.
    Tracks status, question, mode, user context, plan, and iteration progress.
    """

    __tablename__ = "research_runs"

    id: Mapped[uuid.UUID] = mapped_column(
        UniversalUUID,
        primary_key=True,
        default=uuid.uuid4,
    )
    question: Mapped[str] = mapped_column(Text, nullable=False)
    mode: Mapped[str] = mapped_column(String(50), default="quick", nullable=False)
    status: Mapped[str] = mapped_column(
        String(50),
        default="queued",
        nullable=False,
        index=True,
    )  # queued, running, completed, failed, cancelled
    user_context: Mapped[dict[str, Any]] = mapped_column(
        UniversalJSON,
        default=dict,
        nullable=False,
    )
    plan: Mapped[dict[str, Any]] = mapped_column(
        UniversalJSON,
        default=dict,
        nullable=False,
    )
    iteration: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)

    # Relationships (cascade on delete)
    sub_questions: Mapped[list["SubQuestion"]] = relationship(
        "SubQuestion",
        back_populates="run",
        cascade="all, delete-orphan",
    )
    sources: Mapped[list["Source"]] = relationship(
        "Source",
        back_populates="run",
        cascade="all, delete-orphan",
    )
    claims: Mapped[list["Claim"]] = relationship(
        "Claim",
        back_populates="run",
        cascade="all, delete-orphan",
    )
    contradictions: Mapped[list["Contradiction"]] = relationship(
        "Contradiction",
        back_populates="run",
        cascade="all, delete-orphan",
    )
    reports: Mapped[list["Report"]] = relationship(
        "Report",
        back_populates="run",
        cascade="all, delete-orphan",
    )
    events: Mapped[list["RunEvent"]] = relationship(
        "RunEvent",
        back_populates="run",
        cascade="all, delete-orphan",
    )
    usage_records: Mapped[list["LLMUsage"]] = relationship(
        "LLMUsage",
        back_populates="run",
        cascade="all, delete-orphan",
    )


class SubQuestion(Base, TimestampMixin):
    """
    Sub-question produced by the Planner agent.
    """

    __tablename__ = "sub_questions"

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
    text: Mapped[str] = mapped_column(Text, nullable=False)
    domain: Mapped[str] = mapped_column(String(100), default="web", nullable=False)
    status: Mapped[str] = mapped_column(
        String(50),
        default="pending",
        nullable=False,
    )  # pending, researching, completed, failed

    run: Mapped["ResearchRun"] = relationship("ResearchRun", back_populates="sub_questions")
    passages: Mapped[list["Passage"]] = relationship(
        "Passage",
        back_populates="sub_question",
        cascade="all, delete-orphan",
    )


class Report(Base, TimestampMixin):
    """
    Final synthesized research report.
    """

    __tablename__ = "reports"

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
    content: Mapped[dict[str, Any]] = mapped_column(
        UniversalJSON,
        default=dict,
        nullable=False,
    )
    markdown: Mapped[str] = mapped_column(Text, nullable=False)
    overall_confidence: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    version: Mapped[int] = mapped_column(Integer, default=1, nullable=False)

    run: Mapped["ResearchRun"] = relationship("ResearchRun", back_populates="reports")


class RunEvent(Base):
    """
    Streamed progress event for an agent in a run.
    """

    __tablename__ = "run_events"

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
    ts: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )
    agent: Mapped[str] = mapped_column(String(100), nullable=False)
    type: Mapped[str] = mapped_column(String(100), nullable=False)  # agent_started, etc.
    payload: Mapped[dict[str, Any]] = mapped_column(
        UniversalJSON,
        default=dict,
        nullable=False,
    )

    run: Mapped["ResearchRun"] = relationship("ResearchRun", back_populates="events")


class LLMUsage(Base):
    """
    Token and cost tracking per agent invocation.
    """

    __tablename__ = "llm_usage"

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
    agent: Mapped[str] = mapped_column(String(100), nullable=False)
    model: Mapped[str] = mapped_column(String(100), nullable=False)
    tokens_in: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    tokens_out: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    cost_usd: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=lambda: datetime.now(UTC),
        nullable=False,
    )

    run: Mapped["ResearchRun"] = relationship("ResearchRun", back_populates="usage_records")
