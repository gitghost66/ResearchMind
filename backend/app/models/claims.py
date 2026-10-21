"""
Claim and Contradiction models: claims, claim_evidence, contradictions, and contradiction_claims.
"""

import uuid
from typing import TYPE_CHECKING

from sqlalchemy import Float, ForeignKey, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.models.base import Base, TimestampMixin, UniversalUUID

if TYPE_CHECKING:
    from app.models.evidence import Passage
    from app.models.research import ResearchRun


class Claim(Base, TimestampMixin):
    """
    Atomic factual claim extracted from passages.
    """

    __tablename__ = "claims"

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
    sub_question_id: Mapped[uuid.UUID | None] = mapped_column(
        UniversalUUID,
        ForeignKey("sub_questions.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    text: Mapped[str] = mapped_column(Text, nullable=False)
    verdict: Mapped[str] = mapped_column(
        String(50),
        default="unverified",
        nullable=False,
    )  # supported, weak, unsupported, mixed
    confidence: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)

    run: Mapped["ResearchRun"] = relationship("ResearchRun", back_populates="claims")
    evidence_links: Mapped[list["ClaimEvidence"]] = relationship(
        "ClaimEvidence",
        back_populates="claim",
        cascade="all, delete-orphan",
    )
    contradiction_links: Mapped[list["ContradictionClaim"]] = relationship(
        "ContradictionClaim",
        back_populates="claim",
        cascade="all, delete-orphan",
    )


class ClaimEvidence(Base):
    """
    Many-to-many relationship linking claims to supporting or contradicting passages.
    """

    __tablename__ = "claim_evidence"

    claim_id: Mapped[uuid.UUID] = mapped_column(
        UniversalUUID,
        ForeignKey("claims.id", ondelete="CASCADE"),
        primary_key=True,
    )
    passage_id: Mapped[uuid.UUID] = mapped_column(
        UniversalUUID,
        ForeignKey("passages.id", ondelete="CASCADE"),
        primary_key=True,
    )
    stance: Mapped[str] = mapped_column(
        String(50),
        default="supports",
        nullable=False,
    )  # supports, contradicts, neutral

    claim: Mapped["Claim"] = relationship("Claim", back_populates="evidence_links")
    passage: Mapped["Passage"] = relationship("Passage", back_populates="evidence_links")


class Contradiction(Base, TimestampMixin):
    """
    Contradiction discovered between two or more verified claims.
    """

    __tablename__ = "contradictions"

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
    explanation: Mapped[str] = mapped_column(Text, nullable=False)
    likely_cause: Mapped[str] = mapped_column(Text, nullable=False, default="")

    run: Mapped["ResearchRun"] = relationship("ResearchRun", back_populates="contradictions")
    claim_links: Mapped[list["ContradictionClaim"]] = relationship(
        "ContradictionClaim",
        back_populates="contradiction",
        cascade="all, delete-orphan",
    )


class ContradictionClaim(Base):
    """
    Associative table linking contradictions to the conflicting claims.
    """

    __tablename__ = "contradiction_claims"

    contradiction_id: Mapped[uuid.UUID] = mapped_column(
        UniversalUUID,
        ForeignKey("contradictions.id", ondelete="CASCADE"),
        primary_key=True,
    )
    claim_id: Mapped[uuid.UUID] = mapped_column(
        UniversalUUID,
        ForeignKey("claims.id", ondelete="CASCADE"),
        primary_key=True,
    )

    contradiction: Mapped["Contradiction"] = relationship(
        "Contradiction",
        back_populates="claim_links",
    )
    claim: Mapped["Claim"] = relationship(
        "Claim",
        back_populates="contradiction_links",
    )
