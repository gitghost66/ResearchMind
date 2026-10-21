"""
SQLAlchemy models for ResearchMind.
All 11 tables required by PRD Section 7.
"""

from app.models.base import Base, TimestampMixin, UniversalUUID
from app.models.claims import (
    Claim,
    ClaimEvidence,
    Contradiction,
    ContradictionClaim,
)
from app.models.evidence import Passage, Source
from app.models.research import (
    LLMUsage,
    Report,
    ResearchRun,
    RunEvent,
    SubQuestion,
)

__all__ = [
    "Base",
    "UniversalUUID",
    "TimestampMixin",
    "ResearchRun",
    "SubQuestion",
    "Source",
    "Passage",
    "Claim",
    "ClaimEvidence",
    "Contradiction",
    "ContradictionClaim",
    "Report",
    "RunEvent",
    "LLMUsage",
]
