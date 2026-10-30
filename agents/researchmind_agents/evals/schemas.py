"""
What the evaluation's models must return (AGENTS.md rule 3).

* ``ClosedBookAnswer`` — the baseline LLM: sentences, each with the URLs of
  pages the model says state it.
* ``RagAnswer`` — the single-agent RAG baseline: sentences citing source
  labels (``S1``, ``S2``, ...).
* ``SupportJudgement`` — the judge: for each answer sentence, the stance of
  every cited excerpt and a verdict.
* ``CoverageJudgement`` — the judge: which expected sub-topics an answer
  covers, and which sentences are off-topic.

Every field is required, as in ``state.py``, which keeps the JSON schema sent
to the provider simple.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from researchmind_agents.state import Stance

__all__ = [
    "SupportVerdict",
    "SUPPORT_VERDICTS",
    "ReferencedStatement",
    "ClosedBookAnswer",
    "CitedStatement",
    "RagAnswer",
    "EvidenceStance",
    "StatementJudgement",
    "SupportJudgement",
    "TopicCoverage",
    "CoverageJudgement",
]

# How well an answer sentence is backed by the excerpts it cites.
SupportVerdict = Literal["supported", "partial", "unsupported", "contradicted"]
SUPPORT_VERDICTS: tuple[str, ...] = ("supported", "partial", "unsupported", "contradicted")


class ReferencedStatement(BaseModel):
    text: str = Field(description="One factual sentence, without URLs or citation markers.")
    sources: list[str] = Field(
        description="Full URLs of specific web pages that state this sentence; empty if none."
    )


class ClosedBookAnswer(BaseModel):
    statements: list[ReferencedStatement]


class CitedStatement(BaseModel):
    text: str = Field(description="One factual sentence, without citation markers.")
    citations: list[str] = Field(description="Labels of the sources that support it, such as S2.")


class RagAnswer(BaseModel):
    statements: list[CitedStatement]


class EvidenceStance(BaseModel):
    evidence: str = Field(description="Evidence label such as E3.")
    stance: Stance


class StatementJudgement(BaseModel):
    statement: str = Field(description="Sentence label such as A2.")
    verdict: SupportVerdict
    evidence: list[EvidenceStance]
    reason: str = Field(description="One short sentence explaining the verdict.")


class SupportJudgement(BaseModel):
    judgements: list[StatementJudgement]


class TopicCoverage(BaseModel):
    topic: str = Field(description="Topic label such as T1.")
    covered: bool


class CoverageJudgement(BaseModel):
    topics: list[TopicCoverage]
    off_topic: list[str] = Field(
        description="Labels of sentences that do not help answer the question; empty if none."
    )
