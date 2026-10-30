"""
Shared LangGraph state and the Pydantic schemas nodes exchange.

Two kinds of schema live here:

* **LLM output schemas** (``PlanOutput``, ``SourceSelection``, ``ReportOutput``
  and their parts) — what a model must return. Every field is required, which
  keeps the JSON schema sent to the provider simple and unambiguous.
* **Records** (``SubQuestionItem``, ``SourceItem``, ``PassageItem``,
  ``CitationItem``, ``ResearchReport``) — what nodes store and pass on.

The graph state itself holds only JSON-compatible values (``model_dump`` output),
never model instances, so any LangGraph checkpointer can serialise it —
including the strict msgpack mode that refuses arbitrary classes.
"""

from __future__ import annotations

import operator
import uuid
from typing import Annotated, Any, Literal, TypedDict

from pydantic import BaseModel, Field
from pydantic.json_schema import SkipJsonSchema

from researchmind_agents.modes import SUPPORTED_MODES

__all__ = [
    "SUPPORTED_MODES",
    "SubQuestionStatus",
    "Verdict",
    "ClaimVerdict",
    "Stance",
    "ContradictionCause",
    "USABLE_VERDICTS",
    # LLM output schemas
    "PlannedSubQuestion",
    "PlanOutput",
    "GapSubQuestion",
    "GapPlanOutput",
    "SourceSelection",
    "ReportStatement",
    "ReportSection",
    "ReportOutput",
    "ExtractedClaim",
    "ClaimExtractionOutput",
    "EvidenceJudgement",
    "ClaimCheck",
    "FactCheckOutput",
    "FindingStatement",
    "FindingSection",
    "VerifiedReportOutput",
    "DecisionReportOutput",
    "DetectedContradiction",
    "ContradictionOutput",
    "SufficiencyOutput",
    # Records
    "stable_id",
    "SubQuestionItem",
    "SourceItem",
    "PassageItem",
    "EvidenceItem",
    "ClaimItem",
    "ContradictionItem",
    "GapItem",
    "CitationItem",
    "ReportEvidence",
    "ReportClaim",
    "ReportContradiction",
    "ResearchReport",
    # Graph state
    "ResearchState",
    "WebResearchTask",
    "new_research_state",
]

SubQuestionStatus = Literal["pending", "researching", "completed", "failed", "skipped"]

# What the fact checker may conclude about a claim (PRD s6.2).
Verdict = Literal["supported", "weak", "unsupported", "mixed"]
# A stored claim's verdict: "unverified" when it could not be checked.
ClaimVerdict = Literal["supported", "weak", "unsupported", "mixed", "unverified"]
# How one passage bears on a claim (the claim_evidence.stance column).
Stance = Literal["supports", "contradicts", "neutral"]
# Why two claims disagree (the contradictions.likely_cause column). Mirrors
# ``config.CONTRADICTION_CAUSES``; which causes count as resolved is a
# scoring setting (``ScoringConfig.resolved_causes``).
ContradictionCause = Literal[
    "conditions", "date", "version", "dataset", "methodology", "definition", "unexplained"
]

# Verdicts a report may rest on. Unsupported and unverified claims never
# reach the synthesizer (PRD s14: unsupported claims are dropped).
USABLE_VERDICTS: frozenset[str] = frozenset({"supported", "weak", "mixed"})


# ---------------------------------------------------------------------------
# LLM output schemas
# ---------------------------------------------------------------------------


class PlannedSubQuestion(BaseModel):
    """One sub-question as the planner proposes it."""

    text: str = Field(description="A self-contained question that can be researched on its own.")
    queries: list[str] = Field(description="Web search queries for this sub-question.")
    # Defaults to "web" so a model that omits it (or a quick-mode plan) still
    # validates; the planner node maps anything else to an allowed domain.
    domain: str = Field(
        default="web", description="Which researcher handles it: one of the listed domains."
    )


class PlanOutput(BaseModel):
    """The planner's decomposition of the user's question."""

    objective: str = Field(description="One sentence stating what the research must establish.")
    sub_questions: list[PlannedSubQuestion]


class GapSubQuestion(PlannedSubQuestion):
    """A gap-filling sub-question, and the gap it fills."""

    # Optional, like ``domain``: a sub-question that names no known gap is
    # still researched, as a new top-level sub-question.
    gap: str = Field(default="", description="Label of the gap it fills, such as G2.")


class GapPlanOutput(BaseModel):
    """New sub-questions the planner proposes in a gap-filling round."""

    sub_questions: list[GapSubQuestion]


class SourceSelection(BaseModel):
    """Search results the web researcher chose to read, by label."""

    selected: list[str] = Field(description="Result labels such as R1, R4, most useful first.")


class ReportStatement(BaseModel):
    """One sentence of the report and the sources that support it."""

    text: str = Field(description="One factual sentence, without citation markers.")
    citations: list[str] = Field(description="Source labels such as S1, S3 that support it.")
    # Verified modes: the claim labels (C1, C2, ...) the statement rests on.
    # Set by code from the model's claim citations, and hidden from the schema
    # the model is given, so a model never fills it in.
    claims: SkipJsonSchema[list[str]] = Field(default_factory=list)


class ReportSection(BaseModel):
    heading: str
    statements: list[ReportStatement]


class ReportOutput(BaseModel):
    """What the synthesizer model returns. Markdown is rendered from this by code."""

    title: str
    executive_summary: list[ReportStatement]
    sections: list[ReportSection]
    limitations: list[str]
    # Decision mode: set by code from DecisionReportOutput, and hidden from
    # the schema the model is given (quick-mode models never see it).
    recommendation: SkipJsonSchema[list[ReportStatement]] = Field(default_factory=list)


class ExtractedClaim(BaseModel):
    """One claim as the claim extractor states it."""

    text: str = Field(description="One atomic, self-contained factual statement.")
    passages: list[str] = Field(description="Labels of the passages that state it, such as P2.")


class ClaimExtractionOutput(BaseModel):
    claims: list[ExtractedClaim]


class EvidenceJudgement(BaseModel):
    """How one evidence excerpt bears on a claim."""

    evidence: str = Field(description="Evidence label such as E3.")
    stance: Stance


class ClaimCheck(BaseModel):
    """The fact checker's judgement of one claim. Confidence is computed by code."""

    claim: str = Field(description="Claim label such as C2.")
    verdict: Verdict
    evidence: list[EvidenceJudgement]
    reason: str = Field(description="One short sentence explaining the verdict.")


class FactCheckOutput(BaseModel):
    checks: list[ClaimCheck]


class FindingStatement(BaseModel):
    """One sentence of a verified report and the claims it rests on."""

    text: str = Field(description="One sentence, without citation markers.")
    claims: list[str] = Field(description="Claim labels such as C1, C4 that support it.")


class FindingSection(BaseModel):
    heading: str
    statements: list[FindingStatement]


class VerifiedReportOutput(BaseModel):
    """What the synthesizer returns in verified modes: statements cite claims."""

    title: str
    executive_summary: list[FindingStatement]
    sections: list[FindingSection]
    limitations: list[str]


class DecisionReportOutput(BaseModel):
    """Decision mode: a verified report that ends in a recommendation."""

    title: str
    executive_summary: list[FindingStatement]
    sections: list[FindingSection]
    recommendation: list[FindingStatement] = Field(
        description="What to choose and why, for the asker's situation; each sentence cites claims."
    )
    limitations: list[str]


class DetectedContradiction(BaseModel):
    """Claims that cannot all be true as stated, and why their sources differ."""

    claims: list[str] = Field(description="Labels of the conflicting claims, such as C1, C4.")
    likely_cause: ContradictionCause
    explanation: str = Field(description="One or two sentences: what differs and why.")


class ContradictionOutput(BaseModel):
    contradictions: list[DetectedContradiction]


class SufficiencyOutput(BaseModel):
    """The model's judgement of whether the verified claims answer the question."""

    sufficient: bool
    gaps: list[str] = Field(description="What is still missing; empty when sufficient.")
    reason: str = Field(description="One short sentence.")


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------

_ID_NAMESPACE = uuid.UUID("5d2f7a0e-3c1b-4e8a-9f6d-2a7c4b1e8d93")


def stable_id(*parts: object) -> str:
    """
    A UUID derived from *parts*: the same parts always give the same ID.

    Rows a node stores are keyed this way when the node can run twice for the
    same input: after a resume, an interrupted step runs again, and its rows
    then replace the first attempt's instead of duplicating them.
    """
    return str(uuid.uuid5(_ID_NAMESPACE, "".join(str(part) for part in parts)))


class SubQuestionItem(BaseModel):
    """A planned sub-question with the ID it is stored under."""

    id: str
    text: str
    queries: list[str] = Field(default_factory=list)
    domain: str = "web"
    status: SubQuestionStatus = "pending"
    # Research round that added it: 0 for the first plan, then 1, 2, ...
    iteration: int = 0
    # Gap-filling sub-questions: the sub-question whose gap this one fills.
    # Its claims count toward that sub-question's coverage.
    parent_id: str | None = None


class SourceItem(BaseModel):
    """A fetched page (or paper) that was stored as a source."""

    id: str
    url: str
    title: str = ""
    domain: str = ""
    content_hash: str = ""
    sub_question_id: str | None = None
    # Scored by reliability.py when the source is stored (PRD s8).
    domain_type: str = "unknown"
    reliability_score: float = 0.5
    # ISO 8601, or None when unknown. A string keeps graph state JSON-only.
    published_at: str | None = None


class PassageItem(BaseModel):
    """
    A chunk of a source's text.

    The embedding is deliberately absent: vectors go straight to the store and
    would otherwise bloat every checkpoint.
    """

    id: str
    source_id: str
    sub_question_id: str | None = None
    text: str
    chunk_index: int = 0


class EvidenceItem(BaseModel):
    """A passage linked to a claim, and how it bears on the claim."""

    passage_id: str
    source_id: str
    stance: Stance = "supports"
    # The part of the passage the fact checker read.
    excerpt: str = ""


class ClaimItem(BaseModel):
    """
    An atomic claim and its evidence.

    Every claim links to at least one passage: the extractor drops claims
    without a valid passage label, and ``evidence`` starts as those passages.
    """

    id: str
    text: str
    sub_question_id: str | None = None
    # Passages the claim was extracted from.
    passage_ids: list[str]
    evidence: list[EvidenceItem] = Field(default_factory=list)
    verdict: ClaimVerdict = "unverified"
    # Computed by reliability.claim_confidence, never by a model.
    confidence: float = 0.0
    # The fact checker's one-line reason.
    reason: str = ""
    # Research round it was extracted in; each round checks only its own.
    iteration: int = 0


class ContradictionItem(BaseModel):
    """Claims that cannot all be true as stated (PRD s6.2)."""

    id: str
    # Two or more claim IDs.
    claim_ids: list[str]
    explanation: str
    likely_cause: ContradictionCause
    # Set by code from the cause (``ScoringConfig.resolved_causes``): the
    # claims differ in scope (date, version, ...), so both can hold.
    # Unresolved contradictions lower the report's confidence.
    resolved: bool = False
    iteration: int = 0


class GapItem(BaseModel):
    """Something the sufficiency check found missing."""

    label: str
    description: str
    # The (top-level) sub-question it concerns, if any.
    sub_question_id: str | None = None


class CitationItem(BaseModel):
    """A source label used in the report, resolved to the stored source."""

    label: str
    source_id: str
    url: str
    title: str = ""
    # Passages from this source that were shown to the model.
    passage_ids: list[str] = Field(default_factory=list)


class ReportEvidence(BaseModel):
    """One passage behind a key finding."""

    passage_id: str
    source_label: str
    stance: Stance
    excerpt: str = ""


class ReportClaim(BaseModel):
    """A verified claim as the report lists it under key findings."""

    label: str
    claim_id: str
    text: str
    sub_question_id: str | None = None
    verdict: ClaimVerdict
    confidence: float
    # Source labels: supporting sources, plus contradicting ones for "mixed".
    citations: list[str] = Field(default_factory=list)
    evidence: list[ReportEvidence] = Field(default_factory=list)


class ReportContradiction(BaseModel):
    """A contradiction between key findings, as the report lists it."""

    label: str
    contradiction_id: str
    # Labels of the key findings (C1, ...) that conflict.
    claims: list[str]
    # Source labels behind those findings.
    citations: list[str] = Field(default_factory=list)
    likely_cause: str
    explanation: str
    resolved: bool


class ResearchReport(BaseModel):
    """The final report: validated content plus provenance, stored as JSON."""

    title: str
    question: str
    mode: str
    executive_summary: list[ReportStatement] = Field(default_factory=list)
    sections: list[ReportSection] = Field(default_factory=list)
    # Decision mode: what to choose, each statement resting on claims.
    recommendation: list[ReportStatement] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    methodology: str = ""
    sources: list[CitationItem] = Field(default_factory=list)
    coverage: dict[str, Any] = Field(default_factory=dict)
    # Verified modes: the claims the report rests on, with verdicts and
    # confidence (PRD s9 key_findings/evidence). Empty in quick mode.
    key_findings: list[ReportClaim] = Field(default_factory=list)
    # Contradictions between key findings (PRD s9 contradictory_evidence).
    contradictory_evidence: list[ReportContradiction] = Field(default_factory=list)
    # Claim counts by stage and verdict; empty in quick mode.
    verification: dict[str, Any] = Field(default_factory=dict)
    # reliability.overall_confidence; None when no claims were verified
    # (quick mode). Never an LLM number.
    overall_confidence: float | None = None
    # "llm" (normal), "extractive" (no LLM: budget or model failure), "empty".
    generation: Literal["llm", "extractive", "empty"] = "llm"
    dropped_statements: int = 0
    prompt_version: str = ""


# ---------------------------------------------------------------------------
# Graph state
# ---------------------------------------------------------------------------


class ResearchState(TypedDict, total=False):
    """
    The state LangGraph threads through the graph.

    List fields written by parallel branches use ``operator.add`` so each
    branch's results are appended instead of overwriting one another.
    """

    run_id: str
    question: str
    mode: str
    user_context: dict[str, Any]
    iteration: int
    max_iterations: int

    plan: dict[str, Any]
    sub_questions: list[dict[str, Any]]

    sources: Annotated[list[dict[str, Any]], operator.add]
    passages: Annotated[list[dict[str, Any]], operator.add]
    # One entry per researched sub-question: status, counts, error.
    research_results: Annotated[list[dict[str, Any]], operator.add]
    # ClaimItem dumps, every round's. One writer at a time: the claim
    # extractor adds the round's claims and the fact checker replaces them
    # with the verified versions.
    claims: list[dict[str, Any]]
    # Passages the claim extractor has read, so a later round reads new ones.
    extracted_passage_ids: list[str]
    # ContradictionItem dumps, every round's (the detector's alone).
    contradictions: list[dict[str, Any]]
    # The latest sufficiency check: status, decision, gaps, coverage.
    sufficiency: dict[str, Any]
    events: Annotated[list[dict[str, Any]], operator.add]
    llm_calls: Annotated[list[dict[str, Any]], operator.add]

    report: dict[str, Any]
    report_markdown: str


class WebResearchTask(TypedDict):
    """Payload for one web-researcher branch (sent via ``Send``)."""

    run_id: str
    question: str
    sub_question: dict[str, Any]


def new_research_state(
    *,
    run_id: str,
    question: str,
    mode: str = "quick",
    user_context: dict[str, Any] | None = None,
    max_iterations: int = 1,
) -> ResearchState:
    """Initial state for a run."""
    return ResearchState(
        run_id=run_id,
        question=question,
        mode=mode,
        user_context=dict(user_context or {}),
        iteration=0,
        max_iterations=max_iterations,
        plan={},
        sub_questions=[],
        sources=[],
        passages=[],
        research_results=[],
        claims=[],
        extracted_passage_ids=[],
        contradictions=[],
        sufficiency={},
        events=[],
        llm_calls=[],
        report={},
        report_markdown="",
    )
