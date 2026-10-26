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
from typing import Annotated, Any, Literal, TypedDict

from pydantic import BaseModel, Field

__all__ = [
    "SUPPORTED_MODES",
    "SubQuestionStatus",
    # LLM output schemas
    "PlannedSubQuestion",
    "PlanOutput",
    "SourceSelection",
    "ReportStatement",
    "ReportSection",
    "ReportOutput",
    # Records
    "SubQuestionItem",
    "SourceItem",
    "PassageItem",
    "CitationItem",
    "ResearchReport",
    # Graph state
    "ResearchState",
    "WebResearchTask",
    "new_research_state",
]

# M3 implements the quick pipeline only; the other modes arrive in M6.
SUPPORTED_MODES: frozenset[str] = frozenset({"quick"})

SubQuestionStatus = Literal["pending", "researching", "completed", "failed", "skipped"]


# ---------------------------------------------------------------------------
# LLM output schemas
# ---------------------------------------------------------------------------


class PlannedSubQuestion(BaseModel):
    """One sub-question as the planner proposes it."""

    text: str = Field(description="A self-contained question that can be researched on its own.")
    queries: list[str] = Field(description="Web search queries for this sub-question.")


class PlanOutput(BaseModel):
    """The planner's decomposition of the user's question."""

    objective: str = Field(description="One sentence stating what the research must establish.")
    sub_questions: list[PlannedSubQuestion]


class SourceSelection(BaseModel):
    """Search results the web researcher chose to read, by label."""

    selected: list[str] = Field(description="Result labels such as R1, R4, most useful first.")


class ReportStatement(BaseModel):
    """One sentence of the report and the sources that support it."""

    text: str = Field(description="One factual sentence, without citation markers.")
    citations: list[str] = Field(description="Source labels such as S1, S3 that support it.")


class ReportSection(BaseModel):
    heading: str
    statements: list[ReportStatement]


class ReportOutput(BaseModel):
    """What the synthesizer model returns. Markdown is rendered from this by code."""

    title: str
    executive_summary: list[ReportStatement]
    sections: list[ReportSection]
    limitations: list[str]


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


class SubQuestionItem(BaseModel):
    """A planned sub-question with the ID it is stored under."""

    id: str
    text: str
    queries: list[str] = Field(default_factory=list)
    domain: str = "web"
    status: SubQuestionStatus = "pending"


class SourceItem(BaseModel):
    """A fetched page that was stored as a source."""

    id: str
    url: str
    title: str = ""
    domain: str = ""
    content_hash: str = ""
    sub_question_id: str | None = None


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


class CitationItem(BaseModel):
    """A source label used in the report, resolved to the stored source."""

    label: str
    source_id: str
    url: str
    title: str = ""
    # Passages from this source that were shown to the model.
    passage_ids: list[str] = Field(default_factory=list)


class ResearchReport(BaseModel):
    """The final report: validated content plus provenance, stored as JSON."""

    title: str
    question: str
    mode: str
    executive_summary: list[ReportStatement] = Field(default_factory=list)
    sections: list[ReportSection] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    methodology: str = ""
    sources: list[CitationItem] = Field(default_factory=list)
    coverage: dict[str, Any] = Field(default_factory=dict)
    # Not scored until claim verification lands (M5); never an LLM number.
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
        events=[],
        llm_calls=[],
        report={},
        report_markdown="",
    )
