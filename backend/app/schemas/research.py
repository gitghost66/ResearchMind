"""
Request and response models for ``/api/v1/research`` (PRD s9).

This is the contract the frontend will be built against, so field names are
stable and every response is a declared model. Limits that depend on
``Settings`` (question length, context size, ``max_sources``) are checked in
the service, because a static ``Field`` constraint cannot read them.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "CancelResponse",
    "ErrorBody",
    "ErrorEnvelope",
    "ReportResponse",
    "ResearchCreate",
    "RunCreated",
    "RunDetail",
    "RunEventOut",
    "RunLinks",
    "RunList",
    "RunProgress",
    "RunStatus",
    "RunSummary",
    "SourceList",
    "SourceOut",
    "StepProgress",
    "StepStatus",
    "SubQuestionOut",
    "UsageBucket",
    "UsageLimits",
    "UsageResponse",
]

RunStatus = Literal["queued", "running", "completed", "failed", "cancelled"]
StepStatus = Literal["pending", "running", "completed", "incomplete", "skipped"]


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ErrorBody(BaseModel):
    code: str
    message: str
    details: Any = None


class ErrorEnvelope(BaseModel):
    error: ErrorBody


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------


class ResearchCreate(BaseModel):
    """Body of ``POST /api/v1/research``."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, description="The research question.")
    mode: str | None = Field(
        default=None, description="Research mode; the server default when omitted."
    )
    user_context: dict[str, Any] | None = Field(
        default=None, description="Free-form context about the asker, stored with the run."
    )
    max_sources: int | None = Field(
        default=None, ge=1, description="Most sources this run may store."
    )


# ---------------------------------------------------------------------------
# Runs
# ---------------------------------------------------------------------------


class RunLinks(BaseModel):
    self: str
    events: str
    report: str
    graph: str
    sources: str
    usage: str
    cancel: str

    @classmethod
    def for_run(cls, run_id: str) -> RunLinks:
        base = f"/api/v1/research/{run_id}"
        return cls(
            self=base,
            events=f"{base}/events",
            report=f"{base}/report",
            graph=f"{base}/graph",
            sources=f"{base}/sources",
            usage=f"{base}/usage",
            cancel=f"{base}/cancel",
        )


class RunCreated(BaseModel):
    run_id: str
    status: RunStatus
    links: RunLinks


class RunSummary(BaseModel):
    run_id: str
    question: str
    mode: str
    status: RunStatus
    error: str | None = None
    created_at: datetime | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None


class RunList(BaseModel):
    items: list[RunSummary]
    total: int
    limit: int
    offset: int


class SubQuestionOut(BaseModel):
    id: str
    text: str
    domain: str
    status: str


class StepProgress(BaseModel):
    """One agent's progress, read from its events."""

    agent: str
    status: StepStatus
    started: int = Field(description="agent_started events (one per branch for fan-out agents).")
    finished: int
    errors: int


class RunProgress(BaseModel):
    steps: list[StepProgress]
    sub_questions: dict[str, int] = Field(
        description="Sub-question counts by status, plus 'total'."
    )
    sources: int
    passages: int
    last_event_seq: int = Field(
        description="seq of the latest event; pass as Last-Event-ID to stream from here."
    )


class RunDetail(RunSummary):
    user_context: dict[str, Any]
    options: dict[str, Any]
    plan: dict[str, Any]
    iteration: int
    sub_questions: list[SubQuestionOut]
    progress: RunProgress
    report_available: bool
    links: RunLinks


class CancelResponse(BaseModel):
    run_id: str
    status: RunStatus


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


class RunEventOut(BaseModel):
    """One event as sent in the SSE ``data`` field."""

    seq: int
    run_id: str
    ts: str | None
    agent: str
    type: str
    payload: dict[str, Any]


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


class ReportResponse(BaseModel):
    run_id: str
    report_id: str
    version: int
    created_at: datetime | None = None
    # Null for quick-mode reports (no verified claims); never an LLM-invented number.
    overall_confidence: float | None = None
    content: dict[str, Any]
    markdown: str


class SourceOut(BaseModel):
    id: str
    url: str
    title: str
    domain_type: str
    reliability_score: float
    published_at: datetime | None = None
    passages: int
    cited: bool = Field(description="Whether the latest report cites this source.")
    created_at: datetime | None = None


class SourceList(BaseModel):
    run_id: str
    items: list[SourceOut]
    total: int


class UsageBucket(BaseModel):
    calls: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cost_usd: float = 0.0
    latency_ms: int = Field(default=0, description="Total model latency.")


class UsageLimits(BaseModel):
    max_tokens_per_run: int
    max_cost_per_run_usd: float


class UsageResponse(BaseModel):
    run_id: str
    totals: UsageBucket
    per_agent: dict[str, UsageBucket]
    per_model: dict[str, UsageBucket]
    limits: UsageLimits
