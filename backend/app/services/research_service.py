"""
Research runs as the API and the workers see them.

* :class:`ResearchJobs` — executes one claimed run: read it, close the steps
  an interruption left open (on a resume), run the graph from its last
  checkpoint, and record the outcome as its owner. Used by
  :class:`~app.workers.research_worker.RunWorker`.
* :class:`ResearchService` — what the routes call: create, read, cancel,
  resume, and delete runs, and read their report, evidence graph, sources, and
  usage. Creating a run only queues it; a worker claims it from the database.

No prompt text or agent logic here (AGENTS.md rule 2): the graph runs inside
``execute_run``; this module only owns the run's lifecycle and its reads.
"""

from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections.abc import Callable, Mapping
from typing import Any, Protocol

from researchmind_agents.graph import pipeline_steps
from researchmind_agents.state import SUPPORTED_MODES
from sqlalchemy import delete, func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings
from app.core.errors import ApiError
from app.models.evidence import Passage, Source
from app.models.research import LLMUsage, Report, ResearchRun, RunEvent, SubQuestion
from app.schemas.graph import EvidenceGraph
from app.schemas.research import (
    CancelResponse,
    ReportResponse,
    ResearchCreate,
    RunCreated,
    RunDetail,
    RunLinks,
    RunList,
    RunProgress,
    RunSummary,
    SourceList,
    SourceOut,
    StepProgress,
    StepStatus,
    SubQuestionOut,
    UsageBucket,
    UsageLimits,
    UsageResponse,
)
from app.services.agent_runtime import build_agent_config
from app.services.checkpointer import SqlCheckpointSaver
from app.services.evidence_graph import build_evidence_graph, load_graph_data
from app.services.research_runner import execute_run
from app.services.run_events import (
    ACTIVE_STATUSES,
    RUNNER_AGENT,
    TERMINAL_STATUSES,
    RunEventRecorder,
)
from app.services.run_queue import Claim, RunQueue

logger = logging.getLogger(__name__)

__all__ = [
    "INTERRUPTED_STEP",
    "LocalWorker",
    "ResearchJobs",
    "ResearchService",
    "RuntimeKwargsFactory",
    "build_steps",
]

# Builds per-run overrides for build_agent_runtime (tests inject fakes here).
RuntimeKwargsFactory = Callable[[Settings], Mapping[str, Any]]

_STEP_EVENTS = ("agent_started", "agent_finished", "error")

# Payload status of the agent_finished events that close steps an
# interruption cut short.
INTERRUPTED_STEP = "interrupted"

# Recording a run's outcome is retried this often, waiting 0.5s, 1s, ... between.
_FINISH_ATTEMPTS = 3
_FINISH_RETRY_DELAY = 0.5

# Retry-After for a full queue: runs take minutes, so any short wait is a guess.
_QUEUE_FULL_RETRY_AFTER_SECONDS = 30


def _uuid(value: str | uuid.UUID) -> uuid.UUID:
    return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))


class LocalWorker(Protocol):
    """The worker running in this process, if any (it can stop a run at once)."""

    def is_active(self, run_id: str) -> bool: ...

    async def stop(self, run_id: str, *, wait: float) -> bool: ...


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------


class ResearchJobs:
    """Executes claimed runs; one instance serves every run of a worker."""

    def __init__(
        self,
        *,
        settings: Settings,
        session_factory: async_sessionmaker[AsyncSession],
        recorder: RunEventRecorder,
        runtime_kwargs: RuntimeKwargsFactory | None = None,
    ) -> None:
        self._settings = settings
        self._session_factory = session_factory
        self._recorder = recorder
        self._runtime_kwargs = runtime_kwargs
        self.checkpoints = SqlCheckpointSaver(session_factory)

    async def execute(self, claim: Claim) -> tuple[str, str | None]:
        """
        Run the graph for a claimed run; returns the status to record
        (``completed`` or ``failed``) and its error.

        Any error, including one reading the run, ends it as failed, with its
        checkpoints kept so it can be resumed. Cancellation propagates.
        """
        run_id = claim.run_id
        try:
            async with self._session_factory() as session:
                run = await session.get(ResearchRun, _uuid(run_id))
                if run is None:
                    return "completed", None  # deleted meanwhile; finishing is a no-op
                question, mode = run.question, run.mode
                user_context, options = dict(run.user_context or {}), dict(run.options or {})

            if claim.resumed:
                await self._close_interrupted_steps(run_id)
            await execute_run(
                run_id,
                question=question,
                mode=mode,
                user_context=user_context,
                options=options,
                settings=self._settings,
                session_factory=self._session_factory,
                recorder=self._recorder,
                runtime_kwargs=(
                    self._runtime_kwargs(self._settings) if self._runtime_kwargs else None
                ),
                checkpointer=self.checkpoints,
                resume=claim.resumed,
            )
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # noqa: BLE001 - recorded on the run
            logger.warning("Run %s failed: %s", run_id, exc, extra={"run_id": run_id})
            return "failed", f"{type(exc).__name__}: {exc}"
        return "completed", None

    async def finish(
        self,
        run_id: str,
        status: str,
        *,
        error: str | None = None,
        owner: str | None = None,
    ) -> bool:
        """
        Record the outcome as *owner*, retrying briefly: a run left active
        never ends its streams (its lease lapsing is the backstop).
        """
        for attempt in range(1, _FINISH_ATTEMPTS + 1):
            try:
                return await self._recorder.transition(
                    run_id, status, error=error, from_statuses={"running"}, owner=owner
                )
            except Exception as exc:  # noqa: BLE001 - the lease is the backstop
                if attempt == _FINISH_ATTEMPTS:
                    logger.error("Failed to mark run %s as %s: %s", run_id, status, exc)
                    return False
                await asyncio.sleep(_FINISH_RETRY_DELAY * attempt)
        return False

    async def drop_checkpoints(self, run_id: str) -> None:
        try:
            await self.checkpoints.adelete_thread(run_id)
        except Exception as exc:  # noqa: BLE001 - only space is lost
            logger.warning("Could not delete checkpoints of run %s: %s", run_id, exc)

    async def _close_interrupted_steps(self, run_id: str) -> None:
        """
        Close the steps the interruption left open, so progress counts stay
        balanced: one ``agent_finished`` (status ``interrupted``) per
        ``agent_started`` that never finished. The resumed run then starts
        those steps again.
        """
        async with self._session_factory() as session:
            rows = await session.execute(
                select(RunEvent.agent, RunEvent.type, func.count())
                .where(
                    RunEvent.run_id == _uuid(run_id),
                    RunEvent.type.in_(("agent_started", "agent_finished")),
                )
                .group_by(RunEvent.agent, RunEvent.type)
            )
            counts: dict[str, dict[str, int]] = {}
            for agent, event_type, count in rows.all():
                counts.setdefault(agent, {})[event_type] = int(count)
        for agent, c in counts.items():
            for _ in range(c.get("agent_started", 0) - c.get("agent_finished", 0)):
                await self._recorder.record(
                    run_id=run_id,
                    agent=agent,
                    event_type="agent_finished",
                    payload={"status": INTERRUPTED_STEP},
                )


# ---------------------------------------------------------------------------
# Progress
# ---------------------------------------------------------------------------


def build_steps(
    mode: str,
    counts: Mapping[str, Mapping[str, int]],
    first_seq: Mapping[str, int],
    *,
    active: bool,
) -> list[StepProgress]:
    """
    Per-agent progress from event counts.

    Steps are the mode's pipeline in order, then any other agent that emitted
    events. A step is ``completed`` once every branch that started finished and
    a later step has begun (or the run ended) — a fan-out agent whose next
    branch has not started yet is still ``running``.
    """
    agents = list(pipeline_steps(mode))
    agents += sorted(
        (a for a in counts if a not in agents and a != RUNNER_AGENT),
        key=lambda a: first_seq.get(a, 0),
    )
    steps: list[StepProgress] = []
    for index, agent in enumerate(agents):
        c = counts.get(agent, {})
        started, finished = c.get("agent_started", 0), c.get("agent_finished", 0)
        later_started = any(counts.get(a, {}).get("agent_started", 0) for a in agents[index + 1 :])
        status: StepStatus
        if started == 0:
            status = "pending" if active else "skipped"
        elif finished >= started and (later_started or not active):
            status = "completed"
        else:
            status = "running" if active else "incomplete"
        steps.append(
            StepProgress(
                agent=agent,
                status=status,
                started=started,
                finished=finished,
                errors=c.get("error", 0),
            )
        )
    return steps


# ---------------------------------------------------------------------------
# API operations
# ---------------------------------------------------------------------------


class ResearchService:
    """Everything the research routes do, given the app's shared pieces."""

    def __init__(
        self,
        *,
        settings: Settings,
        session_factory: async_sessionmaker[AsyncSession],
        recorder: RunEventRecorder,
        queue: RunQueue,
        worker: LocalWorker | None = None,
    ) -> None:
        self.settings = settings
        self.session_factory = session_factory
        self.recorder = recorder
        self.queue = queue
        self.worker = worker
        self.checkpoints = SqlCheckpointSaver(session_factory)

    # -- create -----------------------------------------------------------

    def validate(self, body: ResearchCreate) -> tuple[str, str, dict[str, Any], dict[str, Any]]:
        """Apply the settings-dependent limits; returns question, mode, context, options."""
        settings = self.settings
        errors: list[dict[str, Any]] = []

        question = body.question.strip()
        if not question:
            errors.append(_field_error("question", "must not be blank", "value_error"))
        elif len(question) > settings.max_question_chars:
            errors.append(
                _field_error(
                    "question",
                    f"must be at most {settings.max_question_chars} characters",
                    "string_too_long",
                )
            )

        mode = body.mode or settings.default_mode
        if mode not in SUPPORTED_MODES:
            errors.append(
                _field_error(
                    "mode",
                    f"mode {mode!r} is not available yet; supported: "
                    + ", ".join(sorted(SUPPORTED_MODES)),
                    "unsupported_mode",
                )
            )

        user_context = dict(body.user_context or {})
        size = len(json.dumps(user_context, separators=(",", ":")).encode())
        if size > settings.max_user_context_bytes:
            errors.append(
                _field_error(
                    "user_context",
                    f"must be at most {settings.max_user_context_bytes} bytes as JSON",
                    "too_large",
                )
            )

        options: dict[str, Any] = {}
        if body.max_sources is not None:
            if body.max_sources > settings.max_sources_limit:
                errors.append(
                    _field_error(
                        "max_sources",
                        f"must be at most {settings.max_sources_limit}",
                        "less_than_equal",
                    )
                )
            options["max_sources"] = body.max_sources

        if errors:
            raise ApiError(422, "validation_error", "Invalid request", errors)
        return question, mode, user_context, options

    async def create(self, body: ResearchCreate) -> RunCreated:
        """Queue a run; a worker claims it (one in this process, if enabled, at once)."""
        question, mode, user_context, options = self.validate(body)
        await self._check_capacity()
        run_id = await self.recorder.create_run(
            question=question, mode=mode, user_context=user_context, options=options
        )
        return RunCreated(run_id=run_id, status="queued", links=RunLinks.for_run(run_id))

    async def _check_capacity(self) -> None:
        """
        503 when ``MAX_PENDING_RUNS`` runs are queued or running. Two requests
        racing can each see one free place, so the cap can be passed by a run
        or two; it bounds the queue, it does not meter it.
        """
        counts = await self.queue.counts()
        if counts["queued"] + counts["running"] >= max(1, self.settings.max_pending_runs):
            raise _busy()

    # -- read -------------------------------------------------------------

    async def get(self, session: AsyncSession, run_id: uuid.UUID) -> ResearchRun:
        run = await session.get(ResearchRun, run_id)
        if run is None:
            raise ApiError(404, "run_not_found", f"No research run {run_id}")
        return run

    async def list_runs(
        self,
        session: AsyncSession,
        *,
        limit: int,
        offset: int,
        status: str | None = None,
        mode: str | None = None,
    ) -> RunList:
        filters = []
        if status:
            filters.append(ResearchRun.status == status)
        if mode:
            filters.append(ResearchRun.mode == mode)
        total = int(
            (
                await session.execute(select(func.count()).select_from(ResearchRun).where(*filters))
            ).scalar_one()
        )
        rows = (
            await session.execute(
                select(ResearchRun)
                .where(*filters)
                .order_by(ResearchRun.created_at.desc(), ResearchRun.id)
                .offset(offset)
                .limit(limit)
            )
        ).scalars()
        return RunList(items=[_summary(r) for r in rows], total=total, limit=limit, offset=offset)

    async def detail(self, session: AsyncSession, run: ResearchRun) -> RunDetail:
        sub_questions = (
            await session.execute(
                select(SubQuestion)
                .where(SubQuestion.run_id == run.id)
                .order_by(SubQuestion.created_at, SubQuestion.id)
            )
        ).scalars().all()

        counts: dict[str, dict[str, int]] = {}
        first_seq: dict[str, int] = {}
        event_rows = await session.execute(
            select(RunEvent.agent, RunEvent.type, func.count(), func.min(RunEvent.seq))
            .where(RunEvent.run_id == run.id, RunEvent.type.in_(_STEP_EVENTS))
            .group_by(RunEvent.agent, RunEvent.type)
        )
        for agent, event_type, count, seq in event_rows.all():
            counts.setdefault(agent, {})[event_type] = int(count)
            first_seq[agent] = min(first_seq.get(agent, seq), seq)

        last_seq = (
            await session.execute(
                select(func.coalesce(func.max(RunEvent.seq), 0)).where(RunEvent.run_id == run.id)
            )
        ).scalar_one()
        sources = (
            await session.execute(
                select(func.count()).select_from(Source).where(Source.run_id == run.id)
            )
        ).scalar_one()
        passages = (
            await session.execute(
                select(func.count())
                .select_from(Passage)
                .join(Source, Passage.source_id == Source.id)
                .where(Source.run_id == run.id)
            )
        ).scalar_one()
        has_report = (
            await session.execute(select(Report.id).where(Report.run_id == run.id).limit(1))
        ).first() is not None

        by_status: dict[str, int] = {"total": len(sub_questions)}
        for sq in sub_questions:
            by_status[sq.status] = by_status.get(sq.status, 0) + 1

        return RunDetail(
            **_summary(run).model_dump(),
            user_context=dict(run.user_context or {}),
            options=dict(run.options or {}),
            plan=dict(run.plan or {}),
            iteration=run.iteration,
            attempts=run.attempts,
            sub_questions=[
                SubQuestionOut(id=str(sq.id), text=sq.text, domain=sq.domain, status=sq.status)
                for sq in sub_questions
            ],
            progress=RunProgress(
                steps=build_steps(
                    run.mode, counts, first_seq, active=run.status in ACTIVE_STATUSES
                ),
                sub_questions=by_status,
                sources=int(sources),
                passages=int(passages),
                last_event_seq=int(last_seq),
            ),
            report_available=has_report,
            links=RunLinks.for_run(str(run.id)),
        )

    async def report(self, session: AsyncSession, run: ResearchRun) -> ReportResponse:
        report = await _latest_report(session, run.id)
        if report is None:
            if run.status in ACTIVE_STATUSES:
                raise ApiError(
                    409,
                    "report_not_ready",
                    "The run has not produced a report yet",
                    {"status": run.status},
                )
            raise ApiError(
                404, "report_not_found", "This run has no report", {"status": run.status}
            )
        return ReportResponse(
            run_id=str(run.id),
            report_id=str(report.id),
            version=report.version,
            created_at=report.created_at,
            overall_confidence=(report.content or {}).get("overall_confidence"),
            content=dict(report.content or {}),
            markdown=report.markdown,
        )

    async def graph(self, session: AsyncSession, run: ResearchRun) -> EvidenceGraph:
        """The run's evidence graph, from what it has stored so far (any status)."""
        scoring = build_agent_config(self.settings).scoring
        data = await load_graph_data(session, run)
        return build_evidence_graph(data, is_resolved=scoring.is_resolved)

    async def sources(self, session: AsyncSession, run: ResearchRun) -> SourceList:
        report = await _latest_report(session, run.id)
        cited = {
            str(item.get("source_id"))
            for item in ((report.content or {}).get("sources") or [] if report else [])
            if isinstance(item, Mapping)
        }
        rows = await session.execute(
            select(Source, func.count(Passage.id))
            .outerjoin(Passage, Passage.source_id == Source.id)
            .where(Source.run_id == run.id)
            .group_by(Source.id)
            .order_by(Source.reliability_score.desc(), Source.created_at, Source.id)
        )
        items = [
            SourceOut(
                id=str(source.id),
                url=source.url,
                title=source.title,
                domain_type=source.domain_type,
                reliability_score=source.reliability_score,
                published_at=source.published_at,
                passages=int(passage_count),
                cited=str(source.id) in cited,
                created_at=source.created_at,
            )
            for source, passage_count in rows.all()
        ]
        return SourceList(run_id=str(run.id), items=items, total=len(items))

    async def usage(self, session: AsyncSession, run: ResearchRun) -> UsageResponse:
        rows = await session.execute(
            select(
                LLMUsage.agent,
                LLMUsage.model,
                func.count(),
                func.coalesce(func.sum(LLMUsage.tokens_in), 0),
                func.coalesce(func.sum(LLMUsage.tokens_out), 0),
                func.coalesce(func.sum(LLMUsage.cost_usd), 0.0),
                func.coalesce(func.sum(LLMUsage.latency_ms), 0),
            )
            .where(LLMUsage.run_id == run.id)
            .group_by(LLMUsage.agent, LLMUsage.model)
        )
        totals = UsageBucket()
        per_agent: dict[str, UsageBucket] = {}
        per_model: dict[str, UsageBucket] = {}
        for agent, model, calls, tokens_in, tokens_out, cost, latency in rows.all():
            for bucket in (
                totals,
                per_agent.setdefault(agent, UsageBucket()),
                per_model.setdefault(model, UsageBucket()),
            ):
                bucket.calls += int(calls)
                bucket.tokens_in += int(tokens_in)
                bucket.tokens_out += int(tokens_out)
                bucket.cost_usd = round(bucket.cost_usd + float(cost), 10)
                bucket.latency_ms += int(latency)
        return UsageResponse(
            run_id=str(run.id),
            totals=totals,
            per_agent=per_agent,
            per_model=per_model,
            limits=UsageLimits(
                max_tokens_per_run=self.settings.max_tokens_per_run,
                max_cost_per_run_usd=self.settings.max_cost_per_run_usd,
            ),
        )

    # -- control ----------------------------------------------------------

    async def cancel(self, run_id: uuid.UUID) -> CancelResponse:
        """
        Mark the run cancelled. Its worker, in whatever process, is notified
        and stops; one in this process is waited for (``CANCEL_WAIT_SECONDS``).
        """
        key = str(run_id)
        async with self.session_factory() as session:
            run = await self.get(session, run_id)
            if run.status in TERMINAL_STATUSES:
                raise ApiError(
                    409, "run_not_active", f"The run is already {run.status}", {"status": run.status}
                )

        await self.recorder.transition(key, "cancelled")
        if self.worker is not None:
            await self.worker.stop(key, wait=self.settings.cancel_wait_seconds)

        async with self.session_factory() as session:
            run = await self.get(session, run_id)
        if run.status in TERMINAL_STATUSES and run.status != "cancelled":
            # It finished (or failed) on its own before the cancel took effect.
            raise ApiError(
                409, "run_not_active", f"The run is already {run.status}", {"status": run.status}
            )
        return CancelResponse(run_id=key, status=run.status)  # type: ignore[arg-type]

    async def resume(self, run_id: uuid.UUID) -> RunCreated:
        """
        Queue a failed run again. A worker resumes it from its last
        checkpoint: steps that finished are not run again, and its budget
        counts what earlier attempts spent.
        """
        key = str(run_id)
        async with self.session_factory() as session:
            run = await self.get(session, run_id)
        if run.status != "failed":
            raise ApiError(
                409,
                "run_not_resumable",
                f"Only a failed run can be resumed; this one is {run.status}",
                {"status": run.status},
            )
        await self._check_capacity()
        if not await self.queue.requeue_failed(key):
            async with self.session_factory() as session:
                run = await self.get(session, run_id)
            raise ApiError(
                409,
                "run_not_resumable",
                f"Only a failed run can be resumed; this one is {run.status}",
                {"status": run.status},
            )
        return RunCreated(run_id=key, status="queued", links=RunLinks.for_run(key))

    async def delete(self, run_id: uuid.UUID) -> None:
        key = str(run_id)
        async with self.session_factory() as session:
            await self.get(session, run_id)
        # Cancelling first tells the run's worker, wherever it is, to stop.
        await self.recorder.transition(key, "cancelled")
        if self.worker is not None:
            await self.worker.stop(key, wait=self.settings.cancel_wait_seconds)
        async with self.session_factory() as session:
            # Foreign keys cascade to every child table (migration 0001).
            await session.execute(delete(ResearchRun).where(ResearchRun.id == run_id))
            await session.commit()
        await self.checkpoints.adelete_thread(key)
        self.recorder.notifier.notify(key)  # open streams notice and close


def _field_error(field: str, message: str, kind: str) -> dict[str, Any]:
    return {"loc": ["body", field], "msg": message, "type": kind}


def _busy() -> ApiError:
    return ApiError(
        503,
        "queue_full",
        "Too many research runs are queued; try again shortly",
        headers={"Retry-After": str(_QUEUE_FULL_RETRY_AFTER_SECONDS)},
    )


def _summary(run: ResearchRun) -> RunSummary:
    return RunSummary(
        run_id=str(run.id),
        question=run.question,
        mode=run.mode,
        status=run.status,  # type: ignore[arg-type]
        error=run.error,
        created_at=run.created_at,
        started_at=run.started_at,
        finished_at=run.finished_at,
    )


async def _latest_report(session: AsyncSession, run_id: uuid.UUID) -> Report | None:
    return (
        await session.execute(
            select(Report)
            .where(Report.run_id == run_id)
            .order_by(Report.version.desc(), Report.created_at.desc())
            .limit(1)
        )
    ).scalar_one_or_none()
