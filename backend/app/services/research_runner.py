"""
Run one research question end to end.

Two entry points share one execution path:

* :func:`execute_run` runs the graph for a run row that already exists. It
  does not touch the run's status — whoever owns the run's lifecycle does.
  The API's background job (``app.services.research_service``) is that owner.
* :func:`execute_research` is the one-call version for scripts
  (``run_mvp.py``): create the run, mark it running, execute, and record the
  outcome.

With a session factory the run is persisted (run row, plan, sources, passages,
events, usage, report). Without one it runs against the agents package's
in-memory store, which is enough for scripts and evals with no database.

No prompt text or agent logic here (AGENTS.md rule 2).
"""

from __future__ import annotations

import asyncio
import logging
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from researchmind_agents.graph import run_research
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings, get_settings
from app.services.agent_runtime import build_agent_runtime
from app.services.run_events import RunEventRecorder

logger = logging.getLogger(__name__)

__all__ = ["ResearchOutcome", "execute_research", "execute_run", "settings_for_run"]


@dataclass
class ResearchOutcome:
    """What a finished run produced."""

    run_id: str
    question: str
    mode: str
    report: dict[str, Any]
    markdown: str
    usage: dict[str, Any]
    stored_source_ids: set[str]
    seconds: float
    persisted: bool
    state: dict[str, Any] = field(default_factory=dict, repr=False)


def settings_for_run(settings: Settings, options: Mapping[str, Any] | None) -> Settings:
    """*settings* with the run's own options (from the API request) applied."""
    overrides: dict[str, Any] = {}
    max_sources = (options or {}).get("max_sources")
    if max_sources:
        overrides["max_sources_per_run"] = int(max_sources)
    return settings.model_copy(update=overrides) if overrides else settings


async def execute_run(
    run_id: str,
    *,
    question: str,
    mode: str,
    user_context: dict[str, Any] | None = None,
    options: Mapping[str, Any] | None = None,
    settings: Settings | None = None,
    session_factory: async_sessionmaker[AsyncSession] | None = None,
    recorder: RunEventRecorder | None = None,
    runtime_kwargs: Mapping[str, Any] | None = None,
) -> ResearchOutcome:
    """
    Execute the graph for *run_id* and return what it produced.

    The run's status is left to the caller. Exceptions — including
    cancellation — propagate after the runtime's clients are closed.
    *runtime_kwargs* is forwarded to :func:`build_agent_runtime`; tests use it
    to inject fakes.
    """
    resolved = settings_for_run(settings or get_settings(), options)
    started = time.monotonic()
    runtime = build_agent_runtime(
        settings=resolved,
        session_factory=session_factory,
        recorder=recorder,
        **dict(runtime_kwargs or {}),
    )
    try:
        state = await run_research(
            runtime,
            run_id=run_id,
            question=question,
            mode=mode,
            user_context=user_context,
        )
        stored_source_ids = await runtime.store.list_source_ids(run_id=run_id)
    finally:
        await runtime.aclose()

    return ResearchOutcome(
        run_id=run_id,
        question=question,
        mode=mode,
        report=dict(state.get("report", {})),
        markdown=state.get("report_markdown", ""),
        usage=runtime.budget.summary(),
        stored_source_ids=stored_source_ids,
        seconds=time.monotonic() - started,
        persisted=session_factory is not None,
        state=dict(state),
    )


async def execute_research(
    question: str,
    *,
    mode: str | None = None,
    user_context: dict[str, Any] | None = None,
    options: Mapping[str, Any] | None = None,
    settings: Settings | None = None,
    session_factory: async_sessionmaker[AsyncSession] | None = None,
    runtime_kwargs: Mapping[str, Any] | None = None,
) -> ResearchOutcome:
    """
    Create the run (when persisting), execute the graph, and record the outcome.

    The run is marked ``failed`` (or ``cancelled``) with the error if the graph
    raises, and the exception propagates.
    """
    resolved = settings or get_settings()
    resolved_mode = mode or resolved.default_mode

    recorder: RunEventRecorder | None = None
    if session_factory is not None:
        recorder = RunEventRecorder(session_factory)
        # save_plan writes to this row, so it must exist before the graph runs.
        run_id = await recorder.create_run(
            question=question,
            mode=resolved_mode,
            user_context=user_context,
            options=dict(options or {}),
        )
        await recorder.transition(run_id, "running", from_statuses={"queued"})
    else:
        run_id = str(uuid.uuid4())

    try:
        outcome = await execute_run(
            run_id,
            question=question,
            mode=resolved_mode,
            user_context=user_context,
            options=options,
            settings=resolved,
            session_factory=session_factory,
            recorder=recorder,
            runtime_kwargs=runtime_kwargs,
        )
    except BaseException as exc:
        status = "cancelled" if isinstance(exc, asyncio.CancelledError) else "failed"
        await _finish(recorder, run_id, status, error=f"{type(exc).__name__}: {exc}")
        raise

    await _finish(recorder, run_id, "completed")
    return outcome


async def _finish(
    recorder: RunEventRecorder | None,
    run_id: str,
    status: str,
    *,
    error: str | None = None,
) -> None:
    if recorder is None:
        return
    try:
        await recorder.transition(run_id, status, error=error)
    except Exception as exc:  # noqa: BLE001 - never mask the run's own outcome
        logger.error("Failed to mark run %s as %s: %s", run_id, status, exc)
