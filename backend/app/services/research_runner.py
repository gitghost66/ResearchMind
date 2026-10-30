"""
Run one research question end to end.

Two entry points share one execution path:

* :func:`execute_run` runs the graph for a run row that already exists. It
  does not touch the run's status — whoever owns the run's lifecycle does.
  The API's background job (``app.services.research_service``) is that owner.
* :func:`execute_research` is the one-call version for scripts
  (``run_mvp.py``): create the run, claim it (holding a lease, so no worker
  takes it over while the script is alive), execute, and record the outcome.

With a session factory the run is persisted (run row, plan, sources, passages,
events, usage, report) and checkpointed (``SqlCheckpointSaver``), so it can
resume. Without one it runs against the agents package's in-memory store,
which is enough for scripts and evals with no database.

A resumed run (``resume=True``) continues from its last checkpoint. Its budget
starts from what earlier attempts spent (``llm_usage``), and its source cap
from the sources already stored.

No prompt text or agent logic here (AGENTS.md rule 2).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import socket
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from researchmind_agents.graph import run_research
from researchmind_agents.runtime import AgentRuntime
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings, get_settings
from app.models.evidence import Source
from app.models.research import LLMUsage
from app.services.agent_runtime import build_agent_runtime
from app.services.checkpointer import SqlCheckpointSaver
from app.services.run_events import RunEventRecorder
from app.services.run_queue import RunQueue

logger = logging.getLogger(__name__)

__all__ = [
    "ResearchOutcome",
    "execute_research",
    "execute_run",
    "script_worker_id",
    "settings_for_run",
]


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
    checkpointer: BaseCheckpointSaver[Any] | None = None,
    resume: bool = False,
) -> ResearchOutcome:
    """
    Execute the graph for *run_id* and return what it produced.

    The run's status is left to the caller. Exceptions — including
    cancellation — propagate after the runtime's clients are closed.
    *runtime_kwargs* is forwarded to :func:`build_agent_runtime`; tests use it
    to inject fakes.

    With a session factory, checkpoints go to the database unless
    *checkpointer* is given. *resume* says the run started before: its
    earlier spend and stored sources are counted before it continues.
    """
    resolved = settings_for_run(settings or get_settings(), options)
    started = time.monotonic()
    runtime = build_agent_runtime(
        settings=resolved,
        session_factory=session_factory,
        recorder=recorder,
        **dict(runtime_kwargs or {}),
    )
    if checkpointer is None and session_factory is not None:
        checkpointer = SqlCheckpointSaver(session_factory)
    try:
        if resume and session_factory is not None:
            await _count_earlier_attempts(runtime, session_factory, run_id)
        state = await run_research(
            runtime,
            run_id=run_id,
            question=question,
            mode=mode,
            user_context=user_context,
            checkpointer=checkpointer,
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


async def _count_earlier_attempts(
    runtime: AgentRuntime,
    session_factory: async_sessionmaker[AsyncSession],
    run_id: str,
) -> None:
    """Seed a resumed run's budget and source cap with what earlier attempts used."""
    run_uuid = uuid.UUID(run_id)
    async with session_factory() as session:
        calls, tokens, cost = (
            await session.execute(
                select(
                    func.count(),
                    func.coalesce(func.sum(LLMUsage.tokens_in + LLMUsage.tokens_out), 0),
                    func.coalesce(func.sum(LLMUsage.cost_usd), 0.0),
                ).where(LLMUsage.run_id == run_uuid)
            )
        ).one()
        sources = (
            await session.execute(
                select(func.count()).select_from(Source).where(Source.run_id == run_uuid)
            )
        ).scalar_one()
    runtime.budget.seed(tokens=int(tokens), cost_usd=float(cost), calls=int(calls))
    runtime.source_slots = int(sources)


def script_worker_id() -> str:
    """The lease owner for a run a script executes in this process."""
    return f"script:{socket.gethostname()}:{os.getpid()}"


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
    lease: contextlib.AbstractAsyncContextManager[None] = contextlib.nullcontext()
    if session_factory is not None:
        recorder = RunEventRecorder(session_factory)
        queue = RunQueue(
            session_factory,
            recorder,
            lease_seconds=resolved.run_lease_seconds,
            max_attempts=resolved.max_run_attempts,
        )
        # save_plan writes to this row, so it must exist before the graph runs.
        run_id = await recorder.create_run(
            question=question,
            mode=resolved_mode,
            user_context=user_context,
            options=dict(options or {}),
        )
        # Claimed under a lease like a worker's, so a worker on the same
        # database leaves it alone while this process is alive.
        owner = script_worker_id()
        if await queue.claim_run(run_id, owner) is None:
            raise RuntimeError(f"Run {run_id} was claimed by a worker before this script")
        lease = queue.keep_lease(run_id, owner)
    else:
        run_id = str(uuid.uuid4())

    try:
        async with lease:
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
    if session_factory is not None:
        await SqlCheckpointSaver(session_factory).adelete_thread(run_id)
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
