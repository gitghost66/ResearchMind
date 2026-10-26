"""
Run one research question end to end.

This owns the run's lifecycle — the ``research_runs`` row and its status — and
hands the research itself to the agents package's graph. M3's ``run_mvp.py``
calls it directly; M4's job runner will call it from a background task.

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
from app.repositories.research_repository import ResearchRunRepository
from app.services.agent_runtime import build_agent_runtime

logger = logging.getLogger(__name__)

__all__ = ["ResearchOutcome", "execute_research"]


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


async def execute_research(
    question: str,
    *,
    mode: str | None = None,
    user_context: dict[str, Any] | None = None,
    settings: Settings | None = None,
    session_factory: async_sessionmaker[AsyncSession] | None = None,
    runtime_kwargs: Mapping[str, Any] | None = None,
) -> ResearchOutcome:
    """
    Create the run (when persisting), execute the graph, and record the outcome.

    *runtime_kwargs* is forwarded to :func:`build_agent_runtime` — tests use it
    to inject fakes. The run is marked ``failed`` with the error if the graph
    raises, and the exception propagates.
    """
    resolved = settings or get_settings()
    resolved_mode = mode or resolved.default_mode
    started = time.monotonic()

    if session_factory is not None:
        # save_plan writes to this row, so it must exist before the graph runs.
        async with session_factory() as session:
            run = await ResearchRunRepository(session).create(
                question=question,
                mode=resolved_mode,
                user_context=dict(user_context or {}),
            )
            await ResearchRunRepository(session).update_status(run.id, "running")
            await session.commit()
            run_id = str(run.id)
    else:
        run_id = str(uuid.uuid4())

    runtime = build_agent_runtime(
        settings=resolved,
        session_factory=session_factory,
        **dict(runtime_kwargs or {}),
    )
    try:
        state = await run_research(
            runtime,
            run_id=run_id,
            question=question,
            mode=resolved_mode,
            user_context=user_context,
        )
        stored_source_ids = await runtime.store.list_source_ids(run_id=run_id)
    except BaseException as exc:
        status = "cancelled" if isinstance(exc, asyncio.CancelledError) else "failed"
        await _mark(session_factory, run_id, status, error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        await runtime.aclose()

    await _mark(session_factory, run_id, "completed")
    return ResearchOutcome(
        run_id=run_id,
        question=question,
        mode=resolved_mode,
        report=dict(state.get("report", {})),
        markdown=state.get("report_markdown", ""),
        usage=runtime.budget.summary(),
        stored_source_ids=stored_source_ids,
        seconds=time.monotonic() - started,
        persisted=session_factory is not None,
        state=dict(state),
    )


async def _mark(
    session_factory: async_sessionmaker[AsyncSession] | None,
    run_id: str,
    status: str,
    *,
    error: str | None = None,
) -> None:
    if session_factory is None:
        return
    try:
        async with session_factory() as session:
            await ResearchRunRepository(session).update_status(run_id, status, error=error)
            await session.commit()
    except Exception as exc:  # noqa: BLE001 - never mask the run's own outcome
        logger.error("Failed to mark run %s as %s: %s", run_id, status, exc)
