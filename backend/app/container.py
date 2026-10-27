"""
Composition root: the long-lived pieces the API shares, built once per app.

The lifespan builds an :class:`AppServices` from ``Settings`` when a database
is configured; tests build one around a SQLite engine and fake agents and put
it on ``app.state.services`` directly.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings
from app.core.rate_limit import SlidingWindowRateLimiter
from app.services.research_service import (
    ResearchJobs,
    ResearchService,
    RuntimeKwargsFactory,
    recover_orphaned_runs,
)
from app.services.run_events import RunEventRecorder
from app.workers.job_runner import JobRunner

__all__ = ["AppServices", "build_services"]


@dataclass
class AppServices:
    settings: Settings
    session_factory: async_sessionmaker[AsyncSession]
    recorder: RunEventRecorder
    runner: JobRunner
    research: ResearchService
    request_limiter: SlidingWindowRateLimiter
    run_limiter: SlidingWindowRateLimiter

    async def startup(self) -> list[str]:
        """Settle runs a previous process left unfinished; returns their IDs."""
        if not self.settings.recover_orphaned_runs:
            return []
        return await recover_orphaned_runs(self.session_factory, self.recorder)

    async def shutdown(self) -> None:
        await self.runner.shutdown(grace=self.settings.shutdown_grace_seconds)


def build_services(
    settings: Settings,
    session_factory: async_sessionmaker[AsyncSession],
    *,
    runtime_kwargs: RuntimeKwargsFactory | None = None,
) -> AppServices:
    recorder = RunEventRecorder(session_factory)
    jobs = ResearchJobs(
        settings=settings,
        session_factory=session_factory,
        recorder=recorder,
        runtime_kwargs=runtime_kwargs,
    )
    runner = JobRunner(
        jobs.run,
        on_interrupt=jobs.interrupted,
        max_concurrent=settings.max_concurrent_runs,
        max_pending=settings.max_pending_runs,
    )
    return AppServices(
        settings=settings,
        session_factory=session_factory,
        recorder=recorder,
        runner=runner,
        research=ResearchService(
            settings=settings,
            session_factory=session_factory,
            recorder=recorder,
            runner=runner,
        ),
        request_limiter=SlidingWindowRateLimiter(
            settings.rate_limit_requests, settings.rate_limit_window_seconds
        ),
        run_limiter=SlidingWindowRateLimiter(
            settings.rate_limit_runs, settings.rate_limit_window_seconds
        ),
    )
