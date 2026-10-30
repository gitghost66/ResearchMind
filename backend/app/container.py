"""
Composition root: the long-lived pieces a process shares, built once.

* :func:`build_services` — the API's: event recorder, run queue, research
  service, rate limiters, and (``RUN_WORKER_IN_API``) a worker in the same
  process.
* :func:`build_worker_services` — a standalone worker's
  (``python -m app.workers.research_worker``).

Both listen for PostgreSQL notifications when the database is PostgreSQL and
a LISTEN-capable URL exists (``app.services.pg_listener``). Tests build the
API's services around a SQLite engine and fake agents and put them on
``app.state.services`` directly; SQLite has no LISTEN, and the in-process
notifier covers a single process.
"""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings
from app.core.rate_limit import SlidingWindowRateLimiter
from app.services.pg_listener import PgListener, listen_url_for
from app.services.research_service import (
    ResearchJobs,
    ResearchService,
    RuntimeKwargsFactory,
)
from app.services.run_events import RunEventRecorder
from app.services.run_queue import RunQueue
from app.workers.research_worker import RunWorker

__all__ = ["AppServices", "WorkerServices", "build_services", "build_worker_services"]


@dataclass
class AppServices:
    settings: Settings
    session_factory: async_sessionmaker[AsyncSession]
    recorder: RunEventRecorder
    queue: RunQueue
    research: ResearchService
    request_limiter: SlidingWindowRateLimiter
    run_limiter: SlidingWindowRateLimiter
    # A worker in this process (RUN_WORKER_IN_API), or None.
    worker: RunWorker | None = None
    listener: PgListener | None = None

    async def startup(self) -> None:
        if self.listener is not None:
            await self.listener.start()
        if self.worker is not None:
            await self.worker.start()

    async def shutdown(self) -> None:
        if self.worker is not None:
            await self.worker.shutdown(grace=self.settings.shutdown_grace_seconds)
        if self.listener is not None:
            await self.listener.stop()


@dataclass
class WorkerServices:
    settings: Settings
    recorder: RunEventRecorder
    queue: RunQueue
    worker: RunWorker
    listener: PgListener | None = None

    async def startup(self) -> None:
        if self.listener is not None:
            await self.listener.start()
        await self.worker.start()

    async def shutdown(self) -> None:
        await self.worker.shutdown(grace=self.settings.shutdown_grace_seconds)
        if self.listener is not None:
            await self.listener.stop()


def _listener(
    settings: Settings,
    session_factory: async_sessionmaker[AsyncSession],
    recorder: RunEventRecorder,
) -> PgListener | None:
    bind = session_factory.kw.get("bind")
    if bind is None or bind.dialect.name != "postgresql":
        return None
    url = listen_url_for(settings.database_url, settings.database_listen_url)
    return PgListener(url, recorder.notifier) if url else None


def _worker(
    settings: Settings,
    session_factory: async_sessionmaker[AsyncSession],
    recorder: RunEventRecorder,
    queue: RunQueue,
    runtime_kwargs: RuntimeKwargsFactory | None,
) -> RunWorker:
    jobs = ResearchJobs(
        settings=settings,
        session_factory=session_factory,
        recorder=recorder,
        runtime_kwargs=runtime_kwargs,
    )
    return RunWorker(
        settings=settings,
        session_factory=session_factory,
        recorder=recorder,
        queue=queue,
        jobs=jobs,
    )


def _queue(
    settings: Settings,
    session_factory: async_sessionmaker[AsyncSession],
    recorder: RunEventRecorder,
) -> RunQueue:
    return RunQueue(
        session_factory,
        recorder,
        lease_seconds=settings.run_lease_seconds,
        max_attempts=settings.max_run_attempts,
    )


def build_services(
    settings: Settings,
    session_factory: async_sessionmaker[AsyncSession],
    *,
    runtime_kwargs: RuntimeKwargsFactory | None = None,
) -> AppServices:
    recorder = RunEventRecorder(session_factory)
    queue = _queue(settings, session_factory, recorder)
    worker = (
        _worker(settings, session_factory, recorder, queue, runtime_kwargs)
        if settings.run_worker_in_api
        else None
    )
    return AppServices(
        settings=settings,
        session_factory=session_factory,
        recorder=recorder,
        queue=queue,
        research=ResearchService(
            settings=settings,
            session_factory=session_factory,
            recorder=recorder,
            queue=queue,
            worker=worker,
        ),
        request_limiter=SlidingWindowRateLimiter(
            settings.rate_limit_requests, settings.rate_limit_window_seconds
        ),
        run_limiter=SlidingWindowRateLimiter(
            settings.rate_limit_runs, settings.rate_limit_window_seconds
        ),
        worker=worker,
        listener=_listener(settings, session_factory, recorder),
    )


def build_worker_services(
    settings: Settings,
    session_factory: async_sessionmaker[AsyncSession],
    *,
    runtime_kwargs: RuntimeKwargsFactory | None = None,
) -> WorkerServices:
    recorder = RunEventRecorder(session_factory)
    queue = _queue(settings, session_factory, recorder)
    return WorkerServices(
        settings=settings,
        recorder=recorder,
        queue=queue,
        worker=_worker(settings, session_factory, recorder, queue, runtime_kwargs),
        listener=_listener(settings, session_factory, recorder),
    )
