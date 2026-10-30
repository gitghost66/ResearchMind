"""
The research worker: claims queued runs from the database and executes them.

Run it as its own process (``python -m app.workers.research_worker``), as many
as you like, or inside the API process (``RUN_WORKER_IN_API=true``, the
default, so one container is enough). Either way it is the same
:class:`RunWorker`:

* **Claim** — when told a run was queued (a notification, from this process or
  over PostgreSQL ``LISTEN``), when a slot frees up, and every
  ``WORKER_POLL_SECONDS``, it fills its free slots (``MAX_CONCURRENT_RUNS``)
  through :meth:`RunQueue.claim_next`: runs whose worker died first, resumed
  from their checkpoints, then queued runs.
* **Lease** — every ``RUN_LEASE_SECONDS / 3`` it renews its runs' leases. A run
  it no longer owns (cancelled, deleted, taken over) is stopped.
* **Cancel** — a run cancelled through the API anywhere is stopped as soon as
  the status notification arrives, else at the next renewal.
* **Shutdown** — runs are interrupted and released, not failed: another worker
  (or this one, restarted) resumes them at once. A worker killed outright
  cannot release anything; its leases lapse and its runs are resumed then.

The ``JobRunner`` underneath keeps the tasks and interrupts them cleanly.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import signal
import socket
import uuid
from collections.abc import Coroutine
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.config import Settings
from app.core.logging import bind_log_context, log_context
from app.models.research import ResearchRun
from app.services.research_service import ResearchJobs
from app.services.run_events import TERMINAL_STATUSES, RunEventRecorder
from app.services.run_queue import Claim, RunQueue
from app.workers.job_runner import SHUTDOWN, JobRunner

logger = logging.getLogger(__name__)

__all__ = ["RunWorker", "default_worker_id", "main"]


def default_worker_id() -> str:
    """Host, process, and a random suffix: unique even across restarts."""
    return f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"


class RunWorker:
    def __init__(
        self,
        *,
        settings: Settings,
        session_factory: async_sessionmaker[AsyncSession],
        recorder: RunEventRecorder,
        queue: RunQueue,
        jobs: ResearchJobs,
        worker_id: str | None = None,
    ) -> None:
        self.worker_id = worker_id or default_worker_id()
        self._settings = settings
        self._session_factory = session_factory
        self._recorder = recorder
        self._queue = queue
        self._jobs = jobs
        self.capacity = max(1, settings.max_concurrent_runs)
        self._runner = JobRunner(
            self._execute,
            on_interrupt=self._interrupted,
            max_concurrent=self.capacity,
            max_pending=self.capacity,
        )
        self._claims: dict[str, Claim] = {}
        # Runs whose outcome this worker is recording: their own status
        # notification must not cancel them.
        self._settling: set[str] = set()
        self._wake = asyncio.Event()
        self._loops: list[asyncio.Task[None]] = []
        self._background: set[asyncio.Task[Any]] = set()
        self._unsubscribe: Any = None
        self._closed = False

    # -- state ----------------------------------------------------------

    @property
    def running_count(self) -> int:
        return self._runner.running_count

    def is_active(self, run_id: str) -> bool:
        return self._runner.is_active(run_id)

    # -- lifecycle ------------------------------------------------------

    async def start(self) -> None:
        if self._loops:
            return
        self._unsubscribe = self._recorder.notifier.subscribe_status(self._on_status)
        self._loops = [
            asyncio.create_task(self._claim_loop(), name=f"worker-claims:{self.worker_id}"),
            asyncio.create_task(self._lease_loop(), name=f"worker-leases:{self.worker_id}"),
        ]
        logger.info(
            "Worker %s started (%d slots)", self.worker_id, self.capacity,
            extra={"worker_id": self.worker_id},
        )

    async def shutdown(self, *, grace: float) -> None:
        """Stop claiming, interrupt and release every run, wait up to *grace* seconds."""
        self._closed = True
        if self._unsubscribe is not None:
            self._unsubscribe()
            self._unsubscribe = None
        for task in self._loops:
            task.cancel()
        for task in self._loops:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._loops = []
        await self._runner.shutdown(grace=grace)
        if self._background:
            await asyncio.wait(list(self._background), timeout=grace)

    def wake(self) -> None:
        self._wake.set()

    async def stop(self, run_id: str, *, wait: float) -> bool:
        """Interrupt *run_id* if this worker runs it; ``False`` if it does not."""
        return await self._runner.cancel(run_id, wait=wait)

    async def wait_idle(self) -> None:
        await self._runner.wait_idle()

    # -- claiming -------------------------------------------------------

    async def fill(self) -> list[str]:
        """Claim runs until every slot is busy or nothing is left; returns their IDs."""
        claimed: list[str] = []
        while not self._closed and self._runner.pending_count < self.capacity:
            claim = await self._queue.claim_next(self.worker_id)
            if claim is None:
                break
            self._claims[claim.run_id] = claim
            self._runner.submit(claim.run_id)
            claimed.append(claim.run_id)
        return claimed

    async def _claim_loop(self) -> None:
        while not self._closed:
            self._wake.clear()
            try:
                await self.fill()
            except Exception as exc:  # noqa: BLE001 - e.g. the database is down; retry
                logger.warning("Could not claim runs: %s", exc, extra={"worker_id": self.worker_id})
            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(
                    self._wake.wait(), timeout=max(0.05, self._settings.worker_poll_seconds)
                )

    async def _lease_loop(self) -> None:
        interval = max(0.05, self._queue.lease_seconds / 3)
        while not self._closed:
            await asyncio.sleep(interval)
            try:
                lost = await self._queue.renew(self.worker_id, list(self._claims))
            except Exception as exc:  # noqa: BLE001 - try again next time
                logger.warning("Could not renew leases: %s", exc, extra={"worker_id": self.worker_id})
                continue
            for run_id in lost - self._settling:
                if self._runner.is_active(run_id):
                    logger.warning(
                        "Lost run %s (cancelled, deleted, or taken over); stopping it",
                        run_id, extra={"run_id": run_id},
                    )
                    self._spawn(self._runner.cancel(run_id, wait=0))

    def _on_status(self, run_id: str, status: str) -> None:
        if status == "queued":
            self.wake()
        elif (
            status in TERMINAL_STATUSES
            and self._runner.is_active(run_id)
            and run_id not in self._settling
        ):
            # Settled elsewhere (a cancel through the API): stop working on it.
            self._spawn(self._runner.cancel(run_id, wait=0))

    def _spawn(self, coroutine: Coroutine[Any, Any, Any]) -> None:
        task = asyncio.create_task(coroutine)
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    # -- jobs -----------------------------------------------------------

    async def _execute(self, run_id: str) -> None:
        claim = self._claims[run_id]
        try:
            with log_context(run_id=run_id, worker_id=self.worker_id):
                status, error = await self._jobs.execute(claim)
                self._settling.add(run_id)
                await self._jobs.finish(run_id, status, error=error, owner=self.worker_id)
            await self._cleanup(run_id)
        finally:
            self._claims.pop(run_id, None)
            self._settling.discard(run_id)
            self.wake()  # a slot is free

    async def _interrupted(self, run_id: str, reason: str) -> None:
        """A run's task was cancelled: hand it back on shutdown, else tidy up."""
        if reason == SHUTDOWN:
            if await self._queue.release(self.worker_id, run_id):
                logger.info("Released run %s for another worker", run_id, extra={"run_id": run_id})
            return
        await self._cleanup(run_id)

    async def _cleanup(self, run_id: str) -> None:
        """Drop a run's checkpoints once nothing can resume it (completed, cancelled, deleted)."""
        async with self._session_factory() as session:
            run = await session.get(ResearchRun, uuid.UUID(run_id))
            status = run.status if run is not None else None
        if status is None or status in {"completed", "cancelled"}:
            await self._jobs.drop_checkpoints(run_id)


# ---------------------------------------------------------------------------
# Standalone process
# ---------------------------------------------------------------------------


async def run_worker(settings: Settings) -> None:
    """Run one worker process until SIGTERM/SIGINT."""
    from app.container import build_worker_services
    from app.db.session import get_engine, get_session_factory

    services = build_worker_services(settings, get_session_factory())
    worker = services.worker
    bind_log_context(worker_id=worker.worker_id)

    stop = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signum in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(signum, stop.set)
        except (NotImplementedError, RuntimeError):  # Windows
            signal.signal(signum, lambda *_: loop.call_soon_threadsafe(stop.set))

    await services.startup()
    try:
        await stop.wait()
        logger.info("Worker %s stopping", worker.worker_id)
    finally:
        await services.shutdown()
        await get_engine().dispose()


def main() -> None:
    from app.core.config import get_settings
    from app.core.logging import setup_logging
    from app.core.tracing import configure_tracing

    settings = get_settings()
    setup_logging(level=settings.log_level, fmt=settings.log_format)
    configure_tracing(settings)
    if not settings.database_url:
        raise SystemExit("DATABASE_URL is not set: the worker takes its runs from the database")
    asyncio.run(run_worker(settings))


if __name__ == "__main__":
    main()
