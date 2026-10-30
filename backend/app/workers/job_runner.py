"""
A process's run tasks: one asyncio task per run.

Research runs execute off the request path (PRD s5). Since M8 they are queued
in the database and claimed by :class:`~app.workers.research_worker.RunWorker`,
which hands each claimed run to a :class:`JobRunner`. The runner keeps the
registry of those tasks and enforces two caps:

* ``max_concurrent`` runs execute at once; the rest wait their turn (queued);
* ``max_pending`` runs may be queued or running in total; beyond that,
  :meth:`JobRunner.submit` raises :class:`QueueFullError`.

A job is interrupted for one of two reasons, which its ``on_interrupt``
callback receives so it can record the right outcome:

* :data:`CANCEL` — a client asked (``POST /cancel``, ``DELETE``);
* :data:`SHUTDOWN` — the server is stopping.

The registry is per process; across processes, runs are coordinated through
the database (leases, status notifications), not through this class.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Awaitable, Callable

logger = logging.getLogger(__name__)

__all__ = [
    "CANCEL",
    "SHUTDOWN",
    "JobAlreadyActiveError",
    "JobFn",
    "JobRunner",
    "InterruptFn",
    "QueueFullError",
]

CANCEL = "cancel"
SHUTDOWN = "shutdown"

JobFn = Callable[[str], Awaitable[None]]
InterruptFn = Callable[[str, str], Awaitable[None]]


class QueueFullError(RuntimeError):
    """No room for another run (or the runner is shutting down)."""


class JobAlreadyActiveError(RuntimeError):
    """The run already has a task."""


class JobRunner:
    """Registry of background run tasks with concurrency and queue caps."""

    def __init__(
        self,
        job: JobFn,
        *,
        on_interrupt: InterruptFn,
        max_concurrent: int,
        max_pending: int,
    ) -> None:
        self._job = job
        self._on_interrupt = on_interrupt
        self._max_pending = max(1, max_pending)
        self._slots = asyncio.Semaphore(max(1, max_concurrent))
        self._tasks: dict[str, asyncio.Task[None]] = {}
        # Tasks whose coroutine has begun. A task cancelled before its first
        # step never runs its body (so never its except/finally), so those
        # are flagged instead and stop themselves when they start.
        self._started: set[str] = set()
        self._running: set[str] = set()
        # Why a task was cancelled; set once, so a run is interrupted once.
        self._interrupts: dict[str, str] = {}
        self._closed = False

    # -- state ----------------------------------------------------------

    @property
    def pending_count(self) -> int:
        """Runs queued or running."""
        return len(self._tasks)

    @property
    def running_count(self) -> int:
        return len(self._running)

    @property
    def accepting(self) -> bool:
        return not self._closed and len(self._tasks) < self._max_pending

    def is_active(self, run_id: str) -> bool:
        return run_id in self._tasks

    def is_running(self, run_id: str) -> bool:
        return run_id in self._running

    # -- control --------------------------------------------------------

    def submit(self, run_id: str) -> None:
        """Start *run_id*'s job as soon as a slot is free."""
        if self._closed:
            raise QueueFullError("the server is shutting down")
        if run_id in self._tasks:
            raise JobAlreadyActiveError(run_id)
        if len(self._tasks) >= self._max_pending:
            raise QueueFullError(f"{self._max_pending} runs are already queued or running")
        self._tasks[run_id] = asyncio.create_task(self._run(run_id), name=f"run:{run_id}")

    async def cancel(self, run_id: str, *, wait: float) -> bool:
        """
        Interrupt *run_id* and wait up to *wait* seconds for it to stop.

        Returns ``False`` if it has no task. A run already being interrupted
        is not interrupted again; the call just waits.
        """
        return await self._interrupt(run_id, CANCEL, wait=wait)

    async def shutdown(self, *, grace: float) -> None:
        """Stop accepting runs, interrupt every task, and wait up to *grace* seconds."""
        self._closed = True
        tasks = list(self._tasks.values())
        for run_id in list(self._tasks):
            self._request_interrupt(run_id, SHUTDOWN)
        if tasks:
            _, still_running = await asyncio.wait(tasks, timeout=grace)
            if still_running:
                logger.warning(
                    "%d run(s) did not stop within %.0fs of shutdown", len(still_running), grace
                )

    async def wait_idle(self) -> None:
        """Wait until every task has finished (tests and orderly shutdown)."""
        while self._tasks:
            await asyncio.wait(list(self._tasks.values()))

    # -- internals ------------------------------------------------------

    async def _interrupt(self, run_id: str, reason: str, *, wait: float) -> bool:
        task = self._tasks.get(run_id)
        if task is None:
            return False
        self._request_interrupt(run_id, reason)
        await asyncio.wait({task}, timeout=max(0.0, wait))
        return True

    def _request_interrupt(self, run_id: str, reason: str) -> None:
        task = self._tasks.get(run_id)
        if task is None or run_id in self._interrupts:
            return
        self._interrupts[run_id] = reason
        if run_id in self._started:
            task.cancel()

    async def _run(self, run_id: str) -> None:
        self._started.add(run_id)
        try:
            if run_id in self._interrupts:
                raise asyncio.CancelledError  # interrupted before it started
            async with self._slots:
                self._running.add(run_id)
                try:
                    await self._job(run_id)
                finally:
                    self._running.discard(run_id)
        except asyncio.CancelledError:
            reason = self._interrupts.get(run_id, CANCEL)
            try:
                await self._on_interrupt(run_id, reason)
            except Exception:  # noqa: BLE001 - nothing else can record it
                logger.exception("Failed to record interrupted run %s", run_id)
            raise
        except Exception:  # noqa: BLE001 - a job should record its own failure
            logger.exception("Job for run %s raised", run_id, extra={"run_id": run_id})
        finally:
            self._tasks.pop(run_id, None)
            self._interrupts.pop(run_id, None)
            self._started.discard(run_id)
