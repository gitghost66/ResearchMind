"""
M4 tests — the in-process job runner: caps, cancellation, shutdown.

Jobs here are plain coroutines driven by events, so every state (queued,
running, interrupted) is reached deterministically.
"""

from __future__ import annotations

import asyncio

import pytest

from app.workers.job_runner import (
    CANCEL,
    SHUTDOWN,
    JobAlreadyActiveError,
    JobRunner,
    QueueFullError,
)


class Jobs:
    """Each job waits for its own release event; records starts and interrupts."""

    def __init__(self) -> None:
        self.release: dict[str, asyncio.Event] = {}
        self.started: list[str] = []
        self.finished: list[str] = []
        self.interrupted: list[tuple[str, str]] = []
        self.fail: set[str] = set()

    def event(self, run_id: str) -> asyncio.Event:
        return self.release.setdefault(run_id, asyncio.Event())

    async def job(self, run_id: str) -> None:
        self.started.append(run_id)
        await self.event(run_id).wait()
        if run_id in self.fail:
            raise RuntimeError("job failed")
        self.finished.append(run_id)

    async def on_interrupt(self, run_id: str, reason: str) -> None:
        self.interrupted.append((run_id, reason))


def runner(jobs: Jobs, *, concurrent: int = 2, pending: int = 10) -> JobRunner:
    return JobRunner(
        jobs.job, on_interrupt=jobs.on_interrupt, max_concurrent=concurrent, max_pending=pending
    )


async def settle() -> None:
    for _ in range(5):
        await asyncio.sleep(0)


class TestCaps:
    async def test_runs_beyond_the_concurrency_cap_wait(self) -> None:
        jobs = Jobs()
        r = runner(jobs, concurrent=2)
        for run_id in ("a", "b", "c"):
            r.submit(run_id)
        await settle()

        assert jobs.started == ["a", "b"]
        assert (r.running_count, r.pending_count) == (2, 3)

        jobs.event("a").set()
        await settle()
        assert jobs.started == ["a", "b", "c"]
        assert jobs.finished == ["a"]

        jobs.event("b").set()
        jobs.event("c").set()
        await r.wait_idle()
        assert r.pending_count == 0 and r.running_count == 0

    async def test_pending_cap(self) -> None:
        jobs = Jobs()
        r = runner(jobs, concurrent=1, pending=2)
        r.submit("a")
        r.submit("b")

        assert not r.accepting
        with pytest.raises(QueueFullError):
            r.submit("c")

        jobs.event("a").set()
        await settle()
        assert r.accepting
        r.submit("c")
        jobs.event("b").set()
        jobs.event("c").set()
        await r.wait_idle()

    async def test_a_run_can_only_be_active_once(self) -> None:
        jobs = Jobs()
        r = runner(jobs)
        r.submit("a")
        with pytest.raises(JobAlreadyActiveError):
            r.submit("a")
        jobs.event("a").set()
        await r.wait_idle()
        r.submit("a")  # finished, so it may run again
        await r.cancel("a", wait=1)


class TestCancel:
    async def test_cancel_a_running_job(self) -> None:
        jobs = Jobs()
        r = runner(jobs)
        r.submit("a")
        await settle()

        assert await r.cancel("a", wait=1) is True

        assert jobs.interrupted == [("a", CANCEL)]
        assert jobs.finished == []
        assert not r.is_active("a")

    async def test_cancel_a_queued_job_never_starts_it(self) -> None:
        jobs = Jobs()
        r = runner(jobs, concurrent=1)
        r.submit("a")
        r.submit("b")
        await settle()

        assert await r.cancel("b", wait=1)
        jobs.event("a").set()
        await r.wait_idle()

        assert jobs.started == ["a"]
        assert jobs.interrupted == [("b", CANCEL)]

    async def test_cancel_before_the_task_first_runs(self) -> None:
        jobs = Jobs()
        r = runner(jobs)
        r.submit("a")  # no await: its coroutine has not started

        assert await r.cancel("a", wait=1)

        assert jobs.started == []
        assert jobs.interrupted == [("a", CANCEL)]
        assert not r.is_active("a")
        assert r.pending_count == 0

    async def test_shutdown_before_tasks_first_run(self) -> None:
        jobs = Jobs()
        r = runner(jobs)
        r.submit("a")
        r.submit("b")

        await r.shutdown(grace=1)

        assert jobs.started == []
        assert sorted(jobs.interrupted) == [("a", SHUTDOWN), ("b", SHUTDOWN)]
        assert r.pending_count == 0

    async def test_cancel_unknown_run(self) -> None:
        assert await runner(Jobs()).cancel("nope", wait=0.1) is False

    async def test_a_run_is_interrupted_once(self) -> None:
        jobs = Jobs()
        slow_interrupts: list[str] = []

        async def on_interrupt(run_id: str, reason: str) -> None:
            slow_interrupts.append(reason)
            await asyncio.sleep(0.05)

        r = JobRunner(jobs.job, on_interrupt=on_interrupt, max_concurrent=1, max_pending=5)
        r.submit("a")
        await settle()

        await asyncio.gather(r.cancel("a", wait=1), r.cancel("a", wait=1), r.shutdown(grace=1))

        assert slow_interrupts == [CANCEL]

    async def test_cancel_returns_after_the_wait_even_if_the_job_lingers(self) -> None:
        stuck = asyncio.Event()

        async def stubborn(run_id: str) -> None:
            try:
                await asyncio.sleep(10)
            except asyncio.CancelledError:
                await stuck.wait()  # ignores the first cancel for a while
                raise

        recorded: list[str] = []

        async def on_interrupt(run_id: str, reason: str) -> None:
            recorded.append(reason)

        r = JobRunner(stubborn, on_interrupt=on_interrupt, max_concurrent=1, max_pending=5)
        r.submit("a")
        await settle()

        assert await r.cancel("a", wait=0.05) is True
        assert r.is_active("a")
        stuck.set()
        await r.wait_idle()
        assert recorded == [CANCEL]


class TestFailuresAndShutdown:
    async def test_a_failing_job_is_contained(self) -> None:
        jobs = Jobs()
        jobs.fail.add("a")
        r = runner(jobs)
        r.submit("a")
        r.submit("b")
        jobs.event("a").set()
        jobs.event("b").set()

        await r.wait_idle()

        assert jobs.finished == ["b"]
        assert jobs.interrupted == []

    async def test_an_interrupt_handler_error_is_contained(self) -> None:
        jobs = Jobs()

        async def broken(run_id: str, reason: str) -> None:
            raise RuntimeError("database down")

        r = JobRunner(jobs.job, on_interrupt=broken, max_concurrent=1, max_pending=5)
        r.submit("a")
        await settle()

        assert await r.cancel("a", wait=1)
        assert not r.is_active("a")

    async def test_shutdown_interrupts_everything_and_refuses_new_runs(self) -> None:
        jobs = Jobs()
        r = runner(jobs, concurrent=1)
        r.submit("running")
        r.submit("queued")
        await settle()

        await r.shutdown(grace=1)

        assert sorted(jobs.interrupted) == [("queued", SHUTDOWN), ("running", SHUTDOWN)]
        assert r.pending_count == 0
        assert not r.accepting
        with pytest.raises(QueueFullError, match="shutting down"):
            r.submit("late")

    async def test_cancel_reason_wins_over_a_later_shutdown(self) -> None:
        jobs = Jobs()
        r = runner(jobs)
        r.submit("a")
        await settle()

        await asyncio.gather(r.cancel("a", wait=1), r.shutdown(grace=1))

        assert jobs.interrupted == [("a", CANCEL)]
