"""
M8: the run queue in research_runs — claims, leases, hand-backs, resumes.

On a SQLite file (several connections), since races between workers are
what the conditional updates exist for.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from sqlalchemy import event, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.models.base import Base
from app.models.research import ResearchRun, RunEvent
from app.services.run_events import RunEventRecorder
from app.services.run_queue import RunQueue, gave_up_error


@pytest.fixture
async def factory(tmp_path: Path) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'queue.db'}")

    def wal(dbapi_connection, _record):  # type: ignore[no-untyped-def]
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.close()

    event.listen(engine.sync_engine, "connect", wal)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    await engine.dispose()


@pytest.fixture
def recorder(factory: async_sessionmaker[AsyncSession]) -> RunEventRecorder:
    return RunEventRecorder(factory)


@pytest.fixture
def queue(factory: async_sessionmaker[AsyncSession], recorder: RunEventRecorder) -> RunQueue:
    return RunQueue(factory, recorder, lease_seconds=30, max_attempts=3)


async def new_run(recorder: RunEventRecorder, question: str = "Q?") -> str:
    return await recorder.create_run(question=question, mode="quick")


async def row(factory: async_sessionmaker[AsyncSession], run_id: str) -> ResearchRun:
    async with factory() as session:
        run = await session.get(ResearchRun, uuid.UUID(run_id))
        assert run is not None
        return run


async def lapse(factory: async_sessionmaker[AsyncSession], run_id: str) -> None:
    async with factory() as session:
        await session.execute(
            update(ResearchRun)
            .where(ResearchRun.id == uuid.UUID(run_id))
            .values(lease_expires_at=datetime.now(UTC) - timedelta(seconds=1))
        )
        await session.commit()


async def statuses(factory: async_sessionmaker[AsyncSession], run_id: str) -> list[dict]:
    async with factory() as session:
        rows = await session.execute(
            select(RunEvent.payload)
            .where(RunEvent.run_id == uuid.UUID(run_id), RunEvent.type == "run_status")
            .order_by(RunEvent.seq)
        )
        return [dict(payload) for payload in rows.scalars()]


class TestClaims:
    async def test_the_oldest_queued_run_is_claimed_with_a_lease(
        self, queue: RunQueue, recorder: RunEventRecorder, factory: async_sessionmaker[AsyncSession]
    ) -> None:
        first = await new_run(recorder, "first")
        await new_run(recorder, "second")

        claim = await queue.claim_next("worker-a")

        assert claim is not None
        assert (claim.run_id, claim.attempt, claim.resumed) == (first, 1, False)
        run = await row(factory, first)
        assert run.status == "running" and run.worker_id == "worker-a" and run.attempts == 1
        assert run.started_at is not None
        lease = run.lease_expires_at
        assert lease is not None
        assert lease.replace(tzinfo=lease.tzinfo or UTC) > datetime.now(UTC) + timedelta(seconds=20)
        assert await queue.counts() == {"queued": 1, "running": 1}

    async def test_racing_workers_never_share_a_run(
        self, queue: RunQueue, recorder: RunEventRecorder
    ) -> None:
        runs = {await new_run(recorder, f"q{i}") for i in range(3)}

        claims = await asyncio.gather(*(queue.claim_next(f"worker-{i}") for i in range(6)))

        won = [c.run_id for c in claims if c is not None]
        assert sorted(won) == sorted(runs)  # each run exactly once

    async def test_nothing_to_claim(self, queue: RunQueue) -> None:
        assert await queue.claim_next("worker-a") is None

    async def test_a_cancelled_run_is_not_claimed(
        self, queue: RunQueue, recorder: RunEventRecorder
    ) -> None:
        run_id = await new_run(recorder)
        await recorder.transition(run_id, "cancelled")
        assert await queue.claim_next("worker-a") is None


class TestLeases:
    async def test_renewal_reports_runs_the_worker_lost(
        self, queue: RunQueue, recorder: RunEventRecorder, factory: async_sessionmaker[AsyncSession]
    ) -> None:
        kept = await new_run(recorder, "kept")
        cancelled = await new_run(recorder, "cancelled")
        stolen = await new_run(recorder, "stolen")
        for _ in range(3):
            await queue.claim_next("worker-a")
        await recorder.transition(cancelled, "cancelled")
        async with factory() as session:
            await session.execute(
                update(ResearchRun)
                .where(ResearchRun.id == uuid.UUID(stolen))
                .values(worker_id="worker-b")
            )
            await session.commit()
        deleted = str(uuid.uuid4())

        lost = await queue.renew("worker-a", [kept, cancelled, stolen, deleted])

        assert lost == {cancelled, stolen, deleted}

    async def test_a_lapsed_lease_is_resumed_by_another_worker(
        self, queue: RunQueue, recorder: RunEventRecorder, factory: async_sessionmaker[AsyncSession]
    ) -> None:
        run_id = await new_run(recorder)
        await queue.claim_next("worker-a")
        assert await queue.claim_next("worker-b") is None  # still leased

        await lapse(factory, run_id)
        claim = await queue.claim_next("worker-b")

        assert claim is not None and claim.run_id == run_id
        assert (claim.attempt, claim.resumed) == (2, True)
        run = await row(factory, run_id)
        assert run.worker_id == "worker-b" and run.attempts == 2
        assert (await statuses(factory, run_id))[-1] == {
            "status": "running", "error": None, "resumed": True, "attempt": 2,
        }
        # The first worker's late renewal finds it gone.
        assert await queue.renew("worker-a", [run_id]) == {run_id}

    async def test_a_run_interrupted_too_often_is_failed(
        self, queue: RunQueue, recorder: RunEventRecorder, factory: async_sessionmaker[AsyncSession]
    ) -> None:
        run_id = await new_run(recorder)
        for worker in ("a", "b", "c"):
            assert await queue.claim_next(worker) is not None
            await lapse(factory, run_id)

        assert await queue.claim_next("d") is None

        run = await row(factory, run_id)
        assert run.status == "failed" and run.error == gave_up_error(3)
        assert run.lease_expires_at is None

    async def test_a_run_left_running_before_leases_is_failed(
        self, queue: RunQueue, recorder: RunEventRecorder, factory: async_sessionmaker[AsyncSession]
    ) -> None:
        run_id = await new_run(recorder)
        async with factory() as session:  # how a pre-M8 process left it: running, no lease
            await session.execute(
                update(ResearchRun).where(ResearchRun.id == uuid.UUID(run_id)).values(status="running")
            )
            await session.commit()

        assert await queue.claim_next("worker-a") is None

        run = await row(factory, run_id)
        assert run.status == "failed" and run.error is not None
        assert await queue.requeue_failed(run_id)  # and it can be resumed

    async def test_release_hands_the_run_back_without_counting_an_attempt(
        self, queue: RunQueue, recorder: RunEventRecorder, factory: async_sessionmaker[AsyncSession]
    ) -> None:
        run_id = await new_run(recorder)
        await queue.claim_next("worker-a")
        woken: list[tuple[str, str]] = []
        recorder.notifier.subscribe_status(lambda r, s: woken.append((r, s)))

        assert await queue.release("worker-a", run_id)

        run = await row(factory, run_id)
        assert run.status == "running" and run.worker_id is None and run.attempts == 0
        assert woken == [(run_id, "queued")]
        claim = await queue.claim_next("worker-b")
        assert claim is not None and claim.attempt == 1 and claim.resumed
        # Only the owner can release.
        assert not await queue.release("worker-a", run_id)

    async def test_keep_lease_renews_in_the_background(
        self, factory: async_sessionmaker[AsyncSession], recorder: RunEventRecorder
    ) -> None:
        quick = RunQueue(factory, recorder, lease_seconds=1.0, max_attempts=3)
        run_id = await new_run(recorder)
        assert await quick.claim_run(run_id, "script") is not None
        before = (await row(factory, run_id)).lease_expires_at

        async with quick.keep_lease(run_id, "script"):
            await asyncio.sleep(0.5)

        assert (await row(factory, run_id)).lease_expires_at > before  # type: ignore[operator]


class TestResume:
    async def test_a_failed_run_is_queued_again_and_claimed_as_a_resume(
        self, queue: RunQueue, recorder: RunEventRecorder, factory: async_sessionmaker[AsyncSession]
    ) -> None:
        run_id = await new_run(recorder)
        await queue.claim_next("worker-a")
        await recorder.transition(run_id, "failed", error="quota", owner="worker-a")

        assert await queue.requeue_failed(run_id)

        run = await row(factory, run_id)
        assert run.status == "queued"
        assert (run.error, run.finished_at, run.worker_id, run.attempts) == (None, None, None, 0)
        claim = await queue.claim_next("worker-b")
        assert claim is not None and (claim.attempt, claim.resumed) == (1, True)
        assert not await queue.requeue_failed(run_id)  # only failed runs

    async def test_the_owner_check_stops_a_stale_worker_settling_a_run(
        self, queue: RunQueue, recorder: RunEventRecorder, factory: async_sessionmaker[AsyncSession]
    ) -> None:
        run_id = await new_run(recorder)
        await queue.claim_next("worker-a")
        await lapse(factory, run_id)
        await queue.claim_next("worker-b")

        assert not await recorder.transition(run_id, "completed", owner="worker-a")
        assert (await row(factory, run_id)).status == "running"
        assert await recorder.transition(run_id, "completed", owner="worker-b")
