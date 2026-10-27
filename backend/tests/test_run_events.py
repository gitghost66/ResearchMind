"""
M4 tests — run events and status: sequence numbers, atomic transitions,
terminal finality, the notifier, and the stream built on them.
"""

from __future__ import annotations

import asyncio
import uuid
from collections.abc import AsyncIterator
from typing import Any

import pytest
import pytest_asyncio
from sqlalchemy import event as sa_event
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.models.base import Base
from app.models.research import ResearchRun, RunEvent
from app.services.event_stream import is_terminal_event, stream_run_events
from app.services.research_service import build_steps
from app.services.run_events import (
    RUN_STATUS_EVENT,
    RUNNER_AGENT,
    EventNotifier,
    RunEventRecorder,
)


def _wal(dbapi_connection, connection_record):  # type: ignore[no-untyped-def]
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.close()


@pytest_asyncio.fixture
async def factory(tmp_path: Any) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    # A file, so sessions really are separate connections (as on Postgres).
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'events.db'}")
    sa_event.listen(engine.sync_engine, "connect", _wal)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield async_sessionmaker(bind=engine, expire_on_commit=False)
    await engine.dispose()


@pytest.fixture
def recorder(factory: async_sessionmaker[AsyncSession]) -> RunEventRecorder:
    return RunEventRecorder(factory)


async def events_of(factory: async_sessionmaker[AsyncSession], run_id: str) -> list[RunEvent]:
    async with factory() as session:
        rows = await session.execute(
            select(RunEvent).where(RunEvent.run_id == uuid.UUID(run_id)).order_by(RunEvent.seq)
        )
        return list(rows.scalars())


async def run_row(factory: async_sessionmaker[AsyncSession], run_id: str) -> ResearchRun:
    async with factory() as session:
        run = await session.get(ResearchRun, uuid.UUID(run_id))
        assert run is not None
        return run


# ===========================================================================
# Recorder
# ===========================================================================


class TestRecorder:
    async def test_create_run_is_queued_with_its_first_event(self, recorder, factory) -> None:
        run_id = await recorder.create_run(
            question="q", mode="quick", user_context={"a": 1}, options={"max_sources": 3}
        )

        run = await run_row(factory, run_id)
        assert (run.status, run.user_context, run.options) == ("queued", {"a": 1}, {"max_sources": 3})
        events = await events_of(factory, run_id)
        assert [(e.seq, e.agent, e.type, e.payload) for e in events] == [
            (1, RUNNER_AGENT, RUN_STATUS_EVENT, {"status": "queued", "error": None})
        ]

    async def test_seq_is_gapless_under_concurrent_writers(self, recorder, factory) -> None:
        run_id = await recorder.create_run(question="q", mode="quick")

        seqs = await asyncio.gather(
            *(
                recorder.record(run_id=run_id, agent="web_researcher", event_type=f"e{i}")
                for i in range(25)
            )
        )

        assert sorted(seqs) == list(range(2, 27))
        assert [e.seq for e in await events_of(factory, run_id)] == list(range(1, 27))

    async def test_two_recorders_on_one_run_still_get_unique_seqs(self, factory) -> None:
        first, second = RunEventRecorder(factory), RunEventRecorder(factory)
        run_id = await first.create_run(question="q", mode="quick")

        # Each recorder serialises its own writes, so a write can lose to at
        # most the other recorder's three: within the retry budget.
        await asyncio.gather(
            *(
                (first if i % 2 else second).record(run_id=run_id, agent="a", event_type="t")
                for i in range(6)
            )
        )

        seqs = [e.seq for e in await events_of(factory, run_id)]
        assert seqs == list(range(1, 8))

    async def test_runs_are_numbered_independently(self, recorder, factory) -> None:
        a = await recorder.create_run(question="a", mode="quick")
        b = await recorder.create_run(question="b", mode="quick")
        assert await recorder.record(run_id=a, agent="x", event_type="t") == 2
        assert await recorder.record(run_id=b, agent="x", event_type="t") == 2

    async def test_transition_moves_and_records(self, recorder, factory) -> None:
        run_id = await recorder.create_run(question="q", mode="quick")

        assert await recorder.transition(run_id, "running", from_statuses={"queued"})
        run = await run_row(factory, run_id)
        assert run.status == "running" and run.started_at and not run.finished_at

        assert await recorder.transition(run_id, "failed", error="boom")
        run = await run_row(factory, run_id)
        assert (run.status, run.error) == ("failed", "boom")
        assert run.finished_at

        statuses = [e.payload for e in await events_of(factory, run_id)]
        assert statuses == [
            {"status": "queued", "error": None},
            {"status": "running", "error": None},
            {"status": "failed", "error": "boom"},
        ]

    async def test_a_terminal_status_is_final(self, recorder, factory) -> None:
        run_id = await recorder.create_run(question="q", mode="quick")
        await recorder.transition(run_id, "completed")

        assert not await recorder.transition(run_id, "cancelled")
        assert not await recorder.transition(run_id, "failed", error="late")

        run = await run_row(factory, run_id)
        assert (run.status, run.error) == ("completed", None)
        assert len(await events_of(factory, run_id)) == 2

    async def test_transition_respects_from_statuses(self, recorder, factory) -> None:
        run_id = await recorder.create_run(question="q", mode="quick")
        await recorder.transition(run_id, "cancelled")

        # A job that waited in the queue must not start a cancelled run.
        assert not await recorder.transition(run_id, "running", from_statuses={"queued"})
        assert (await run_row(factory, run_id)).status == "cancelled"

    async def test_transition_of_a_missing_run(self, recorder) -> None:
        assert not await recorder.transition(str(uuid.uuid4()), "failed")

    async def test_writes_wake_listeners(self, recorder) -> None:
        run_id = await recorder.create_run(question="q", mode="quick")
        with recorder.notifier.listen(run_id) as wakeup:
            await recorder.record(run_id=run_id, agent="a", event_type="t")
            assert wakeup.is_set()
            wakeup.clear()
            await recorder.transition(run_id, "running")
            assert wakeup.is_set()


class TestNotifier:
    def test_listeners_are_per_run_and_cleaned_up(self) -> None:
        notifier = EventNotifier()
        with notifier.listen("a") as a, notifier.listen("b") as b:
            assert notifier.listener_count("a") == 1
            notifier.notify("a")
            assert a.is_set() and not b.is_set()
        assert notifier.listener_count("a") == notifier.listener_count("b") == 0
        notifier.notify("a")  # nobody listening: a no-op


# ===========================================================================
# Stream
# ===========================================================================


async def collect(agen: AsyncIterator[dict[str, Any]], timeout: float = 5.0) -> list[dict[str, Any]]:
    async def drain() -> list[dict[str, Any]]:
        return [item async for item in agen]

    return await asyncio.wait_for(drain(), timeout)


class TestStream:
    async def test_replay_from_a_cursor_then_end_on_terminal(self, recorder, factory) -> None:
        run_id = await recorder.create_run(question="q", mode="quick")
        await recorder.transition(run_id, "running")
        await recorder.record(run_id=run_id, agent="planner", event_type="agent_started")
        await recorder.transition(run_id, "completed")

        everything = await collect(
            stream_run_events(run_id, session_factory=factory, notifier=recorder.notifier)
        )
        tail = await collect(
            stream_run_events(
                run_id, session_factory=factory, notifier=recorder.notifier, after_seq=2
            )
        )

        assert [e["seq"] for e in everything] == [1, 2, 3, 4]
        assert is_terminal_event(everything[-1])
        assert [e["seq"] for e in tail] == [3, 4]

    async def test_live_events_wake_the_stream(self, recorder, factory) -> None:
        run_id = await recorder.create_run(question="q", mode="quick")
        received: list[int] = []

        async def read() -> None:
            async for item in stream_run_events(
                run_id,
                session_factory=factory,
                notifier=recorder.notifier,
                poll_interval=30,  # only a wake-up can deliver in time
            ):
                received.append(item["seq"])

        reader = asyncio.create_task(read())
        await asyncio.sleep(0.1)
        assert received == [1]

        await recorder.transition(run_id, "running")
        await recorder.record(run_id=run_id, agent="planner", event_type="agent_started")
        await recorder.transition(run_id, "completed")
        await asyncio.wait_for(reader, 5)

        assert received == [1, 2, 3, 4]

    async def test_events_from_another_process_arrive_by_polling(self, recorder, factory) -> None:
        run_id = await recorder.create_run(question="q", mode="quick")
        elsewhere = RunEventRecorder(factory)  # its own notifier: no wake-ups here

        async def write_later() -> None:
            await asyncio.sleep(0.1)
            await elsewhere.record(run_id=run_id, agent="planner", event_type="agent_started")
            await elsewhere.transition(run_id, "failed", error="x")

        writer = asyncio.create_task(write_later())
        events = await collect(
            stream_run_events(
                run_id, session_factory=factory, notifier=recorder.notifier, poll_interval=0.05
            )
        )
        await writer

        assert [e["type"] for e in events] == [RUN_STATUS_EVENT, "agent_started", RUN_STATUS_EVENT]

    async def test_a_deleted_run_ends_the_stream(self, recorder, factory) -> None:
        run_id = await recorder.create_run(question="q", mode="quick")

        async def delete_later() -> None:
            await asyncio.sleep(0.1)
            async with factory() as session:
                run = await session.get(ResearchRun, uuid.UUID(run_id))
                await session.delete(run)
                await session.commit()
            recorder.notifier.notify(run_id)

        deleter = asyncio.create_task(delete_later())
        events = await collect(
            stream_run_events(run_id, session_factory=factory, notifier=recorder.notifier)
        )
        await deleter

        assert [e["seq"] for e in events] == [1]

    async def test_a_finished_run_without_a_final_event_still_ends(
        self, recorder, factory
    ) -> None:
        """Runs from before M4 have no run_status events: the row decides."""
        run_id = await recorder.create_run(question="q", mode="quick")
        async with factory() as session:
            await session.execute(
                update(ResearchRun)
                .where(ResearchRun.id == uuid.UUID(run_id))
                .values(status="failed", error="old failure")
            )
            await session.commit()

        events = await collect(
            stream_run_events(
                run_id, session_factory=factory, notifier=recorder.notifier, after_seq=1
            )
        )

        assert len(events) == 1
        assert events[0]["seq"] == 1
        assert events[0]["payload"] == {"status": "failed", "error": "old failure"}

    async def test_large_backlogs_are_read_in_batches(self, recorder, factory) -> None:
        run_id = await recorder.create_run(question="q", mode="quick")
        for i in range(12):
            await recorder.record(run_id=run_id, agent="a", event_type=f"e{i}")
        await recorder.transition(run_id, "completed")

        events = await collect(
            stream_run_events(
                run_id, session_factory=factory, notifier=recorder.notifier, batch_size=5
            )
        )

        assert [e["seq"] for e in events] == list(range(1, 15))


# ===========================================================================
# Progress steps
# ===========================================================================


def counts(**agents: tuple[int, int]) -> dict[str, dict[str, int]]:
    return {
        agent: {"agent_started": started, "agent_finished": finished}
        for agent, (started, finished) in agents.items()
    }


def statuses(**kwargs: Any) -> dict[str, str]:
    return {s.agent: s.status for s in build_steps("quick", first_seq={}, **kwargs)}


class TestSteps:
    def test_nothing_started(self) -> None:
        assert statuses(counts={}, active=True) == dict.fromkeys(
            ("planner", "web_researcher", "synthesizer"), "pending"
        )

    def test_fan_out_is_running_until_the_next_step_starts(self) -> None:
        # One branch finished, the next not started yet (concurrency cap 1).
        running = statuses(counts=counts(planner=(1, 1), web_researcher=(1, 1)), active=True)
        assert running == {
            "planner": "completed",
            "web_researcher": "running",
            "synthesizer": "pending",
        }
        done = statuses(
            counts=counts(planner=(1, 1), web_researcher=(2, 2), synthesizer=(1, 0)), active=True
        )
        assert done["web_researcher"] == "completed"
        assert done["synthesizer"] == "running"

    def test_finished_runs(self) -> None:
        completed = statuses(
            counts=counts(planner=(1, 1), web_researcher=(2, 2), synthesizer=(1, 1)), active=False
        )
        assert set(completed.values()) == {"completed"}

        cancelled = statuses(counts=counts(planner=(1, 1), web_researcher=(2, 1)), active=False)
        assert cancelled == {
            "planner": "completed",
            "web_researcher": "incomplete",
            "synthesizer": "skipped",
        }

    def test_errors_are_counted_and_unknown_agents_listed_after_the_pipeline(self) -> None:
        steps = build_steps(
            "quick",
            {"planner": {"agent_started": 1, "agent_finished": 1, "error": 2}, "extra": {"error": 1}},
            {"planner": 2, "extra": 5},
            active=True,
        )
        assert [s.agent for s in steps] == ["planner", "web_researcher", "synthesizer", "extra"]
        assert steps[0].errors == 2
        assert steps[-1].errors == 1
