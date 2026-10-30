"""
M8: the SQL checkpoint saver.

Checked two ways: the saver contract on its own (what LangGraph reads back is
what it wrote), and a small fan-out graph killed mid-step and resumed through
a fresh saver, which is what a worker restart does.
"""

from __future__ import annotations

import asyncio
import contextlib
import operator
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Annotated, Any, TypedDict

import pytest
from langgraph.checkpoint.base import empty_checkpoint
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.models.base import Base
from app.models.checkpoints import GraphCheckpointBlob
from app.services.checkpointer import SqlCheckpointSaver


@pytest.fixture
async def factory(tmp_path: Path) -> AsyncIterator[async_sessionmaker[AsyncSession]]:
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / 'checkpoints.db'}")
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    yield async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    await engine.dispose()


def config(thread: str = "run-1", checkpoint_id: str | None = None) -> Any:
    configurable: dict[str, Any] = {"thread_id": thread, "checkpoint_ns": ""}
    if checkpoint_id:
        configurable["checkpoint_id"] = checkpoint_id
    return {"configurable": configurable}


def checkpoint(checkpoint_id: str, values: dict[str, Any], versions: dict[str, str]) -> Any:
    item = empty_checkpoint()
    item["id"] = checkpoint_id
    item["channel_values"] = values
    item["channel_versions"] = versions
    return item


class TestSaver:
    async def test_put_then_get_latest_and_by_id(self, factory: async_sessionmaker[AsyncSession]) -> None:
        saver = SqlCheckpointSaver(factory)
        first = await saver.aput(
            config(), checkpoint("1", {"a": [1]}, {"a": "v1"}), {"step": 0}, {"a": "v1"}
        )
        await saver.aput(
            first, checkpoint("2", {"a": [1, 2], "b": "x"}, {"a": "v2", "b": "v1"}),
            {"step": 1}, {"a": "v2", "b": "v1"},
        )

        latest = await saver.aget_tuple(config())
        assert latest is not None
        assert latest.checkpoint["id"] == "2"
        assert latest.checkpoint["channel_values"] == {"a": [1, 2], "b": "x"}
        assert latest.metadata["step"] == 1
        assert latest.parent_config == config(checkpoint_id="1")

        older = await saver.aget_tuple(config(checkpoint_id="1"))
        assert older is not None and older.checkpoint["channel_values"] == {"a": [1]}
        assert older.parent_config is None
        assert await saver.aget_tuple(config("other-run")) is None

    async def test_an_unchanged_channel_is_not_stored_again(
        self, factory: async_sessionmaker[AsyncSession]
    ) -> None:
        saver = SqlCheckpointSaver(factory)
        first = await saver.aput(
            config(), checkpoint("1", {"big": "text"}, {"big": "v1"}), {}, {"big": "v1"}
        )
        # Step 2 changes nothing: no new version, so no new blob.
        await saver.aput(first, checkpoint("2", {"big": "text"}, {"big": "v1"}), {}, {})

        async with factory() as session:
            blobs = (await session.execute(select(func.count()).select_from(GraphCheckpointBlob))).scalar_one()
        assert blobs == 1
        latest = await saver.aget_tuple(config())
        assert latest is not None and latest.checkpoint["channel_values"] == {"big": "text"}

    async def test_writes_are_kept_once_per_task_and_returned_in_order(
        self, factory: async_sessionmaker[AsyncSession]
    ) -> None:
        saver = SqlCheckpointSaver(factory)
        saved = await saver.aput(config(), checkpoint("1", {}, {}), {}, {})
        await saver.aput_writes(saved, [("done", "b")], task_id="task-b", task_path="~b")
        await saver.aput_writes(saved, [("done", "a"), ("count", 1)], task_id="task-a", task_path="~a")
        # A task saving the same writes again (a retry) does not duplicate them.
        await saver.aput_writes(saved, [("done", "a-again")], task_id="task-a", task_path="~a")

        found = await saver.aget_tuple(config())
        assert found is not None
        assert found.pending_writes == [
            ("task-a", "done", "a"),
            ("task-a", "count", 1),
            ("task-b", "done", "b"),
        ]

    async def test_list_filters_and_limits(self, factory: async_sessionmaker[AsyncSession]) -> None:
        saver = SqlCheckpointSaver(factory)
        parent = config()
        for index in range(1, 4):
            parent = await saver.aput(
                parent, checkpoint(str(index), {}, {}), {"source": "loop", "step": index}, {}
            )

        ids = [t.checkpoint["id"] async for t in saver.alist(config())]
        assert ids == ["3", "2", "1"]
        assert [t.checkpoint["id"] async for t in saver.alist(config(), limit=1)] == ["3"]
        assert [
            t.checkpoint["id"] async for t in saver.alist(config(), before=config(checkpoint_id="3"))
        ] == ["2", "1"]
        assert [t.checkpoint["id"] async for t in saver.alist(config(), filter={"step": 2})] == ["2"]

    async def test_delete_thread_removes_only_that_run(
        self, factory: async_sessionmaker[AsyncSession]
    ) -> None:
        saver = SqlCheckpointSaver(factory)
        for thread in ("run-1", "run-2"):
            saved = await saver.aput(config(thread), checkpoint("1", {"a": 1}, {"a": "v1"}), {}, {"a": "v1"})
            await saver.aput_writes(saved, [("a", 2)], task_id="t")

        await saver.adelete_thread("run-1")

        assert await saver.aget_tuple(config("run-1")) is None
        kept = await saver.aget_tuple(config("run-2"))
        assert kept is not None and kept.pending_writes == [("t", "a", 2)]

    def test_versions_increase(self, factory: async_sessionmaker[AsyncSession]) -> None:
        saver = SqlCheckpointSaver(factory)
        first = saver.get_next_version(None, None)
        second = saver.get_next_version(first, None)
        assert first < second and second.startswith("0" * 31 + "2.")


# ---------------------------------------------------------------------------
# A graph killed mid-step and resumed
# ---------------------------------------------------------------------------


class _State(TypedDict, total=False):
    items: list[str]
    done: Annotated[list[str], operator.add]
    report: str


def build_graph(saver: SqlCheckpointSaver, calls: dict[str, int], hold: asyncio.Event) -> Any:
    graph = StateGraph(_State)

    async def plan(state: _State) -> dict[str, Any]:
        calls["plan"] = calls.get("plan", 0) + 1
        return {"items": ["a", "b", "slow"]}

    async def work(task: dict[str, str]) -> dict[str, Any]:
        calls[task["item"]] = calls.get(task["item"], 0) + 1
        if task["item"] == "slow":
            await hold.wait()
        return {"done": [task["item"]]}

    async def report(state: _State) -> dict[str, Any]:
        return {"report": ",".join(sorted(state["done"]))}

    graph.add_node("plan", plan)
    graph.add_node("work", work)
    graph.add_node("report", report)
    graph.add_edge(START, "plan")
    graph.add_conditional_edges(
        "plan", lambda s: [Send("work", {"item": i}) for i in s["items"]], ["work"]
    )
    graph.add_edge("work", "report")
    graph.add_edge("report", END)
    return graph.compile(checkpointer=saver)


class TestResume:
    async def test_a_killed_run_resumes_with_only_unfinished_work(
        self, factory: async_sessionmaker[AsyncSession]
    ) -> None:
        calls: dict[str, int] = {}
        hold = asyncio.Event()
        run = config("run-x")
        graph = build_graph(SqlCheckpointSaver(factory), calls, hold)
        task = asyncio.create_task(graph.ainvoke({"items": []}, config=run, durability="sync"))
        async with asyncio.timeout(10):
            while not {"a", "b", "slow"} <= calls.keys():
                await asyncio.sleep(0.01)
            # The finished branches' writes are saved as they finish.
            while len((await SqlCheckpointSaver(factory).aget_tuple(run)).pending_writes) < 2:  # type: ignore[union-attr]
                await asyncio.sleep(0.01)
        task.cancel()  # the worker dies
        with contextlib.suppress(asyncio.CancelledError):
            await task

        hold.set()
        resumed = build_graph(SqlCheckpointSaver(factory), calls, hold)  # a new process
        assert (await resumed.aget_state(run)).next == ("work",)
        result = await resumed.ainvoke(None, config=run, durability="sync")

        assert result["report"] == "a,b,slow"
        assert calls == {"plan": 1, "a": 1, "b": 1, "slow": 2}
