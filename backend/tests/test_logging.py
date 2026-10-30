"""M8: structured logs carry run_id and worker_id from the context they run in."""

from __future__ import annotations

import asyncio
import json
import logging

from app.core.logging import JSONFormatter, bind_log_context, log_context


def render(message: str, **extra: object) -> dict[str, object]:
    record = logging.LogRecord("test", logging.INFO, __file__, 1, message, None, None)
    for key, value in extra.items():
        setattr(record, key, value)
    return json.loads(JSONFormatter().format(record))


class TestLogContext:
    def test_fields_bound_in_a_block_are_on_every_line(self) -> None:
        with log_context(run_id="run-1", worker_id="w-1"):
            line = render("inside")
        assert (line["run_id"], line["worker_id"]) == ("run-1", "w-1")
        assert "run_id" not in render("outside")

    def test_an_explicit_extra_wins(self) -> None:
        with log_context(run_id="run-1"):
            assert render("x", run_id="run-2")["run_id"] == "run-2"

    def test_blocks_nest_and_restore(self) -> None:
        with log_context(run_id="outer"):
            with log_context(run_id="inner"):
                assert render("x")["run_id"] == "inner"
            assert render("x")["run_id"] == "outer"

    async def test_tasks_started_inside_a_run_inherit_its_id(self) -> None:
        async def branch() -> dict[str, object]:
            await asyncio.sleep(0)
            return render("from a branch")

        with log_context(run_id="run-9"):
            task = asyncio.create_task(branch())
        assert (await task)["run_id"] == "run-9"

    async def test_a_bound_worker_id_lasts_for_the_context(self) -> None:
        async def worker() -> dict[str, object]:
            bind_log_context(worker_id="host:1:abc")
            return render("worker line")

        assert (await asyncio.create_task(worker()))["worker_id"] == "host:1:abc"
        assert "worker_id" not in render("main line")  # the task's context, not ours
