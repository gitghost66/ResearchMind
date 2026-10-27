"""
M4 tests — the research API over HTTP, end to end.

The real app, services, and background job runner run against a SQLite file;
the agents' fakes replace the LLM, search, pages, and embeddings (AGENTS.md
rule 7). The ``gate`` on the harness holds a run at its first model call (the
planner's), which is how a run is caught "running" for the cancel, progress,
and live-stream tests.

M4 acceptance: POST creates a run; SSE streams events live; the report
endpoint returns the result; cancel stops a run.
"""

from __future__ import annotations

import asyncio
import json
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import Any

import pytest
from httpx import ASGITransport, AsyncClient
from sqlalchemy import func, select

from app.main import create_app
from app.models.evidence import Passage, Source
from app.models.research import LLMUsage, Report, ResearchRun, RunEvent, SubQuestion
from app.services import research_service
from app.services.run_events import INTERRUPTED_ERROR

from .conftest import ApiHarness, make_test_settings

MakeApi = Callable[..., Awaitable[ApiHarness]]
BASE = "/api/v1/research"


def parse_sse(text: str) -> list[dict[str, Any]]:
    """Parse an SSE body into ``{"event", "id", "retry", "data"}`` dicts (comments dropped)."""
    events: list[dict[str, Any]] = []
    for block in text.replace("\r\n", "\n").split("\n\n"):
        fields: dict[str, Any] = {}
        data: list[str] = []
        for line in block.split("\n"):
            if not line or line.startswith(":"):
                continue
            name, _, value = line.partition(":")
            value = value[1:] if value.startswith(" ") else value
            if name == "data":
                data.append(value)
            else:
                fields[name] = value
        if data or fields:
            fields["data"] = json.loads("\n".join(data)) if data else None
            events.append(fields)
    return events


async def stream(api: ApiHarness, run_id: str, **kwargs: Any) -> list[dict[str, Any]]:
    response = await api.client.get(f"{BASE}/{run_id}/events", **kwargs)
    assert response.status_code == 200, response.text
    assert response.headers["content-type"].startswith("text/event-stream")
    return parse_sse(response.text)


async def count_rows(api: ApiHarness, model: Any, run_id: str) -> int:
    async with api.session_factory() as session:
        column = model.run_id if hasattr(model, "run_id") else None
        if column is None:  # passages hang off sources
            stmt = (
                select(func.count())
                .select_from(Passage)
                .join(Source, Passage.source_id == Source.id)
                .where(Source.run_id == uuid.UUID(run_id))
            )
        else:
            stmt = select(func.count()).select_from(model).where(column == uuid.UUID(run_id))
        return int((await session.execute(stmt)).scalar_one())


async def hold(api: ApiHarness) -> None:
    """Make the next run stop at its first model call (the planner's)."""
    api.gate.clear()
    api.waiting.clear()


# ===========================================================================
# Create
# ===========================================================================


class TestCreate:
    async def test_post_returns_202_with_run_id_and_links(self, api: ApiHarness) -> None:
        response = await api.client.post(BASE, json={"question": "  RAG versus fine-tuning  "})

        assert response.status_code == 202
        body = response.json()
        run_id = body["run_id"]
        uuid.UUID(run_id)
        assert body["status"] == "queued"
        assert body["links"]["events"] == f"{BASE}/{run_id}/events"
        assert response.headers["location"] == f"{BASE}/{run_id}"

        detail = await api.wait_for(run_id)
        assert detail["status"] == "completed"
        assert detail["question"] == "RAG versus fine-tuning"
        assert detail["mode"] == api.settings.default_mode

    async def test_report_cites_only_stored_sources(self, api: ApiHarness) -> None:
        run_id = await api.create()
        await api.wait_for(run_id, "completed")

        report = (await api.client.get(f"{BASE}/{run_id}/report")).json()
        sources = (await api.client.get(f"{BASE}/{run_id}/sources")).json()

        stored = {item["id"] for item in sources["items"]}
        cited = {c["source_id"] for c in report["content"]["sources"]}
        assert cited and cited <= stored
        assert report["markdown"].startswith("# ")
        assert report["version"] == 1
        assert report["overall_confidence"] is None  # scored from M5

    @pytest.mark.parametrize(
        ("body", "field", "kind"),
        [
            ({"question": ""}, "question", "string_too_short"),
            ({"question": "   "}, "question", "value_error"),
            ({"question": "x" * 101}, "question", "string_too_long"),
            ({"question": "q", "mode": "deep"}, "mode", "unsupported_mode"),
            ({"question": "q", "user_context": {"note": "y" * 300}}, "user_context", "too_large"),
            ({"question": "q", "max_sources": 6}, "max_sources", "less_than_equal"),
            ({"question": "q", "max_sources": 0}, "max_sources", "greater_than_equal"),
            ({"question": "q", "surprise": 1}, "surprise", "extra_forbidden"),
        ],
    )
    async def test_invalid_bodies_are_422_with_field_details(
        self, make_api: MakeApi, body: dict[str, Any], field: str, kind: str
    ) -> None:
        api = await make_api(
            max_question_chars=100, max_user_context_bytes=200, max_sources_limit=5
        )

        response = await api.client.post(BASE, json=body)

        assert response.status_code == 422
        error = response.json()["error"]
        assert error["code"] == "validation_error"
        assert any(d["loc"][-1] == field and d["type"] == kind for d in error["details"]), error
        async with api.session_factory() as session:
            assert (await session.execute(select(func.count(ResearchRun.id)))).scalar_one() == 0

    async def test_max_sources_caps_the_run(self, api: ApiHarness) -> None:
        run_id = await api.create(max_sources=1)

        detail = await api.wait_for(run_id, "completed")

        assert detail["options"] == {"max_sources": 1}
        sources = (await api.client.get(f"{BASE}/{run_id}/sources")).json()
        assert sources["total"] == 1

    async def test_user_context_is_stored(self, api: ApiHarness) -> None:
        run_id = await api.create(user_context={"role": "student"})
        detail = await api.wait_for(run_id)
        assert detail["user_context"] == {"role": "student"}

    async def test_full_queue_is_503(self, make_api: MakeApi) -> None:
        api = await make_api(max_pending_runs=1)
        await hold(api)
        await api.create()
        await asyncio.wait_for(api.waiting.wait(), 10)

        response = await api.client.post(BASE, json={"question": "another"})

        assert response.status_code == 503
        assert response.json()["error"]["code"] == "queue_full"
        assert response.headers["retry-after"]


# ===========================================================================
# Read
# ===========================================================================


class TestRead:
    async def test_detail_of_a_finished_run(self, api: ApiHarness) -> None:
        run_id = await api.create()
        detail = await api.wait_for(run_id, "completed")

        assert detail["started_at"] and detail["finished_at"]
        assert detail["error"] is None
        assert detail["report_available"] is True
        assert detail["plan"]["objective"]
        steps = {s["agent"]: s for s in detail["progress"]["steps"]}
        assert [s["agent"] for s in detail["progress"]["steps"]] == [
            "planner",
            "web_researcher",
            "synthesizer",
        ]
        assert all(s["status"] == "completed" for s in steps.values())
        assert steps["web_researcher"]["started"] == len(detail["sub_questions"]) == 2
        progress = detail["progress"]
        assert progress["sub_questions"] == {"total": 2, "completed": 2}
        assert progress["sources"] >= 1
        assert progress["passages"] >= progress["sources"]
        assert progress["last_event_seq"] > 5

    async def test_detail_while_running(self, api: ApiHarness) -> None:
        await hold(api)
        run_id = await api.create()
        await asyncio.wait_for(api.waiting.wait(), 10)

        detail = (await api.client.get(f"{BASE}/{run_id}")).json()

        assert detail["status"] == "running"
        assert detail["started_at"] and not detail["finished_at"]
        steps = {s["agent"]: s["status"] for s in detail["progress"]["steps"]}
        assert steps == {
            "planner": "running",
            "web_researcher": "pending",
            "synthesizer": "pending",
        }
        assert detail["report_available"] is False

        report = await api.client.get(f"{BASE}/{run_id}/report")
        assert report.status_code == 409
        assert report.json()["error"] == {
            "code": "report_not_ready",
            "message": "The run has not produced a report yet",
            "details": {"status": "running"},
        }

    async def test_list_is_paginated_newest_first_and_filterable(self, api: ApiHarness) -> None:
        ids = [await api.create(question=f"question {i}") for i in range(3)]
        for run_id in ids:
            await api.wait_for(run_id)

        page = (await api.client.get(BASE, params={"limit": 2})).json()
        assert page["total"] == 3
        assert page["limit"] == 2
        assert [item["run_id"] for item in page["items"]] == [ids[2], ids[1]]

        rest = (await api.client.get(BASE, params={"limit": 2, "offset": 2})).json()
        assert [item["run_id"] for item in rest["items"]] == [ids[0]]

        done = (await api.client.get(BASE, params={"status": "completed"})).json()
        assert done["total"] == 3
        none = (await api.client.get(BASE, params={"status": "failed"})).json()
        assert none == {"items": [], "total": 0, "limit": 20, "offset": 0}

        bad = await api.client.get(BASE, params={"limit": 0})
        assert bad.status_code == 422

    async def test_sources(self, api: ApiHarness) -> None:
        run_id = await api.create()
        await api.wait_for(run_id, "completed")

        body = (await api.client.get(f"{BASE}/{run_id}/sources")).json()

        assert body["run_id"] == run_id
        assert body["total"] == len(body["items"]) >= 1
        for item in body["items"]:
            assert item["url"].startswith("https://")
            assert item["passages"] >= 1
            assert 0.0 <= item["reliability_score"] <= 1.0
        assert any(item["cited"] for item in body["items"])

    async def test_usage(self, api: ApiHarness) -> None:
        run_id = await api.create()
        await api.wait_for(run_id, "completed")

        usage = (await api.client.get(f"{BASE}/{run_id}/usage")).json()

        assert set(usage["per_agent"]) >= {"planner", "synthesizer"}
        totals = usage["totals"]
        assert totals["calls"] == sum(a["calls"] for a in usage["per_agent"].values()) >= 2
        assert totals["tokens_in"] == sum(a["tokens_in"] for a in usage["per_agent"].values())
        assert totals["tokens_in"] > 0
        assert set(usage["per_model"]) == {api.settings.llm_model}
        assert usage["limits"] == {
            "max_tokens_per_run": api.settings.max_tokens_per_run,
            "max_cost_per_run_usd": api.settings.max_cost_per_run_usd,
        }

    @pytest.mark.parametrize("suffix", ["", "/report", "/sources", "/usage", "/events"])
    async def test_unknown_run_is_404(self, api: ApiHarness, suffix: str) -> None:
        response = await api.client.get(f"{BASE}/{uuid.uuid4()}{suffix}")
        assert response.status_code == 404
        assert response.json()["error"]["code"] == "run_not_found"

    async def test_malformed_run_id_is_422(self, api: ApiHarness) -> None:
        response = await api.client.get(f"{BASE}/not-a-uuid")
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "validation_error"

    async def test_an_error_marking_the_run_running_still_ends_it(
        self, api: ApiHarness, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        recorder = api.services.recorder
        real = recorder.transition

        async def flaky(run_id: str, status: str, **kwargs: Any) -> bool:
            if status == "running":
                raise RuntimeError("connection reset")
            return await real(run_id, status, **kwargs)

        monkeypatch.setattr(recorder, "transition", flaky)
        run_id = await api.create()

        detail = await api.wait_for(run_id)
        assert detail["status"] == "failed"
        assert detail["error"] == "RuntimeError: connection reset"

    async def test_recording_the_outcome_is_retried(
        self, api: ApiHarness, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(research_service, "_FINISH_RETRY_DELAY", 0.0)
        recorder = api.services.recorder
        real = recorder.transition
        failures = {"completed": 2}

        async def flaky(run_id: str, status: str, **kwargs: Any) -> bool:
            if failures.get(status):
                failures[status] -= 1
                raise RuntimeError("database blip")
            return await real(run_id, status, **kwargs)

        monkeypatch.setattr(recorder, "transition", flaky)
        run_id = await api.create()

        assert (await api.wait_for(run_id))["status"] == "completed"
        assert failures["completed"] == 0

    async def test_failed_run(self, api: ApiHarness, monkeypatch: pytest.MonkeyPatch) -> None:
        async def boom(*args: Any, **kwargs: Any) -> None:
            raise RuntimeError("model provider down")

        monkeypatch.setattr(research_service, "execute_run", boom)
        run_id = await api.create()

        detail = await api.wait_for(run_id)

        assert detail["status"] == "failed"
        assert detail["error"] == "RuntimeError: model provider down"
        report = await api.client.get(f"{BASE}/{run_id}/report")
        assert report.status_code == 404
        assert report.json()["error"]["code"] == "report_not_found"
        last = (await stream(api, run_id))[-1]["data"]
        assert last["payload"] == {"status": "failed", "error": "RuntimeError: model provider down"}


# ===========================================================================
# Events (SSE)
# ===========================================================================


class TestEvents:
    async def test_stream_replays_the_run_and_ends_on_its_terminal_event(
        self, api: ApiHarness
    ) -> None:
        run_id = await api.create()
        await api.wait_for(run_id, "completed")

        events = await stream(api, run_id)

        seqs = [int(e["id"]) for e in events]
        assert seqs == list(range(1, len(events) + 1))
        assert all(e["event"] == e["data"]["type"] for e in events)
        assert all(e["data"]["seq"] == int(e["id"]) for e in events)
        assert "retry" in events[0] and all("retry" not in e for e in events[1:])

        statuses = [e["data"]["payload"]["status"] for e in events if e["event"] == "run_status"]
        assert statuses == ["queued", "running", "completed"]
        assert events[-1]["event"] == "run_status"
        types = {e["event"] for e in events}
        assert {"agent_started", "agent_finished", "source_found", "report_ready"} <= types

    async def test_last_event_id_resumes_after_that_event(self, api: ApiHarness) -> None:
        run_id = await api.create()
        await api.wait_for(run_id, "completed")
        everything = await stream(api, run_id)
        middle = len(everything) // 2

        by_header = await stream(api, run_id, headers={"Last-Event-ID": str(middle)})
        by_query = await stream(api, run_id, params={"after": middle})

        expected = [e["id"] for e in everything[middle:]]
        assert [e["id"] for e in by_header] == expected
        assert [e["id"] for e in by_query] == expected

    async def test_resuming_after_the_end_returns_only_the_final_status(
        self, api: ApiHarness
    ) -> None:
        run_id = await api.create()
        await api.wait_for(run_id, "completed")
        last = (await stream(api, run_id))[-1]

        again = await stream(api, run_id, headers={"Last-Event-ID": last["id"]})

        assert len(again) == 1
        assert again[0]["data"]["payload"]["status"] == "completed"

    async def test_cursors_beyond_any_seq(self, api: ApiHarness) -> None:
        run_id = await api.create()
        await api.wait_for(run_id, "completed")

        huge = await stream(api, run_id, headers={"Last-Event-ID": str(10**30)})
        assert [e["data"]["payload"]["status"] for e in huge] == ["completed"]

        assert (await api.client.get(f"{BASE}/{run_id}/events", params={"after": 2**31})).status_code == 422
        assert (await api.client.get(BASE, params={"offset": 2**31})).status_code == 422

    async def test_events_written_after_connecting_arrive_live(self, api: ApiHarness) -> None:
        await hold(api)
        run_id = await api.create()
        await asyncio.wait_for(api.waiting.wait(), 10)

        reader = asyncio.create_task(stream(api, run_id))
        notifier = api.services.recorder.notifier
        await api.until(lambda: notifier.listener_count(run_id) == 1)
        async with api.session_factory() as session:
            seen_before = (
                await session.execute(
                    select(func.max(RunEvent.seq)).where(RunEvent.run_id == uuid.UUID(run_id))
                )
            ).scalar_one()
        assert not reader.done()

        api.gate.set()
        events = await asyncio.wait_for(reader, 15)

        seqs = [int(e["id"]) for e in events]
        assert seqs == list(range(1, len(events) + 1))
        live = [e for e in events if int(e["id"]) > seen_before]
        assert any(e["event"] == "report_ready" for e in live)
        assert events[-1]["data"]["payload"]["status"] == "completed"
        assert notifier.listener_count(run_id) == 0

    async def test_stream_of_a_deleted_run_ends(self, api: ApiHarness) -> None:
        await hold(api)
        run_id = await api.create()
        await asyncio.wait_for(api.waiting.wait(), 10)
        reader = asyncio.create_task(stream(api, run_id))
        await api.until(lambda: api.services.recorder.notifier.listener_count(run_id) == 1)

        assert (await api.client.delete(f"{BASE}/{run_id}")).status_code == 204

        events = await asyncio.wait_for(reader, 10)
        assert events  # what was written before the delete
        assert events[0]["data"]["payload"]["status"] == "queued"


# ===========================================================================
# Cancel
# ===========================================================================


class TestCancel:
    async def test_cancel_stops_a_running_run(self, api: ApiHarness) -> None:
        await hold(api)
        run_id = await api.create()
        await asyncio.wait_for(api.waiting.wait(), 10)

        response = await api.client.post(f"{BASE}/{run_id}/cancel")

        assert response.status_code == 200
        assert response.json() == {"run_id": run_id, "status": "cancelled"}
        assert not api.services.runner.is_active(run_id)
        detail = (await api.client.get(f"{BASE}/{run_id}")).json()
        assert detail["status"] == "cancelled"
        assert detail["finished_at"]
        steps = {s["agent"]: s["status"] for s in detail["progress"]["steps"]}
        assert steps == {
            "planner": "incomplete",
            "web_researcher": "skipped",
            "synthesizer": "skipped",
        }
        events = await stream(api, run_id)
        assert events[-1]["data"]["payload"] == {"status": "cancelled", "error": None}
        assert (await api.client.get(f"{BASE}/{run_id}/report")).status_code == 404

        # Releasing the held model call must not resurrect it.
        api.gate.set()
        await asyncio.sleep(0.1)
        assert (await api.client.get(f"{BASE}/{run_id}")).json()["status"] == "cancelled"

    async def test_cancel_a_queued_run(self, make_api: MakeApi) -> None:
        api = await make_api(max_concurrent_runs=1)
        await hold(api)
        first = await api.create(question="first")
        await asyncio.wait_for(api.waiting.wait(), 10)
        second = await api.create(question="second")
        assert (await api.client.get(f"{BASE}/{second}")).json()["status"] == "queued"

        response = await api.client.post(f"{BASE}/{second}/cancel")

        assert response.status_code == 200
        assert response.json()["status"] == "cancelled"
        api.gate.set()
        assert (await api.wait_for(first))["status"] == "completed"
        detail = (await api.client.get(f"{BASE}/{second}")).json()
        assert detail["status"] == "cancelled"
        assert detail["started_at"] is None
        assert all(s["started"] == 0 for s in detail["progress"]["steps"])

    async def test_cancel_of_a_run_that_completes_meanwhile_is_409(
        self, api: ApiHarness, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        run_id = await api.services.recorder.create_run(question="racing", mode="quick")

        async def completes_first(key: str, *, wait: float) -> bool:
            await api.services.recorder.transition(key, "completed")
            return True

        monkeypatch.setattr(api.services.runner, "cancel", completes_first)
        response = await api.client.post(f"{BASE}/{run_id}/cancel")

        assert response.status_code == 409
        assert response.json()["error"]["details"] == {"status": "completed"}

    async def test_cancel_a_finished_run_is_409(self, api: ApiHarness) -> None:
        run_id = await api.create()
        await api.wait_for(run_id, "completed")

        response = await api.client.post(f"{BASE}/{run_id}/cancel")

        assert response.status_code == 409
        assert response.json()["error"]["code"] == "run_not_active"
        assert response.json()["error"]["details"] == {"status": "completed"}

    async def test_cancel_a_run_with_no_task_in_this_process(self, api: ApiHarness) -> None:
        run_id = await api.services.recorder.create_run(question="orphan", mode="quick")

        response = await api.client.post(f"{BASE}/{run_id}/cancel")

        assert response.json() == {"run_id": run_id, "status": "cancelled"}

    async def test_cancel_unknown_run_is_404(self, api: ApiHarness) -> None:
        response = await api.client.post(f"{BASE}/{uuid.uuid4()}/cancel")
        assert response.status_code == 404


# ===========================================================================
# Delete
# ===========================================================================


class TestDelete:
    async def test_delete_removes_the_run_and_all_its_data(self, api: ApiHarness) -> None:
        run_id = await api.create()
        await api.wait_for(run_id, "completed")
        children = (SubQuestion, Source, Passage, Report, RunEvent, LLMUsage)
        assert all([await count_rows(api, model, run_id) for model in children])

        response = await api.client.delete(f"{BASE}/{run_id}")

        assert response.status_code == 204
        assert response.content == b""
        assert (await api.client.get(f"{BASE}/{run_id}")).status_code == 404
        assert [await count_rows(api, model, run_id) for model in children] == [0] * 6

    async def test_delete_an_active_run_cancels_it_first(self, api: ApiHarness) -> None:
        await hold(api)
        run_id = await api.create()
        await asyncio.wait_for(api.waiting.wait(), 10)

        assert (await api.client.delete(f"{BASE}/{run_id}")).status_code == 204

        assert not api.services.runner.is_active(run_id)
        assert (await api.client.get(f"{BASE}/{run_id}")).status_code == 404

    async def test_delete_unknown_run_is_404(self, api: ApiHarness) -> None:
        assert (await api.client.delete(f"{BASE}/{uuid.uuid4()}")).status_code == 404


# ===========================================================================
# Security, limits, errors
# ===========================================================================


class TestSecurity:
    async def test_api_key(self, make_api: MakeApi) -> None:
        api = await make_api(api_key="s3cret")

        missing = await api.client.get(BASE)
        wrong = await api.client.get(BASE, headers={"X-API-Key": "nope"})
        right = await api.client.get(BASE, headers={"X-API-Key": "s3cret"})

        assert missing.status_code == 401
        assert missing.json()["error"]["code"] == "missing_api_key"
        assert wrong.status_code == 403
        assert wrong.json()["error"]["code"] == "invalid_api_key"
        assert right.status_code == 200
        # Probes stay open for load balancers.
        assert (await api.client.get("/health")).status_code == 200

    async def test_failed_keys_are_throttled_per_ip_apart_from_valid_ones(
        self, make_api: MakeApi
    ) -> None:
        api = await make_api(api_key="s3cret", rate_limit_requests=2)

        first = await api.client.get(BASE, headers={"X-API-Key": "guess-1"})
        second = await api.client.get(BASE)
        third = await api.client.get(BASE, headers={"X-API-Key": "guess-2"})

        assert (first.status_code, second.status_code) == (403, 401)
        assert third.status_code == 429
        assert "failed API key attempts" in third.json()["error"]["message"]
        assert (await api.client.get(BASE, headers={"X-API-Key": "s3cret"})).status_code == 200

    async def test_auth_is_checked_before_the_database(self) -> None:
        app = create_app(settings=make_test_settings(api_key="s3cret"))
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            anonymous = await client.get(BASE)
            authed = await client.get(BASE, headers={"X-API-Key": "s3cret"})

        assert anonymous.status_code == 401
        assert authed.status_code == 503

    async def test_request_rate_limit(self, make_api: MakeApi) -> None:
        api = await make_api(rate_limit_requests=3)
        for _ in range(3):
            assert (await api.client.get(BASE)).status_code == 200

        response = await api.client.get(BASE)

        assert response.status_code == 429
        error = response.json()["error"]
        assert error["code"] == "rate_limited"
        assert int(response.headers["retry-after"]) >= 1
        assert error["details"]["retry_after_seconds"] == int(response.headers["retry-after"])

    async def test_run_creation_has_its_own_limit(self, make_api: MakeApi) -> None:
        api = await make_api(rate_limit_runs=1)
        await api.wait_for(await api.create())

        response = await api.client.post(BASE, json={"question": "again"})

        assert response.status_code == 429
        assert "research runs" in response.json()["error"]["message"]
        assert (await api.client.get(BASE)).status_code == 200

    async def test_oversized_body_is_413(self, make_api: MakeApi) -> None:
        api = await make_api(max_request_body_bytes=200)

        response = await api.client.post(BASE, json={"question": "x" * 500})

        assert response.status_code == 413
        assert response.json()["error"]["code"] == "payload_too_large"

    async def test_oversized_chunked_body_is_413(self, make_api: MakeApi) -> None:
        api = await make_api(max_request_body_bytes=200)

        async def chunks() -> AsyncIterator[bytes]:
            yield b'{"question": "'
            for _ in range(10):
                yield b"x" * 50
            yield b'"}'

        response = await api.client.post(
            BASE, content=chunks(), headers={"content-type": "application/json"}
        )

        assert response.status_code == 413

    async def test_small_chunked_body_is_accepted(self, api: ApiHarness) -> None:
        async def chunks() -> AsyncIterator[bytes]:
            yield b'{"question": '
            yield b'"chunked question"}'

        response = await api.client.post(
            BASE, content=chunks(), headers={"content-type": "application/json"}
        )

        assert response.status_code == 202
        await api.wait_for(response.json()["run_id"])


class TestErrorEnvelope:
    async def test_unknown_path(self, api: ApiHarness) -> None:
        response = await api.client.get("/api/v1/nope")
        assert response.status_code == 404
        assert response.json() == {
            "error": {"code": "not_found", "message": "Not Found", "details": None}
        }

    async def test_wrong_method(self, api: ApiHarness) -> None:
        response = await api.client.put(BASE)
        assert response.status_code == 405
        assert response.json()["error"]["code"] == "method_not_allowed"

    async def test_invalid_json(self, api: ApiHarness) -> None:
        response = await api.client.post(
            BASE, content=b"{not json", headers={"content-type": "application/json"}
        )
        assert response.status_code == 422
        assert response.json()["error"]["code"] == "validation_error"

    async def test_research_needs_a_database(self) -> None:
        app = create_app(settings=make_test_settings())
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get(BASE)
            ready = await client.get("/ready")

        assert response.status_code == 503
        assert response.json()["error"]["code"] == "database_not_configured"
        assert ready.json()["checks"]["database"] == "not_configured"


# ===========================================================================
# Lifecycle: startup sweep, shutdown, readiness
# ===========================================================================


class TestLifecycle:
    async def test_startup_marks_unfinished_runs_failed(self, api: ApiHarness) -> None:
        recorder = api.services.recorder
        queued = await recorder.create_run(question="left queued", mode="quick")
        running = await recorder.create_run(question="left running", mode="quick")
        await recorder.transition(running, "running")
        done = await api.create()
        await api.wait_for(done, "completed")

        recovered = await api.services.startup()

        assert set(recovered) == {queued, running}
        for run_id in (queued, running):
            detail = (await api.client.get(f"{BASE}/{run_id}")).json()
            assert detail["status"] == "failed"
            assert detail["error"] == INTERRUPTED_ERROR
            last = (await stream(api, run_id))[-1]["data"]
            assert last["payload"] == {"status": "failed", "error": INTERRUPTED_ERROR}
        assert (await api.client.get(f"{BASE}/{done}")).json()["status"] == "completed"

    async def test_lifespan_runs_the_startup_sweep(self, api: ApiHarness) -> None:
        orphan = await api.services.recorder.create_run(question="orphan", mode="quick")

        async with api.app.router.lifespan_context(api.app):
            detail = (await api.client.get(f"{BASE}/{orphan}")).json()

        assert detail["status"] == "failed"
        assert not api.services.runner.accepting  # shut down on exit

    async def test_sweep_can_be_disabled(self, make_api: MakeApi) -> None:
        api = await make_api(recover_orphaned_runs=False)
        orphan = await api.services.recorder.create_run(question="orphan", mode="quick")

        assert await api.services.startup() == []
        assert (await api.client.get(f"{BASE}/{orphan}")).json()["status"] == "queued"

    async def test_shutdown_interrupts_running_runs(self, api: ApiHarness) -> None:
        await hold(api)
        run_id = await api.create()
        await asyncio.wait_for(api.waiting.wait(), 10)

        await api.services.shutdown()

        detail = (await api.client.get(f"{BASE}/{run_id}")).json()
        assert detail["status"] == "failed"
        assert detail["error"] == INTERRUPTED_ERROR
        refused = await api.client.post(BASE, json={"question": "after shutdown"})
        assert refused.status_code == 503

    async def test_ready_pings_the_database(self, api: ApiHarness) -> None:
        body = (await api.client.get("/ready")).json()
        assert body["status"] == "ready"
        assert body["checks"]["database"] == "ok"
        assert body["runs"] == {"running": 0, "pending": 0}

    async def test_ready_reports_a_broken_database(self, api: ApiHarness) -> None:
        await api.engine.dispose()
        broken = api.services.session_factory
        broken.configure(bind=create_bad_engine())

        body = (await api.client.get("/ready")).json()

        assert body["status"] == "degraded"
        assert body["checks"]["database"] == "error"


def create_bad_engine() -> Any:
    from sqlalchemy.ext.asyncio import create_async_engine

    return create_async_engine("sqlite+aiosqlite:////nonexistent-dir/x/y.db")
