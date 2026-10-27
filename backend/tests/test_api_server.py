"""
M4 tests — the API behind a real uvicorn server on a local socket.

The in-process test client buffers a whole response, so it cannot show that
SSE events arrive one by one. Here a real HTTP connection reads the stream
while the run is still held, which shows events are delivered live, and a
client hanging up is shown to release the server-side listener.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Any

import httpx
import uvicorn

from .conftest import ApiHarness

MakeApi = Callable[..., Awaitable[ApiHarness]]


@asynccontextmanager
async def serve(api: ApiHarness) -> AsyncIterator[httpx.AsyncClient]:
    # A socket the client dropped mid-stream can keep asyncio's Server.wait_closed()
    # waiting on Windows after the app has finished with it; bound the wait.
    config = uvicorn.Config(
        api.app,
        host="127.0.0.1",
        port=0,
        log_level="warning",
        lifespan="off",
        timeout_graceful_shutdown=2,
    )
    server = uvicorn.Server(config)
    task = asyncio.create_task(server.serve())
    try:
        async with asyncio.timeout(10):
            while not server.started:
                await asyncio.sleep(0.01)
        port = server.servers[0].sockets[0].getsockname()[1]
        async with httpx.AsyncClient(
            base_url=f"http://127.0.0.1:{port}", trust_env=False, timeout=15
        ) as client:
            yield client
    finally:
        server.should_exit = True
        await asyncio.wait_for(task, 10)


async def sse_events(response: httpx.Response) -> AsyncIterator[dict[str, Any]]:
    """Parse events off a live response as they arrive."""
    fields: dict[str, Any] = {}
    async for line in response.aiter_lines():
        if line == "":
            if fields:
                yield fields
                fields = {}
            continue
        if line.startswith(":"):
            continue
        name, _, value = line.partition(":")
        value = value[1:] if value.startswith(" ") else value
        fields[name] = json.loads(value) if name == "data" else value


async def test_events_arrive_live_over_a_real_connection(make_api: MakeApi) -> None:
    api = await make_api()
    api.gate.clear()
    api.waiting.clear()

    async with serve(api) as client:
        created = await client.post("/api/v1/research", json={"question": "RAG versus fine-tuning"})
        assert created.status_code == 202
        run_id = created.json()["run_id"]
        await asyncio.wait_for(api.waiting.wait(), 10)

        async with client.stream("GET", f"/api/v1/research/{run_id}/events") as response:
            assert response.status_code == 200
            assert response.headers["content-type"].startswith("text/event-stream")
            assert response.headers["cache-control"] == "no-cache"
            events = sse_events(response)

            # Delivered while the run is still held: nothing here waits for it to end.
            early = [await asyncio.wait_for(anext(events), 5) for _ in range(3)]
            assert [e["event"] for e in early] == ["run_status", "run_status", "agent_started"]
            assert [e["data"]["payload"].get("status") for e in early[:2]] == ["queued", "running"]
            assert (await client.get(f"/api/v1/research/{run_id}")).json()["status"] == "running"

            api.gate.set()
            rest = [e async for e in events]

        assert rest[-1]["event"] == "run_status"
        assert rest[-1]["data"]["payload"]["status"] == "completed"
        seqs = [int(e["id"]) for e in early + rest]
        assert seqs == list(range(1, len(seqs) + 1))

        report = await client.get(f"/api/v1/research/{run_id}/report")
        assert report.status_code == 200
        assert report.json()["content"]["sources"]


async def test_a_client_hanging_up_releases_the_stream(make_api: MakeApi) -> None:
    api = await make_api()
    api.gate.clear()
    api.waiting.clear()
    notifier = api.services.recorder.notifier

    async with serve(api) as client:
        run_id = (await client.post("/api/v1/research", json={"question": "q"})).json()["run_id"]
        await asyncio.wait_for(api.waiting.wait(), 10)

        async with client.stream("GET", f"/api/v1/research/{run_id}/events") as response:
            await asyncio.wait_for(anext(sse_events(response)), 5)
            assert notifier.listener_count(run_id) == 1
        # Leaving the block closes the connection mid-stream.

        await api.until(lambda: notifier.listener_count(run_id) == 0, timeout=5)
        cancelled = await client.post(f"/api/v1/research/{run_id}/cancel")
        assert cancelled.json()["status"] == "cancelled"
