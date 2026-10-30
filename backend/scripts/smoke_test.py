"""
Smoke tests against a running ResearchMind deployment (M8).

    # Every endpoint, with one real quick-mode run (a few model calls):
    python backend/scripts/smoke_test.py --base-url http://localhost:8080 --api-key KEY api

    # A run survives its workers being killed (SIGKILL: nothing released)
    # and restarted; another worker resumes it from its checkpoint:
    python backend/scripts/smoke_test.py --base-url http://localhost:8080 --api-key KEY restart \\
        --kill "docker compose -f docker-compose.prod.yml -f docker-compose.localdb.yml kill -s SIGKILL worker" \\
        --start "docker compose -f docker-compose.prod.yml -f docker-compose.localdb.yml up -d --no-deps worker"

Global options (--base-url, --api-key, --mode, --timeout) go before the
command. The API key defaults to $API_KEY. --kill and --start run in the
platform's shell (cmd.exe on Windows), so keep them free of shell-specific
syntax. Exits non-zero on the first failed check. Uses only httpx (a backend
dependency) and the API itself.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections.abc import Iterator
from typing import Any

import httpx

BASE = "/api/v1/research"
QUESTION = "What are the trade-offs of HNSW versus IVFFlat indexes in pgvector?"


class SmokeError(AssertionError):
    pass


def check(condition: bool, label: str, detail: Any = "") -> None:
    if not condition:
        raise SmokeError(f"{label}: {detail}")
    print(f"  ok  {label}")


def client(base_url: str, api_key: str, timeout: float = 30.0) -> httpx.Client:
    headers = {"X-API-Key": api_key} if api_key else {}
    return httpx.Client(base_url=base_url.rstrip("/"), headers=headers, timeout=timeout)


def events(http: httpx.Client, run_id: str, *, deadline: float) -> Iterator[dict[str, Any]]:
    """The run's SSE events, parsed, until the stream ends; reconnects with Last-Event-ID."""
    last = 0
    while time.monotonic() < deadline:
        headers = {"Last-Event-ID": str(last)} if last else {}
        try:
            with http.stream(
                "GET", f"{BASE}/{run_id}/events", headers=headers, timeout=httpx.Timeout(30, read=120)
            ) as response:
                response.raise_for_status()
                data: list[str] = []
                for line in response.iter_lines():
                    if line.startswith("data:"):
                        data.append(line[5:].strip())
                    elif not line and data:
                        event = json.loads("\n".join(data))
                        data = []
                        last = int(event["seq"])
                        yield event
                return  # the server closed the stream after the run ended
        except httpx.TransportError as exc:
            print(f"  ..  stream dropped ({exc}); reconnecting after event {last}")
            time.sleep(2)
    raise SmokeError(f"run {run_id} did not finish in time")


def follow(http: httpx.Client, run_id: str, *, timeout: float, until_agent: str | None = None) -> list[dict[str, Any]]:
    """Print progress; stop at the run's end, or when *until_agent* starts."""
    seen: list[dict[str, Any]] = []
    for event in events(http, run_id, deadline=time.monotonic() + timeout):
        seen.append(event)
        payload = event.get("payload") or {}
        if event["type"] == "run_status":
            extra = " (resumed)" if payload.get("resumed") else ""
            print(f"  ..  run {payload.get('status')}{extra}")
        elif event["type"] in ("agent_started", "agent_finished", "error"):
            print(f"  ..  {event['agent']}: {event['type']}")
        if until_agent and event["type"] == "agent_started" and event["agent"] == until_agent:
            break
    return seen


def create(http: httpx.Client, mode: str) -> str:
    response = http.post(BASE, json={"question": QUESTION, "mode": mode})
    check(response.status_code == 202, f"POST {BASE} -> 202", response.text)
    body = response.json()
    check(response.headers.get("location") == body["links"]["self"], "Location header points at the run")
    return str(body["run_id"])


# ---------------------------------------------------------------------------
# api
# ---------------------------------------------------------------------------


def smoke_api(args: argparse.Namespace) -> None:
    http = client(args.base_url, args.api_key)

    print("health")
    response = http.get("/health")
    check(response.status_code == 200 and response.json()["status"] == "healthy", "GET /health")
    ready = http.get("/ready").json()
    check(ready["status"] == "ready", "GET /ready is ready", ready)
    check(ready["checks"]["database"] == "ok", "the database answers", ready)

    print("auth and errors")
    if args.api_key:
        anonymous = httpx.get(f"{args.base_url.rstrip('/')}{BASE}", timeout=30)
        check(anonymous.status_code == 401, "no API key -> 401", anonymous.text)
        check(anonymous.json()["error"]["code"] == "missing_api_key", "error envelope", anonymous.text)
    missing = http.get(f"{BASE}/00000000-0000-0000-0000-000000000000")
    check(missing.status_code == 404 and missing.json()["error"]["code"] == "run_not_found", "unknown run -> 404")
    invalid = http.post(BASE, json={"question": ""})
    check(invalid.status_code == 422, "blank question -> 422", invalid.text)

    print(f"a {args.mode} run, streamed")
    run_id = create(http, args.mode)
    seen = follow(http, run_id, timeout=args.timeout)
    seqs = [e["seq"] for e in seen]
    check(seqs == sorted(set(seqs)), "event ids increase without gaps or repeats", seqs[:10])
    final = seen[-1]
    check(
        final["type"] == "run_status" and final["payload"]["status"] == "completed",
        "the stream ends on 'completed'",
        final,
    )

    print("reads")
    detail = http.get(f"{BASE}/{run_id}").json()
    check(detail["status"] == "completed" and detail["report_available"], "GET run: completed with a report")
    check(all(s["status"] == "completed" for s in detail["progress"]["steps"]), "every step completed", detail["progress"])
    report = http.get(f"{BASE}/{run_id}/report").json()
    check(bool(report["markdown"]) and bool(report["content"].get("sources")), "GET report: markdown and cited sources")
    sources = http.get(f"{BASE}/{run_id}/sources").json()
    check(sources["total"] > 0 and any(s["cited"] for s in sources["items"]), "GET sources: stored and cited")
    cited = {s["source_id"] for s in report["content"]["sources"]}
    check(cited <= {s["id"] for s in sources["items"]}, "every citation is a stored source")
    usage = http.get(f"{BASE}/{run_id}/usage").json()
    check(usage["totals"]["calls"] > 0, "GET usage: model calls recorded", usage["totals"])
    graph = http.get(f"{BASE}/{run_id}/graph").json()
    check(len(graph["nodes"]) > 1 and len(graph["edges"]) > 0, "GET graph: nodes and edges")
    listed = http.get(BASE, params={"limit": 50}).json()
    check(any(item["run_id"] == run_id for item in listed["items"]), "GET list includes the run")
    replay = http.get(f"{BASE}/{run_id}/events", headers={"Last-Event-ID": str(seqs[-2])})
    replayed = [int(line[3:].strip()) for line in replay.text.splitlines() if line.startswith("id:")]
    check(replayed == [seqs[-1]], "Last-Event-ID replays only what came after it", replayed)

    print("control")
    resume = http.post(f"{BASE}/{run_id}/resume")
    check(resume.status_code == 409, "resume a completed run -> 409", resume.text)
    again = http.post(f"{BASE}/{run_id}/cancel")
    check(again.status_code == 409, "cancel a completed run -> 409", again.text)
    second = create(http, "quick")
    cancelled = http.post(f"{BASE}/{second}/cancel")
    check(cancelled.status_code in (200, 202, 409), "cancel a new run", cancelled.text)
    status = http.get(f"{BASE}/{second}").json()["status"]
    check(status in ("cancelled", "completed"), "the cancelled run is settled", status)
    check(http.delete(f"{BASE}/{second}").status_code == 204, "DELETE -> 204")
    check(http.get(f"{BASE}/{second}").status_code == 404, "the deleted run is gone")

    print(f"\nAPI smoke test passed (run {run_id}).")


# ---------------------------------------------------------------------------
# restart
# ---------------------------------------------------------------------------


def shell(command: str) -> None:
    """Run an operator's command in the platform shell (cmd.exe on Windows)."""
    print(f"  $   {command}")
    result = subprocess.run(command, shell=True)  # noqa: S602 - an operator's own command
    check(result.returncode == 0, f"command succeeded: {command}", f"exit {result.returncode}")


def smoke_restart(args: argparse.Namespace) -> None:
    http = client(args.base_url, args.api_key)
    print(f"a {args.mode} run, killed during research")
    run_id = create(http, args.mode)
    follow(http, run_id, timeout=args.timeout, until_agent=args.kill_at)
    shell(args.kill)
    if args.start:
        shell(args.start)

    print("waiting for another worker to resume it (after its lease lapses)")
    seen = follow(http, run_id, timeout=args.timeout)
    final = seen[-1]
    check(final["payload"].get("status") == "completed", "the run completed", final)

    detail = http.get(f"{BASE}/{run_id}").json()
    check(detail["attempts"] >= 2, "it was resumed (attempts >= 2)", detail["attempts"])
    replay = list(events(http, run_id, deadline=time.monotonic() + 60))
    started = [e for e in replay if e["type"] == "agent_started" and e["agent"] == "planner"]
    check(len(started) == 1, "the planner ran once: finished steps were not redone", len(started))
    resumed = [e for e in replay if e["type"] == "run_status" and e["payload"].get("resumed")]
    check(bool(resumed), "the stream shows the resume", resumed)
    check(http.get(f"{BASE}/{run_id}/report").status_code == 200, "the report is there")
    print(f"\nRestart test passed (run {run_id}, attempt {detail['attempts']}).")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default=os.getenv("BASE_URL", "http://localhost:8080"))
    parser.add_argument("--api-key", default=os.getenv("API_KEY", ""))
    parser.add_argument("--mode", default="quick", choices=["quick", "deep", "academic", "decision"])
    parser.add_argument("--timeout", type=float, default=900.0, help="seconds to wait for a run")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("api", help="check every endpoint with one real run")
    restart = sub.add_parser("restart", help="kill the workers mid-run and check the run resumes")
    restart.add_argument("--kill", required=True, help="shell command that kills the workers")
    restart.add_argument("--start", default="", help="shell command that starts workers again")
    restart.add_argument("--kill-at", default="web_researcher", help="agent whose start triggers the kill")
    args = parser.parse_args()
    try:
        (smoke_api if args.command == "api" else smoke_restart)(args)
    except SmokeError as exc:
        print(f"  FAIL {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
