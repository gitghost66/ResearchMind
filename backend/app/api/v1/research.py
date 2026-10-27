"""
Research endpoints (PRD s9): ``/api/v1/research``.

Routes stay thin: validation, business rules, and queries live in
``ResearchService``; the stream in ``event_stream``. Every route requires the
API key and counts against the caller's rate limit; creating a run also counts
against the tighter run limit. Errors use the ``{"error": {...}}`` envelope.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterable
from typing import Annotated, Any

from fastapi import APIRouter, Depends, Header, Query, Response, status
from fastapi.sse import EventSourceResponse, ServerSentEvent

from app.container import AppServices
from app.core.deps import get_services, limit_requests, limit_run_creation
from app.schemas.research import (
    CancelResponse,
    ErrorEnvelope,
    ReportResponse,
    ResearchCreate,
    RunCreated,
    RunDetail,
    RunList,
    RunStatus,
    SourceList,
    UsageResponse,
)
from app.services.event_stream import stream_run_events

_ERRORS: dict[int | str, dict[str, Any]] = {
    code: {"model": ErrorEnvelope} for code in (401, 403, 404, 422, 429, 503)
}

router = APIRouter(
    prefix="/api/v1/research",
    tags=["research"],
    dependencies=[Depends(limit_requests)],
    responses=_ERRORS,
)

ServicesDep = Annotated[AppServices, Depends(get_services)]

# Largest value an INTEGER column holds: seq and offsets beyond it cannot exist,
# and binding them would fail in the database.
MAX_INT = 2**31 - 1


async def _existing_run(run_id: uuid.UUID, services: ServicesDep) -> uuid.UUID:
    """404 before a response starts (an SSE stream cannot change its status later)."""
    async with services.session_factory() as session:
        await services.research.get(session, run_id)
    return run_id


ExistingRunId = Annotated[uuid.UUID, Depends(_existing_run)]


@router.post(
    "",
    status_code=status.HTTP_202_ACCEPTED,
    dependencies=[Depends(limit_run_creation)],
    summary="Start a research run",
    responses={413: {"model": ErrorEnvelope, "description": "Request body too large."}},
)
async def create_research(
    body: ResearchCreate, services: ServicesDep, response: Response
) -> RunCreated:
    created = await services.research.create(body)
    response.headers["Location"] = created.links.self
    return created


@router.get("", summary="List research runs, newest first")
async def list_research(
    services: ServicesDep,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0, le=MAX_INT)] = 0,
    status: RunStatus | None = None,
    mode: str | None = None,
) -> RunList:
    async with services.session_factory() as session:
        return await services.research.list_runs(
            session, limit=limit, offset=offset, status=status, mode=mode
        )


@router.get("/{run_id}", summary="Run status, plan, and progress per step")
async def get_research(run_id: uuid.UUID, services: ServicesDep) -> RunDetail:
    async with services.session_factory() as session:
        run = await services.research.get(session, run_id)
        return await services.research.detail(session, run)


@router.get(
    "/{run_id}/events",
    response_class=EventSourceResponse,
    summary="Stream the run's events (Server-Sent Events)",
    description=(
        "Replays the run's events, then streams new ones live, and closes after the "
        "terminal `run_status` event. Each event's `event` is its type, its `id` is its "
        "`seq`, and its `data` is a `RunEventOut` object. Reconnect with `Last-Event-ID` "
        "(or `?after=`) to receive only what was missed."
    ),
)
async def stream_events(
    run_id: ExistingRunId,
    services: ServicesDep,
    last_event_id: Annotated[str | None, Header()] = None,
    after: Annotated[int | None, Query(ge=0, le=MAX_INT)] = None,
) -> AsyncIterable[ServerSentEvent]:
    cursor = _cursor(last_event_id, after)
    retry: int | None = services.settings.sse_retry_ms
    async for event in stream_run_events(
        str(run_id),
        session_factory=services.session_factory,
        notifier=services.recorder.notifier,
        after_seq=cursor,
        poll_interval=services.settings.sse_poll_interval_seconds,
    ):
        yield ServerSentEvent(data=event, event=event["type"], id=str(event["seq"]), retry=retry)
        retry = None  # once is enough


@router.get(
    "/{run_id}/report",
    summary="The run's report (JSON and markdown)",
    responses={409: {"model": ErrorEnvelope, "description": "The run is still in progress."}},
)
async def get_report(run_id: uuid.UUID, services: ServicesDep) -> ReportResponse:
    async with services.session_factory() as session:
        run = await services.research.get(session, run_id)
        return await services.research.report(session, run)


@router.get("/{run_id}/sources", summary="Sources the run stored")
async def get_sources(run_id: uuid.UUID, services: ServicesDep) -> SourceList:
    async with services.session_factory() as session:
        run = await services.research.get(session, run_id)
        return await services.research.sources(session, run)


@router.get("/{run_id}/usage", summary="Token and cost breakdown per agent")
async def get_usage(run_id: uuid.UUID, services: ServicesDep) -> UsageResponse:
    async with services.session_factory() as session:
        run = await services.research.get(session, run_id)
        return await services.research.usage(session, run)


@router.post(
    "/{run_id}/cancel",
    summary="Cancel a queued or running run",
    responses={
        200: {"description": "The run is cancelled."},
        202: {"description": "Cancellation was requested; the run is still stopping."},
        409: {"model": ErrorEnvelope, "description": "The run already finished."},
    },
)
async def cancel_research(
    run_id: uuid.UUID, services: ServicesDep, response: Response
) -> CancelResponse:
    result = await services.research.cancel(run_id)
    if result.status in ("queued", "running"):
        response.status_code = status.HTTP_202_ACCEPTED
    return result


@router.delete(
    "/{run_id}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Delete a run and all its data (cancels it first if active)",
)
async def delete_research(run_id: uuid.UUID, services: ServicesDep) -> Response:
    await services.research.delete(run_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


def _cursor(last_event_id: str | None, after: int | None) -> int:
    """
    Where to resume: ``Last-Event-ID`` if it parses, else ``?after=``, else the
    start. A header beyond any possible seq means "after everything".
    """
    if last_event_id:
        try:
            return min(max(0, int(last_event_id.strip())), MAX_INT)
        except ValueError:
            pass
    return after or 0
