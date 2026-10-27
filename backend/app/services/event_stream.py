"""
A run's events as a live stream, for the SSE endpoint.

:func:`stream_run_events` replays the events after a cursor, then yields new
ones as they are written, and ends after the run's terminal ``run_status``
event. Reading always goes to the database by ``seq``, so replay, live
delivery, and reconnects (``Last-Event-ID``) are one code path:

1. register with the notifier *before* reading, so an event committed between
   the read and the wait still wakes the stream;
2. read ``seq > cursor`` in a fresh, short session (a stream can outlive any
   request session) and yield;
3. wait for a wake-up, or ``poll_interval`` for events written by another
   process, and repeat.

The stream also ends if the run is deleted. A run that finished before M4 has
no terminal ``run_status`` event; once its events run out the stream ends with
one built from the run row.

A client hanging up cancels the stream, often in the middle of a read. Each
read therefore runs as its own task behind ``asyncio.shield``: the stream stops
at once, while the read finishes and returns its connection to the pool intact
instead of being torn down mid-query.
"""

from __future__ import annotations

import asyncio
import contextlib
import uuid
from collections.abc import AsyncIterator
from typing import Any

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.models.research import ResearchRun
from app.repositories.research_repository import RunEventRepository
from app.services.run_events import (
    RUN_STATUS_EVENT,
    RUNNER_AGENT,
    TERMINAL_STATUSES,
    EventNotifier,
    event_to_dict,
)

__all__ = ["is_terminal_event", "stream_run_events"]

# Reads a cancelled stream left running; held so they are not garbage collected.
_detached_reads: set[asyncio.Task[Any]] = set()

_RunRow = tuple[str, str | None]


def is_terminal_event(event: dict[str, Any]) -> bool:
    return (
        event.get("type") == RUN_STATUS_EVENT
        and (event.get("payload") or {}).get("status") in TERMINAL_STATUSES
    )


async def stream_run_events(
    run_id: str,
    *,
    session_factory: async_sessionmaker[AsyncSession],
    notifier: EventNotifier,
    after_seq: int = 0,
    poll_interval: float = 2.0,
    batch_size: int = 200,
) -> AsyncIterator[dict[str, Any]]:
    """Yield the run's events with ``seq > after_seq``, live, until the run ends."""
    key = str(uuid.UUID(str(run_id)))
    cursor = max(0, after_seq)
    # Set when the run looked finished but no terminal event was read yet:
    # look once more (the event commits with the status) before giving up.
    settling = False

    with notifier.listen(key) as wakeup:
        while True:
            wakeup.clear()
            read = asyncio.ensure_future(_read(session_factory, key, cursor, batch_size))
            _detached_reads.add(read)
            read.add_done_callback(_settle_read)
            events, run_row = await asyncio.shield(read)

            for event in events:
                cursor = event["seq"]
                yield event
                if is_terminal_event(event):
                    return
            if events:
                settling = False
                continue  # a full batch may have more behind it

            if run_row is None:
                return  # the run was deleted
            status, error = run_row
            if status in TERMINAL_STATUSES:
                if settling:
                    yield {
                        "seq": cursor,
                        "run_id": key,
                        "ts": None,
                        "agent": RUNNER_AGENT,
                        "type": RUN_STATUS_EVENT,
                        "payload": {"status": status, "error": error},
                    }
                    return
                settling = True
                continue

            with contextlib.suppress(TimeoutError):
                await asyncio.wait_for(wakeup.wait(), timeout=poll_interval)


async def _read(
    session_factory: async_sessionmaker[AsyncSession],
    run_id: str,
    after_seq: int,
    limit: int,
) -> tuple[list[dict[str, Any]], _RunRow | None]:
    """Events after *after_seq*; when there are none, also the run's status and error."""
    async with session_factory() as session:
        rows = await RunEventRepository(session).list_by_run(
            run_id, after_seq=after_seq, limit=limit
        )
        events = [event_to_dict(row) for row in rows]
        if events:
            return events, None
        found = await session.execute(
            select(ResearchRun.status, ResearchRun.error).where(ResearchRun.id == uuid.UUID(run_id))
        )
        first = found.first()
        return [], ((first[0], first[1]) if first is not None else None)


def _settle_read(task: asyncio.Task[Any]) -> None:
    _detached_reads.discard(task)
    # A read whose stream is gone has no one to raise to; retrieve its error
    # so asyncio does not warn about it.
    if not task.cancelled():
        task.exception()
