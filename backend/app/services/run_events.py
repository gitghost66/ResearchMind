"""
Run events and run status: the one place both are written.

Every event — agent progress from the graph and status changes from the
backend — goes through :class:`RunEventRecorder`, which

* numbers it with the run's next ``seq`` and commits it while holding a
  per-run lock, so commit order equals ``seq`` order and a reader polling
  ``seq > n`` can never skip an event that committed late;
* then wakes SSE listeners in this process through :class:`EventNotifier`.

A status change and its ``run_status`` event commit in one transaction, and a
run that reached a terminal status never changes again — so a late write (say,
"completed" racing a cancel) cannot flip a finished run.

The database is the source of truth: a woken listener re-reads ``seq > n``; the
notifier only spares it from polling. Events written by another process (for
example ``run_mvp.py``) still reach a stream through its fallback poll. M8 swaps
the notifier for Postgres LISTEN/NOTIFY so several API workers can share runs.

No prompt text or agent logic here (AGENTS.md rule 2).
"""

from __future__ import annotations

import asyncio
import logging
import uuid
import weakref
from collections.abc import Collection, Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.models.research import ResearchRun, RunEvent
from app.repositories.research_repository import RunEventRepository

logger = logging.getLogger(__name__)

__all__ = [
    "ACTIVE_STATUSES",
    "INTERRUPTED_ERROR",
    "RUNNER_AGENT",
    "RUN_STATUS_EVENT",
    "TERMINAL_STATUSES",
    "EventNotifier",
    "RunEventRecorder",
    "event_to_dict",
]

ACTIVE_STATUSES: frozenset[str] = frozenset({"queued", "running"})
TERMINAL_STATUSES: frozenset[str] = frozenset({"completed", "failed", "cancelled"})

# Status changes are events too, so a stream sees the whole run in one place.
RUN_STATUS_EVENT = "run_status"
RUNNER_AGENT = "runner"

INTERRUPTED_ERROR = "Interrupted: the server stopped before the run finished."

# A seq collision means another recorder (or process) got there first;
# re-read and retry. Within one recorder the per-run lock prevents collisions.
_MAX_WRITE_ATTEMPTS = 5


def _uuid(value: str | uuid.UUID) -> uuid.UUID:
    return value if isinstance(value, uuid.UUID) else uuid.UUID(str(value))


def event_to_dict(event: RunEvent) -> dict[str, Any]:
    """The JSON shape of an event, as the API and the stream send it."""
    return {
        "seq": event.seq,
        "run_id": str(event.run_id),
        "ts": event.ts.isoformat() if event.ts else None,
        "agent": event.agent,
        "type": event.type,
        "payload": dict(event.payload or {}),
    }


class EventNotifier:
    """In-process wake-ups for streams waiting on a run's next event."""

    def __init__(self) -> None:
        self._listeners: dict[str, set[asyncio.Event]] = {}

    def notify(self, run_id: str) -> None:
        for listener in self._listeners.get(str(run_id), ()):
            listener.set()

    @contextmanager
    def listen(self, run_id: str) -> Iterator[asyncio.Event]:
        """An event that is set whenever *run_id* gets a new event."""
        key = str(run_id)
        listener = asyncio.Event()
        self._listeners.setdefault(key, set()).add(listener)
        try:
            yield listener
        finally:
            listeners = self._listeners.get(key)
            if listeners is not None:
                listeners.discard(listener)
                if not listeners:
                    del self._listeners[key]

    def listener_count(self, run_id: str) -> int:
        return len(self._listeners.get(str(run_id), ()))


class RunEventRecorder:
    """
    Writes a run's events and status changes, one at a time per run.

    One recorder should serve the whole process: the per-run locks live here.
    A second recorder (or process) writing the same run is still safe — the
    ``(run_id, seq)`` unique index turns a collision into a retry — just not
    guaranteed to commit in ``seq`` order.
    """

    def __init__(
        self,
        session_factory: async_sessionmaker[AsyncSession],
        notifier: EventNotifier | None = None,
    ) -> None:
        self._session_factory = session_factory
        self.notifier = notifier or EventNotifier()
        # A lock lives only while someone holds or awaits it.
        self._locks: weakref.WeakValueDictionary[str, asyncio.Lock] = (
            weakref.WeakValueDictionary()
        )

    def _lock(self, run_id: str) -> asyncio.Lock:
        lock = self._locks.get(run_id)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[run_id] = lock
        return lock

    async def record(
        self,
        *,
        run_id: str,
        agent: str,
        event_type: str,
        payload: dict[str, Any] | None = None,
    ) -> int:
        """Append one event and return its ``seq``."""
        key = str(_uuid(run_id))
        lock = self._lock(key)
        async with lock:
            for attempt in range(1, _MAX_WRITE_ATTEMPTS + 1):
                try:
                    async with self._session_factory() as session:
                        event = await RunEventRepository(session).log_event(
                            key, agent, event_type, dict(payload or {})
                        )
                        seq = event.seq
                        await session.commit()
                    break
                except IntegrityError:
                    if attempt == _MAX_WRITE_ATTEMPTS:
                        raise
        self.notifier.notify(key)
        return seq

    async def create_run(
        self,
        *,
        question: str,
        mode: str,
        user_context: dict[str, Any] | None = None,
        options: dict[str, Any] | None = None,
    ) -> str:
        """Insert a ``queued`` run and its first ``run_status`` event."""
        async with self._session_factory() as session:
            run = ResearchRun(
                question=question,
                mode=mode,
                status="queued",
                user_context=dict(user_context or {}),
                options=dict(options or {}),
            )
            session.add(run)
            await session.flush()
            session.add(
                RunEvent(
                    run_id=run.id,
                    seq=1,
                    agent=RUNNER_AGENT,
                    type=RUN_STATUS_EVENT,
                    payload={"status": "queued", "error": None},
                )
            )
            await session.commit()
            return str(run.id)

    async def transition(
        self,
        run_id: str,
        status: str,
        *,
        error: str | None = None,
        from_statuses: Collection[str] = ACTIVE_STATUSES,
    ) -> bool:
        """
        Move the run to *status* if it is currently in *from_statuses*.

        Returns whether it moved. The update and its ``run_status`` event
        commit together; a run that is missing, or not in *from_statuses*, is
        left alone and no event is written.
        """
        key = str(_uuid(run_id))
        values: dict[str, Any] = {"status": status}
        if status == "running":
            values["started_at"] = datetime.now(UTC)
        if status in TERMINAL_STATUSES:
            values["finished_at"] = datetime.now(UTC)
        if error is not None:
            values["error"] = error

        lock = self._lock(key)
        async with lock:
            for attempt in range(1, _MAX_WRITE_ATTEMPTS + 1):
                try:
                    async with self._session_factory() as session:
                        result = await session.execute(
                            update(ResearchRun)
                            .where(
                                ResearchRun.id == _uuid(key),
                                ResearchRun.status.in_(list(from_statuses)),
                            )
                            .values(**values)
                        )
                        if not getattr(result, "rowcount", 0):
                            await session.rollback()
                            return False
                        await RunEventRepository(session).log_event(
                            key, RUNNER_AGENT, RUN_STATUS_EVENT, {"status": status, "error": error}
                        )
                        await session.commit()
                    break
                except IntegrityError:
                    if attempt == _MAX_WRITE_ATTEMPTS:
                        raise
        self.notifier.notify(key)
        logger.info("Run %s is %s", key, status, extra={"run_id": key})
        return True
