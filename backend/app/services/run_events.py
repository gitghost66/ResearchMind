"""
Run events and run status: the one place both are written.

Every event — agent progress from the graph and status changes from the
backend — goes through :class:`RunEventRecorder`, which

* numbers it with the run's next ``seq`` and commits it while holding a
  per-run lock, so commit order equals ``seq`` order and a reader polling
  ``seq > n`` can never skip an event that committed late;
* then wakes listeners through :class:`EventNotifier`.

A status change and its ``run_status`` event commit in one transaction. A run
that reached a terminal status changes again only through an explicit resume
(``failed`` -> ``queued``), so a late write (say, "completed" racing a cancel)
cannot flip a finished run. A worker finishing a run also names itself as the
owner, so a worker that lost its lease cannot settle a run another resumed.

The database is the source of truth: a woken listener re-reads ``seq > n``;
notifications only spare it from polling. On PostgreSQL every write also
sends ``NOTIFY`` in its transaction (delivered on commit): ``rm_run_events``
with the run ID, and for status changes ``rm_run_status`` with
``"<run_id>:<status>"``. ``app.services.pg_listener`` turns those back into
notifier calls in every API and worker process, which is how a stream served by
one process sees events written by a worker in another.

No prompt text or agent logic here (AGENTS.md rule 2).
"""

from __future__ import annotations

import asyncio
import logging
import uuid
import weakref
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from typing import Any

from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker
from sqlalchemy.sql import ColumnElement

from app.models.research import ResearchRun, RunEvent
from app.repositories.research_repository import RunEventRepository

logger = logging.getLogger(__name__)

__all__ = [
    "ACTIVE_STATUSES",
    "EVENTS_CHANNEL",
    "INTERRUPTED_ERROR",
    "RUNNER_AGENT",
    "RUN_STATUS_EVENT",
    "STATUS_CHANNEL",
    "TERMINAL_STATUSES",
    "EventNotifier",
    "RunEventRecorder",
    "StatusCallback",
    "event_to_dict",
]

ACTIVE_STATUSES: frozenset[str] = frozenset({"queued", "running"})
TERMINAL_STATUSES: frozenset[str] = frozenset({"completed", "failed", "cancelled"})

# Status changes are events too, so a stream sees the whole run in one place.
RUN_STATUS_EVENT = "run_status"
RUNNER_AGENT = "runner"

INTERRUPTED_ERROR = "Interrupted: the server stopped before the run finished."

# PostgreSQL NOTIFY channels (see the module docstring).
EVENTS_CHANNEL = "rm_run_events"
STATUS_CHANNEL = "rm_run_status"

# A seq collision means another recorder (or process) got there first;
# re-read and retry. Within one recorder the per-run lock prevents collisions.
_MAX_WRITE_ATTEMPTS = 5

StatusCallback = Callable[[str, str], None]


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
    """
    In-process wake-ups: streams waiting on a run's next event, and
    subscribers (workers) told about status changes. Fed by this process's own
    writes and, on PostgreSQL, by the LISTEN connection for other processes'.
    """

    def __init__(self) -> None:
        self._listeners: dict[str, set[asyncio.Event]] = {}
        self._status_subscribers: list[StatusCallback] = []

    def notify(self, run_id: str) -> None:
        for listener in self._listeners.get(str(run_id), ()):
            listener.set()

    def wake_all(self) -> None:
        """Wake every stream (after a lost LISTEN connection, events may have been missed)."""
        for listeners in self._listeners.values():
            for listener in listeners:
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

    def notify_status(self, run_id: str, status: str) -> None:
        for callback in list(self._status_subscribers):
            try:
                callback(str(run_id), status)
            except Exception:  # noqa: BLE001 - one subscriber must not break the others
                logger.exception("Status subscriber failed for run %s", run_id)

    def subscribe_status(self, callback: StatusCallback) -> Callable[[], None]:
        """Call *callback(run_id, status)* on every status change; returns an unsubscribe."""
        self._status_subscribers.append(callback)

        def unsubscribe() -> None:
            if callback in self._status_subscribers:
                self._status_subscribers.remove(callback)

        return unsubscribe


async def _pg_notify(session: AsyncSession, channel: str, payload: str) -> None:
    """NOTIFY on PostgreSQL, sent when the transaction commits; nothing elsewhere."""
    if session.get_bind().dialect.name == "postgresql":
        await session.execute(select(func.pg_notify(channel, payload)))


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
                        await _pg_notify(session, EVENTS_CHANNEL, key)
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
        """Insert a ``queued`` run and its first ``run_status`` event; wakes workers."""
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
            key = str(run.id)
            session.add(
                RunEvent(
                    run_id=run.id,
                    seq=1,
                    agent=RUNNER_AGENT,
                    type=RUN_STATUS_EVENT,
                    payload={"status": "queued", "error": None},
                )
            )
            await _pg_notify(session, STATUS_CHANNEL, f"{key}:queued")
            await session.commit()
        self.notifier.notify(key)
        self.notifier.notify_status(key, "queued")
        return key

    async def transition(
        self,
        run_id: str,
        status: str,
        *,
        error: str | None = None,
        from_statuses: Collection[str] = ACTIVE_STATUSES,
        owner: str | None = None,
        values: Mapping[str, Any] | None = None,
        where: Sequence[ColumnElement[bool]] = (),
        details: Mapping[str, Any] | None = None,
    ) -> bool:
        """
        Move the run to *status* if it is currently in *from_statuses*.

        Returns whether it moved. The update and its ``run_status`` event
        commit together; a run that is missing, or not in *from_statuses*, is
        left alone and no event is written.

        * *owner* — only if the run's ``worker_id`` is this worker.
        * *values* — further columns to set (a claim's lease, a resume's reset).
        * *where* — further conditions (a lapsed lease).
        * *details* — extra fields for the event's payload.

        ``started_at`` is set the first time a run starts; a terminal status
        sets ``finished_at`` and ends the lease.
        """
        key = str(_uuid(run_id))
        now = datetime.now(UTC)
        changes: dict[str, Any] = {"status": status}
        if status == "running":
            changes["started_at"] = func.coalesce(ResearchRun.started_at, now)
        if status in TERMINAL_STATUSES:
            changes["finished_at"] = now
            changes["lease_expires_at"] = None
        if error is not None:
            changes["error"] = error
        changes.update(values or {})

        conditions = [ResearchRun.id == _uuid(key), ResearchRun.status.in_(list(from_statuses))]
        if owner is not None:
            conditions.append(ResearchRun.worker_id == owner)
        conditions.extend(where)

        payload: dict[str, Any] = {"status": status, "error": error, **dict(details or {})}
        lock = self._lock(key)
        async with lock:
            for attempt in range(1, _MAX_WRITE_ATTEMPTS + 1):
                try:
                    async with self._session_factory() as session:
                        result = await session.execute(
                            update(ResearchRun)
                            .where(*conditions)
                            .values(**changes)
                            .execution_options(synchronize_session=False)
                        )
                        if not getattr(result, "rowcount", 0):
                            await session.rollback()
                            return False
                        await RunEventRepository(session).log_event(
                            key, RUNNER_AGENT, RUN_STATUS_EVENT, payload
                        )
                        await _pg_notify(session, EVENTS_CHANNEL, key)
                        await _pg_notify(session, STATUS_CHANNEL, f"{key}:{status}")
                        await session.commit()
                    break
                except IntegrityError:
                    if attempt == _MAX_WRITE_ATTEMPTS:
                        raise
        self.notifier.notify(key)
        self.notifier.notify_status(key, status)
        logger.info("Run %s is %s", key, status, extra={"run_id": key})
        return True
