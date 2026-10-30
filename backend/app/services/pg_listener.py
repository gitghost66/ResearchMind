"""
PostgreSQL LISTEN for run events and status changes (M8 multi-worker).

Every write by :class:`~app.services.run_events.RunEventRecorder` sends a
``NOTIFY`` (see that module). One :class:`PgListener` per process holds a
dedicated connection that listens on both channels and replays each
notification into the process's :class:`EventNotifier`:

* ``rm_run_events`` wakes the SSE streams following that run, whichever
  process wrote the event;
* ``rm_run_status`` tells workers a run was queued (claim it) or settled
  elsewhere (stop working on it).

The connection is reopened with back-off if it drops, and every stream is
woken then, since notifications sent while it was down are lost. Correctness
never depends on it: streams poll (``SSE_POLL_INTERVAL_SECONDS``), workers poll
(``WORKER_POLL_SECONDS``) and renew leases.

LISTEN needs a session that stays on one server connection, which Supabase's
transaction pooler (port 6543) cannot give. Point ``DATABASE_LISTEN_URL`` at
the session pooler or a direct connection; without it, a transaction-pooler
URL is not listened on and processes rely on polling.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Awaitable, Callable
from typing import Any

from sqlalchemy.engine import make_url

from app.db.session import TRANSACTION_POOLER_PORT
from app.services.run_events import EVENTS_CHANNEL, STATUS_CHANNEL, EventNotifier

logger = logging.getLogger(__name__)

__all__ = ["PgListener", "asyncpg_dsn", "listen_url_for"]

Connect = Callable[[str], Awaitable[Any]]


def asyncpg_dsn(database_url: str) -> str:
    """A SQLAlchemy URL (``postgresql+asyncpg://...``) as a plain libpq DSN for asyncpg."""
    url = make_url(database_url).set(drivername="postgresql")
    return url.render_as_string(hide_password=False)


def listen_url_for(database_url: str, listen_url: str = "") -> str | None:
    """
    The URL to LISTEN on, or ``None`` when there is none that works: no
    PostgreSQL, or only a transaction-pooler URL.
    """
    if listen_url:
        return listen_url
    if not database_url:
        return None
    url = make_url(database_url)
    if url.get_backend_name() != "postgresql":
        return None
    if url.port == TRANSACTION_POOLER_PORT:
        logger.warning(
            "DATABASE_URL uses Supabase's transaction pooler, which cannot LISTEN; "
            "set DATABASE_LISTEN_URL to the session pooler (port 5432) or a direct "
            "connection for live events across processes. Falling back to polling."
        )
        return None
    return database_url


def _resolver(future: asyncio.Future[None]) -> Callable[[Any], None]:
    """A termination callback that resolves *future* (once)."""

    def resolve(_connection: Any) -> None:
        if not future.done():
            future.set_result(None)

    return resolve


async def _asyncpg_connect(dsn: str) -> Any:
    import asyncpg

    return await asyncpg.connect(dsn)


class PgListener:
    """Feeds NOTIFY messages from PostgreSQL into an :class:`EventNotifier`."""

    def __init__(
        self,
        database_url: str,
        notifier: EventNotifier,
        *,
        connect: Connect | None = None,
        initial_delay: float = 1.0,
        max_delay: float = 30.0,
    ) -> None:
        self._dsn = asyncpg_dsn(database_url)
        self._notifier = notifier
        self._connect = connect or _asyncpg_connect
        self._initial_delay = initial_delay
        self._max_delay = max_delay
        self._task: asyncio.Task[None] | None = None
        # Set while a connection is listening (tests and /ready).
        self.listening = asyncio.Event()

    async def start(self) -> None:
        if self._task is None:
            self._task = asyncio.create_task(self._run(), name="pg-listener")

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._task
            self._task = None

    # -- callbacks (asyncpg calls them on the event loop) ----------------

    def _on_event(self, _connection: Any, _pid: int, _channel: str, payload: str) -> None:
        self._notifier.notify(payload)

    def _on_status(self, _connection: Any, _pid: int, _channel: str, payload: str) -> None:
        run_id, _, status = payload.partition(":")
        if run_id and status:
            self._notifier.notify_status(run_id, status)

    # -- connection loop -------------------------------------------------

    async def _run(self) -> None:
        delay = self._initial_delay
        while True:
            connection: Any = None
            try:
                connection = await self._connect(self._dsn)
                closed: asyncio.Future[None] = asyncio.get_running_loop().create_future()
                connection.add_termination_listener(_resolver(closed))
                await connection.add_listener(EVENTS_CHANNEL, self._on_event)
                await connection.add_listener(STATUS_CHANNEL, self._on_status)
                self.listening.set()
                delay = self._initial_delay
                logger.info("Listening for run notifications")
                # Anything sent while there was no connection is lost.
                self._notifier.wake_all()
                await closed
                logger.warning("LISTEN connection closed; reconnecting")
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # noqa: BLE001 - retry with back-off
                logger.warning("LISTEN connection failed: %s; retrying in %.0fs", exc, delay)
            finally:
                self.listening.clear()
                if connection is not None:
                    with contextlib.suppress(Exception):
                        await connection.close()
            await asyncio.sleep(delay)
            delay = min(delay * 2, self._max_delay)
