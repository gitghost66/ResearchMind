"""
M8: PostgreSQL LISTEN -> in-process notifier, with a fake asyncpg connection.

SQLite cannot LISTEN, so the bridge is tested against a stand-in that
behaves like asyncpg's connection: listeners registered by channel, a
termination callback, and close().
"""

from __future__ import annotations

import asyncio
from typing import Any

from app.services.pg_listener import PgListener, asyncpg_dsn, listen_url_for
from app.services.run_events import EVENTS_CHANNEL, STATUS_CHANNEL, EventNotifier

PG_URL = "postgresql+asyncpg://user:p%40ss@db.example:5432/research"


class FakeConnection:
    def __init__(self) -> None:
        self.listeners: dict[str, Any] = {}
        self.on_terminate: list[Any] = []
        self.closed = False

    async def add_listener(self, channel: str, callback: Any) -> None:
        self.listeners[channel] = callback

    def add_termination_listener(self, callback: Any) -> None:
        self.on_terminate.append(callback)

    async def close(self) -> None:
        self.closed = True

    def send(self, channel: str, payload: str) -> None:
        self.listeners[channel](self, 1234, channel, payload)

    def drop(self) -> None:
        for callback in self.on_terminate:
            callback(self)


class FakeServer:
    def __init__(self, failures: int = 0) -> None:
        self.connections: list[FakeConnection] = []
        self.dsns: list[str] = []
        self.failures = failures

    async def connect(self, dsn: str) -> FakeConnection:
        self.dsns.append(dsn)
        if self.failures:
            self.failures -= 1
            raise OSError("connection refused")
        connection = FakeConnection()
        self.connections.append(connection)
        return connection


async def started(server: FakeServer, notifier: EventNotifier) -> PgListener:
    listener = PgListener(PG_URL, notifier, connect=server.connect, initial_delay=0.01)
    await listener.start()
    await asyncio.wait_for(listener.listening.wait(), 5)
    return listener


class TestBridge:
    async def test_notifications_reach_streams_and_workers(self) -> None:
        notifier, server = EventNotifier(), FakeServer()
        statuses: list[tuple[str, str]] = []
        notifier.subscribe_status(lambda run_id, status: statuses.append((run_id, status)))
        listener = await started(server, notifier)
        try:
            with notifier.listen("run-1") as wakeup:
                server.connections[0].send(EVENTS_CHANNEL, "run-1")
                assert wakeup.is_set()
            server.connections[0].send(STATUS_CHANNEL, "run-2:cancelled")
            server.connections[0].send(STATUS_CHANNEL, "garbage")
            assert statuses == [("run-2", "cancelled")]
            assert server.dsns == ["postgresql://user:p%40ss@db.example:5432/research"]
        finally:
            await listener.stop()
        assert server.connections[0].closed

    async def test_a_dropped_connection_is_reopened_and_streams_are_woken(self) -> None:
        notifier, server = EventNotifier(), FakeServer()
        listener = await started(server, notifier)
        try:
            with notifier.listen("run-1") as wakeup:
                wakeup.clear()
                server.connections[0].drop()
                async with asyncio.timeout(5):
                    while len(server.connections) < 2 or not listener.listening.is_set():
                        await asyncio.sleep(0.01)
                # Notifications sent while it was down are lost: wake everyone.
                assert wakeup.is_set()
        finally:
            await listener.stop()

    async def test_connection_failures_are_retried(self) -> None:
        notifier, server = EventNotifier(), FakeServer(failures=2)
        listener = await started(server, notifier)
        await listener.stop()
        assert len(server.dsns) == 3


class TestListenUrl:
    def test_postgres_url_is_listened_on(self) -> None:
        assert listen_url_for(PG_URL) == PG_URL

    def test_an_explicit_listen_url_wins(self) -> None:
        assert listen_url_for(PG_URL, "postgresql+asyncpg://x@direct:5432/db") == (
            "postgresql+asyncpg://x@direct:5432/db"
        )

    def test_the_transaction_pooler_cannot_listen(self) -> None:
        pooler = "postgresql+asyncpg://u:p@aws-0-x.pooler.supabase.com:6543/postgres"
        assert listen_url_for(pooler) is None
        assert listen_url_for(pooler, "postgresql+asyncpg://u:p@host:5432/postgres") is not None

    def test_no_database_or_sqlite(self) -> None:
        assert listen_url_for("") is None
        assert listen_url_for("sqlite+aiosqlite:///x.db") is None

    def test_dsn_drops_the_sqlalchemy_driver(self) -> None:
        assert asyncpg_dsn(PG_URL).startswith("postgresql://user:p%40ss@")
