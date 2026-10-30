"""Shared test fixtures for the backend."""

import asyncio
import dataclasses
from collections.abc import AsyncGenerator, Awaitable, Callable
from typing import Any

import pytest_asyncio
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from researchmind_agents.config import LLMConfig
from researchmind_agents.llm.base import UsageSink
from researchmind_agents.testing import (
    FakePageExtract,
    FakeSearchProvider,
    FakeWeb,
    ScriptedLLM,
    fake_embedding_service,
    fake_scholarly_tool,
)
from researchmind_agents.tools.search import SearchTool
from sqlalchemy import event
from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.container import AppServices, build_services
from app.core.config import Settings, get_settings
from app.db.session import get_db_session
from app.main import app, create_app
from app.models.base import Base
from app.services.agent_runtime import build_agent_config
from app.services.run_events import TERMINAL_STATUSES
from app.workers.research_worker import RunWorker

TEST_DB_URL = "sqlite+aiosqlite:///:memory:"

# The suite must behave the same on every machine and in CI, so it never reads
# a developer's real .env (API keys, DATABASE_URL). Tests that need a value
# pass it to Settings(...) explicitly.
Settings.model_config["env_file"] = None
get_settings.cache_clear()


# Enable foreign key support for SQLite connections
@event.listens_for(Engine, "connect")
def _set_sqlite_pragma(dbapi_connection, connection_record):  # type: ignore[no-untyped-def]
    try:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()
    except Exception:
        pass


@pytest_asyncio.fixture(scope="function")
async def async_engine():
    """Create a fresh in-memory SQLite database for each test function."""
    engine = create_async_engine(
        TEST_DB_URL,
        echo=False,
    )
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    yield engine

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)
    await engine.dispose()


@pytest_asyncio.fixture(scope="function")
async def db_session(async_engine) -> AsyncGenerator[AsyncSession, None]:
    """Yield a database session bound to the in-memory test database."""
    session_factory = async_sessionmaker(
        bind=async_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    async with session_factory() as session:
        yield session


@pytest_asyncio.fixture(scope="function")
async def client(db_session: AsyncSession) -> AsyncGenerator[AsyncClient, None]:
    """Async HTTP client with overridden DB session dependency."""

    async def _override_get_db():
        yield db_session

    app.dependency_overrides[get_db_session] = _override_get_db

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac

    app.dependency_overrides.clear()


# ---------------------------------------------------------------------------
# API harness (M4, M8): the real app, services, and in-process worker over a
# SQLite file, with the agents' fakes standing in for the LLM, search, pages,
# and embeddings.
# ---------------------------------------------------------------------------

API_QUESTION = "RAG versus fine-tuning"


def _page_text(seed: str) -> str:
    return " ".join(
        f"The {seed} study sentence {k} compares rag benefits with fine-tuning costs and risks."
        for k in range(10)
    )


def make_fake_web() -> FakeWeb:
    return FakeWeb(
        pages={
            "https://a.example/rag": ("RAG benefits", _page_text("alpha")),
            "https://b.example/ft": ("Fine-tuning costs", _page_text("beta")),
            "https://c.example/compare": ("RAG vs fine-tuning risks", _page_text("gamma")),
        }
    )


class GatedLLM(ScriptedLLM):
    """
    Model calls wait for ``gate``; ``waiting`` is set when one starts. With
    schema names in ``hold``, only those calls wait (e.g. ``ReportOutput``
    holds a quick run at its synthesizer, after the planner and research).

    Clearing the gate before a run starts holds it at the planner's call: one
    task, no query open, so cancelling it there is clean. (Cancelling a run
    while a branch is mid-query can leave aiosqlite holding a write lock until
    GC; asyncpg closes the socket instead, so Postgres is unaffected.)
    """

    def __init__(
        self,
        *args: Any,
        gate: asyncio.Event,
        waiting: asyncio.Event,
        hold: set[str] | None = None,
        **kwargs: Any,
    ):
        super().__init__(*args, **kwargs)
        self._gate = gate
        self._waiting = waiting
        self._hold = hold if hold is not None else set()

    async def _complete(self, **kwargs: Any) -> Any:
        schema = kwargs.get("json_schema")
        name = schema.__name__ if schema is not None else "text"
        if not self._hold or name in self._hold:
            self._waiting.set()
            await self._gate.wait()
        return await super()._complete(**kwargs)


def make_test_settings(**overrides: Any) -> Settings:
    values: dict[str, Any] = {
        "gemini_api_key": "test-key",
        "database_url": "",
        "api_key": "",
        "sse_poll_interval_seconds": 0.2,
        "worker_poll_seconds": 0.1,
        "cancel_wait_seconds": 5.0,
        "shutdown_grace_seconds": 5.0,
    }
    values.update(overrides)
    return Settings(_env_file=None, **values)  # type: ignore[call-arg]


@dataclasses.dataclass
class ApiHarness:
    app: FastAPI
    client: AsyncClient
    services: AppServices
    settings: Settings
    engine: AsyncEngine
    web: FakeWeb
    # Clear to hold the next run at its first model call (the planner's), or
    # at the first call whose schema is in ``hold``.
    gate: asyncio.Event
    waiting: asyncio.Event
    hold: set[str]
    runtime_kwargs: Callable[[Settings], dict[str, Any]]

    @property
    def session_factory(self) -> async_sessionmaker[AsyncSession]:
        return self.services.session_factory

    @property
    def worker(self) -> RunWorker:
        assert self.services.worker is not None, "this harness runs no worker"
        return self.services.worker

    async def create(self, **body: Any) -> str:
        body.setdefault("question", API_QUESTION)
        response = await self.client.post("/api/v1/research", json=body)
        assert response.status_code == 202, response.text
        return str(response.json()["run_id"])

    async def wait_for(self, run_id: str, *statuses: str, timeout: float = 15.0) -> dict[str, Any]:
        wanted = set(statuses) or set(TERMINAL_STATUSES)
        async with asyncio.timeout(timeout):
            while True:
                response = await self.client.get(f"/api/v1/research/{run_id}")
                assert response.status_code == 200, response.text
                data: dict[str, Any] = response.json()
                if data["status"] in wanted:
                    return data
                await asyncio.sleep(0.02)

    async def until(self, predicate: Callable[[], bool], timeout: float = 10.0) -> None:
        async with asyncio.timeout(timeout):
            while not predicate():
                await asyncio.sleep(0.01)


def _sqlite_wal(dbapi_connection, connection_record):  # type: ignore[no-untyped-def]
    """WAL: readers never wait for writers, as on Postgres."""
    cursor = dbapi_connection.cursor()
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.close()


def make_runtime_kwargs(
    web: FakeWeb, gate: asyncio.Event, waiting: asyncio.Event, hold: set[str] | None = None
) -> Callable[[Settings], dict[str, Any]]:
    """Fakes for every external service; model calls wait for *gate*."""

    def runtime_kwargs(run_settings: Settings) -> dict[str, Any]:
        config = build_agent_config(run_settings)

        def llm_factory(llm_config: LLMConfig, sink: UsageSink | None) -> ScriptedLLM:
            return GatedLLM(
                dataclasses.replace(llm_config, base_delay=0.0, overload_delay=0.0),
                sink,
                gate=gate,
                waiting=waiting,
                hold=hold,
            )

        return {
            "llm_factory": llm_factory,
            "embeddings": fake_embedding_service(config.embedding),
            "search": SearchTool(config.search, provider=FakeSearchProvider(web)),
            "page_extract": FakePageExtract(web),
            # Never reach arXiv or Semantic Scholar from tests.
            "scholarly": fake_scholarly_tool(config),
        }

    return runtime_kwargs


async def _build_harness(tmp_path: Any, index: int, **overrides: Any) -> ApiHarness:
    settings = make_test_settings(**overrides)
    engine = create_async_engine(f"sqlite+aiosqlite:///{tmp_path / f'api-{index}.db'}")
    event.listen(engine.sync_engine, "connect", _sqlite_wal)
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
    factory = async_sessionmaker(bind=engine, class_=AsyncSession, expire_on_commit=False)

    web = make_fake_web()
    gate = asyncio.Event()
    gate.set()
    waiting = asyncio.Event()
    hold: set[str] = set()
    runtime_kwargs = make_runtime_kwargs(web, gate, waiting, hold)

    services = build_services(settings, factory, runtime_kwargs=runtime_kwargs)
    # ASGITransport sends no lifespan events, so start what the lifespan would.
    await services.startup()
    app = create_app(settings=settings, services=services)
    client = AsyncClient(transport=ASGITransport(app=app), base_url="http://test")
    return ApiHarness(
        app=app,
        client=client,
        services=services,
        settings=settings,
        engine=engine,
        web=web,
        gate=gate,
        waiting=waiting,
        hold=hold,
        runtime_kwargs=runtime_kwargs,
    )


@pytest_asyncio.fixture
async def make_api(tmp_path: Any) -> AsyncGenerator[Callable[..., Awaitable[ApiHarness]], None]:
    """Factory: ``await make_api(**settings_overrides)`` -> a ready :class:`ApiHarness`."""
    built: list[ApiHarness] = []

    async def make(**overrides: Any) -> ApiHarness:
        harness = await _build_harness(tmp_path, len(built), **overrides)
        built.append(harness)
        return harness

    yield make

    for harness in built:
        # Shut down before releasing held runs (see GatedLLM). Tests leave
        # their runs finished or held at the gate.
        await harness.services.shutdown()
        harness.gate.set()
        await harness.client.aclose()
        await harness.engine.dispose()


@pytest_asyncio.fixture
async def api(make_api: Callable[..., Awaitable[ApiHarness]]) -> ApiHarness:
    return await make_api()
