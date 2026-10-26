"""Shared test fixtures for the backend."""

from collections.abc import AsyncGenerator

import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy import event
from sqlalchemy.engine import Engine
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from app.core.config import Settings, get_settings
from app.db.session import get_db_session
from app.main import app
from app.models.base import Base

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
