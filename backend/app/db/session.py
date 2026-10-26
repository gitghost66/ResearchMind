"""
Database session management.

Uses SQLAlchemy 2.0 async with asyncpg connecting to Supabase PostgreSQL.
Every engine — the API's, Alembic's, and scripts' — comes from
:func:`build_engine`, so connection quirks are handled in one place.
"""

import uuid
from collections.abc import AsyncGenerator
from typing import Any

from sqlalchemy.engine import make_url
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from app.core.config import get_settings

# Supabase's pooler serves transaction mode on this port. Connections are
# shared between clients per transaction, so asyncpg's prepared statements
# break unless they are uncached and uniquely named.
TRANSACTION_POOLER_PORT = 6543

_engine: AsyncEngine | None = None
_session_factory: async_sessionmaker[AsyncSession] | None = None


def engine_options(database_url: str) -> dict[str, Any]:
    """Extra ``create_async_engine`` arguments needed by *database_url*."""
    url = make_url(database_url)
    if url.get_backend_name() != "postgresql" or url.port != TRANSACTION_POOLER_PORT:
        return {}
    return {
        "connect_args": {
            "statement_cache_size": 0,
            "prepared_statement_cache_size": 0,
            "prepared_statement_name_func": lambda: f"__asyncpg_{uuid.uuid4()}__",
        }
    }


def build_engine(database_url: str, **kwargs: Any) -> AsyncEngine:
    """Create an async engine with the options *database_url* needs."""
    options = engine_options(database_url)
    options.update(kwargs)
    return create_async_engine(database_url, **options)


def get_engine() -> AsyncEngine:
    """Get or create the async engine (lazy singleton)."""
    global _engine
    if _engine is None:
        settings = get_settings()
        if not settings.database_url:
            raise RuntimeError(
                "DATABASE_URL is not configured. "
                "Set it in backend/.env to your Supabase connection string."
            )
        _engine = build_engine(
            settings.database_url,
            echo=settings.debug,
            pool_size=5,
            max_overflow=10,
            pool_pre_ping=True,
        )
    return _engine


def get_session_factory() -> async_sessionmaker[AsyncSession]:
    """Get or create the session factory (lazy singleton)."""
    global _session_factory
    if _session_factory is None:
        _session_factory = async_sessionmaker(
            bind=get_engine(),
            class_=AsyncSession,
            expire_on_commit=False,
        )
    return _session_factory


async def get_db_session() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI dependency that yields a DB session."""
    factory = get_session_factory()
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
