"""
DDL helpers shared by Alembic and application startup.

Kept out of ``alembic/env.py`` so it can be unit-tested: ``env.py`` runs
migrations as an import side effect and cannot be imported from a test.
"""

from __future__ import annotations

from sqlalchemy import text
from sqlalchemy.engine import Connection

__all__ = ["CREATE_VECTOR_EXTENSION", "ensure_pgvector"]

# SQLAlchemy 2.x executes Executable objects only — a bare string raises
# ObjectNotExecutableError, which is what broke `alembic upgrade head`.
CREATE_VECTOR_EXTENSION = text("CREATE EXTENSION IF NOT EXISTS vector")


def ensure_pgvector(connection: Connection) -> bool:
    """
    Create the pgvector extension if the dialect supports it.

    Returns ``True`` when the statement was issued. Non-PostgreSQL dialects
    (SQLite in tests) are a no-op, so the same migration path works in both.
    """
    if connection.dialect.name != "postgresql":
        return False
    connection.execute(CREATE_VECTOR_EXTENSION)
    return True
