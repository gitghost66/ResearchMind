"""
Tests for database URL handling: Supabase-style URLs and the transaction pooler.
"""

import pytest

from app.core.config import Settings
from app.db.session import TRANSACTION_POOLER_PORT, build_engine, engine_options


class TestDatabaseUrlDriver:
    @pytest.mark.parametrize(
        ("given", "expected"),
        [
            ("postgresql://u:p@h:5432/db", "postgresql+asyncpg://u:p@h:5432/db"),
            ("postgres://u:p@h/db", "postgresql+asyncpg://u:p@h/db"),
            ("  postgresql://u:p@h/db  ", "postgresql+asyncpg://u:p@h/db"),
            ("postgresql+asyncpg://u:p@h/db", "postgresql+asyncpg://u:p@h/db"),
            ("sqlite+aiosqlite:///:memory:", "sqlite+aiosqlite:///:memory:"),
            ("", ""),
        ],
    )
    def test_plain_postgres_urls_get_the_async_driver(self, given: str, expected: str) -> None:
        assert Settings(database_url=given).database_url == expected


class TestEngineOptions:
    def test_transaction_pooler_disables_prepared_statement_caching(self) -> None:
        url = f"postgresql+asyncpg://u:p@pooler.example:{TRANSACTION_POOLER_PORT}/postgres"
        args = engine_options(url)["connect_args"]

        assert args["statement_cache_size"] == 0
        assert args["prepared_statement_cache_size"] == 0
        first, second = args["prepared_statement_name_func"](), args["prepared_statement_name_func"]()
        assert first != second, "statement names must be unique across pooled connections"

    @pytest.mark.parametrize(
        "url",
        [
            "postgresql+asyncpg://u:p@db.example:5432/postgres",
            "postgresql+asyncpg://u:p@db.example/postgres",
            "sqlite+aiosqlite:///:memory:",
        ],
    )
    def test_other_urls_need_no_extra_options(self, url: str) -> None:
        assert engine_options(url) == {}

    def test_percent_encoded_password_survives(self) -> None:
        engine = build_engine("postgresql+asyncpg://user:p%40ss%2Fw%25rd@db.example:5432/postgres")
        try:
            assert engine.url.password == "p@ss/w%rd"
        finally:
            engine.sync_engine.dispose()
