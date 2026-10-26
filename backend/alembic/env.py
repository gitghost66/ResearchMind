import asyncio
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.engine import Connection

from alembic import context
from app.core.config import get_settings
from app.db.ddl import ensure_pgvector
from app.db.session import build_engine
from app.models import Base

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config

# Interpret the config file for Python logging.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Set target metadata from our models
target_metadata = Base.metadata

# Override sqlalchemy.url from Settings if DATABASE_URL is set in environment
settings = get_settings()
if settings.database_url:
    # Alembic's config uses ConfigParser interpolation, so a percent-encoded
    # password ("%40" for "@") must be escaped as "%%".
    config.set_main_option("sqlalchemy.url", settings.database_url.replace("%", "%%"))


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection: Connection) -> None:
    context.configure(
        connection=connection,
        target_metadata=target_metadata,
    )

    with context.begin_transaction():
        # Ensure pgvector exists before any table with a Vector column is created.
        ensure_pgvector(connection)
        context.run_migrations()


async def run_async_migrations() -> None:
    """Run migrations in 'online' mode with an async engine."""
    url = config.get_main_option("sqlalchemy.url")
    if not url:
        raise RuntimeError("No database URL: set DATABASE_URL in backend/.env")
    # build_engine adds the options Supabase's transaction pooler needs.
    connectable = build_engine(url, poolclass=pool.NullPool)

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    asyncio.run(run_async_migrations())


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
