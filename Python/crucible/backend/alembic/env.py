"""
Alembic environment configuration for Crucible.

This file wires Alembic to:
  - Our SQLAlchemy metadata (all model tables)
  - The DATABASE_URL from settings (same connection as the app)
  - Async engine via run_sync() so migrations work with aiosqlite / asyncpg

HOW TO USE
----------
Create a migration after adding or changing an ORM model:
  alembic revision --autogenerate -m "add_prediction_log_table"

Apply pending migrations:
  alembic upgrade head

Rollback one migration:
  alembic downgrade -1

Check current state:
  alembic current

FIRST-RUN NOTE
--------------
If the database already has tables from create_all() (dev environment),
run `alembic stamp head` once to mark the current schema as the baseline
without re-running the initial migration.
"""

import asyncio
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.engine import Connection
from sqlalchemy.ext.asyncio import create_async_engine

from alembic import context

# ── Import all models so their tables are in Base.metadata ───────────────────
# Every model imported here will be included in --autogenerate comparisons.
# If you add a new model file, import it here.
from models import Base          # noqa: F401 — registers all table metadata
from models import (             # noqa: F401
    Dataset, Connector, Experiment, RAGDocument, User
)

# ── Alembic config ─────────────────────────────────────────────────────────
config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def get_url() -> str:
    """Read DATABASE_URL from application settings (same source as the app)."""
    from config import settings
    return settings.database_url


# ── Offline migration (generates SQL without a live connection) ───────────────

def run_migrations_offline() -> None:
    """
    Runs migrations in 'offline' mode — generates SQL without connecting.
    Useful for reviewing what will be executed before applying.
    """
    url = get_url()
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


# ── Online migration (runs against a live database) ───────────────────────────

def do_run_migrations(connection: Connection) -> None:
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """
    Creates an async engine and runs migrations through a sync wrapper.
    Alembic's run_sync() bridges the async engine to Alembic's sync API.
    """
    connectable = create_async_engine(get_url(), poolclass=pool.NullPool)
    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)
    await connectable.dispose()


def run_migrations_online() -> None:
    connectable = config.attributes.get("connection", None)
    if connectable is None:
        asyncio.run(run_async_migrations())
    else:
        do_run_migrations(connectable)

# ── Entry point ───────────────────────────────────────────────────────────────

if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
