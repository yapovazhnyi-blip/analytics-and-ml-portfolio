"""
Standalone database migration runner.

Run this BEFORE starting the app:  python migrate.py

Runs as a separate process before uvicorn, so there is no running event loop
and Alembic's asyncio.run() works without conflict. Fails loudly on error
(exits 1) so the docker-compose command "python migrate.py && uvicorn ..."
stops before starting a server against a broken schema.
"""

from __future__ import annotations

import sys
from pathlib import Path

from sqlalchemy import create_engine, inspect as sa_inspect, text
from alembic.config import Config
from alembic import command

from config import settings


def _sync_url(database_url: str) -> str:
    """Rewrite async driver prefix to its synchronous equivalent.

    Only the scheme changes — slashes, host, and path are preserved exactly.
    sqlite+aiosqlite:////app/data/crucible.db → sqlite:////app/data/crucible.db
    postgresql+asyncpg://u:p@h/db            → postgresql://u:p@h/db
    """
    return (
        database_url
        .replace("sqlite+aiosqlite:", "sqlite:", 1)
        .replace("postgresql+asyncpg:", "postgresql:", 1)
    )


def _current_revision(engine) -> tuple[str | None, set[str]]:
    """Return (current_alembic_revision, set_of_existing_table_names).

    Uses a dedicated short-lived connection that is closed before any
    migration work begins, so there is no connection conflict with the
    upgrade/stamp step.
    """
    with engine.connect() as conn:
        inspector = sa_inspect(conn)
        tables = set(inspector.get_table_names())
        if "alembic_version" not in tables:
            return None, tables
        row = conn.execute(text("SELECT version_num FROM alembic_version")).fetchone()
        return (row[0] if row else None), tables


def run_migrations() -> None:
    alembic_ini = Path(__file__).parent / "alembic.ini"
    if not alembic_ini.exists():
        raise FileNotFoundError(f"alembic.ini not found at {alembic_ini}")

    sync_url = _sync_url(settings.database_url)
    cfg = Config(str(alembic_ini))
    cfg.set_main_option("sqlalchemy.url", sync_url)

    engine = create_engine(sync_url)
    try:
        current_rev, tables = _current_revision(engine)

        if current_rev is not None:
            # Database has an Alembic revision — apply any pending migrations.
            print(f"crucible.migrations_current: at revision {current_rev}, upgrading to head")
            with engine.connect() as conn:
                cfg.attributes["connection"] = conn
                command.upgrade(cfg, "head")
                conn.commit()

        elif tables - {"alembic_version"}:
            # Tables exist but no Alembic tracking — created by create_all()
            # fallback. Stamp at head so future migrations work correctly.
            print("crucible.migrations_stamp: tables exist without Alembic tracking, stamping at head")
            with engine.connect() as conn:
                cfg.attributes["connection"] = conn
                command.stamp(cfg, "head")
                conn.commit()

        else:
            # Genuinely fresh database — run all migrations from scratch.
            print("crucible.migrations_fresh: empty database, running all migrations")
            with engine.connect() as conn:
                cfg.attributes["connection"] = conn
                command.upgrade(cfg, "head")
                conn.commit()

    finally:
        engine.dispose()


if __name__ == "__main__":
    try:
        run_migrations()
        print("crucible.migrations_applied: database is at head revision")
    except Exception as exc:
        print(
            f"crucible.migrations_failed: {type(exc).__name__}: {exc}",
            file=sys.stderr,
        )
        raise SystemExit(1)
