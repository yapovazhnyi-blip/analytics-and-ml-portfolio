"""
Database layer for Crucible.

Promotes the async SQLAlchemy patterns validated in the spike:
  - make_async_engine: pool config differs for SQLite vs Postgres
  - get_session: one AsyncSession per request, auto-commit/rollback
  - get_db: FastAPI dependency that yields a session per request

The sync engine mirror is also created here for the pandas bridge
(pd.read_sql cannot use the async engine — validated in spike).
"""

from contextlib import asynccontextmanager
from typing import AsyncIterator

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from config import settings

# ── Engine factory ─────────────────────────────────────────────────────────

def _make_engine(url: str) -> AsyncEngine:
    if url.startswith("sqlite"):
        from sqlalchemy.pool import StaticPool
        return create_async_engine(
            url,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
            echo=settings.debug,
        )
    # Postgres / other
    return create_async_engine(
        url,
        pool_size=5,
        max_overflow=10,
        pool_pre_ping=True,      # validates connections on checkout
        pool_recycle=1800,       # recycle after 30 min to avoid stale connections
        echo=settings.debug,
    )


# ── Module-level singletons ────────────────────────────────────────────────
# Created once at import time; replaced during testing via override.

engine: AsyncEngine = _make_engine(settings.database_url)

SessionFactory: async_sessionmaker[AsyncSession] = async_sessionmaker(
    engine,
    expire_on_commit=False,  # objects readable after commit without lazy-load
)

# Alias used by background tasks that need their own session outside a request lifecycle
AsyncSessionLocal = SessionFactory

# Sync URL mirror for pandas bridge (strip async driver prefix)
# "sqlite+aiosqlite:///./data.db" → "sqlite:///./data.db"
# "postgresql+asyncpg://..."      → "postgresql://..."
sync_database_url: str = (
    settings.database_url
    .replace("+aiosqlite", "")
    .replace("+asyncpg", "")
)


# ── Session context manager ────────────────────────────────────────────────

@asynccontextmanager
async def get_session() -> AsyncIterator[AsyncSession]:
    """
    One AsyncSession per logical operation.

    Usage (service layer):
        async with get_session() as session:
            result = await session.execute(select(Dataset))

    Auto-commits on clean exit, rolls back on any exception.
    Never share a session across requests — validated in spike.
    """
    async with SessionFactory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


# ── FastAPI dependency ─────────────────────────────────────────────────────

async def get_db() -> AsyncIterator[AsyncSession]:
    """
    FastAPI dependency — yields one session per HTTP request.

    Usage in router:
        @router.get("/datasets")
        async def list_datasets(db: AsyncSession = Depends(get_db)):
            ...

    The session commits on clean response, rolls back on exception.
    expire_on_commit=False means ORM objects remain readable after
    the session closes (important for returning Pydantic-serialised
    response bodies).
    """
    async with SessionFactory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise


# ── Schema initialisation ──────────────────────────────────────────────────

async def init_db() -> None:
    """
    Ensure the database schema is ready for the running application.

    migrate.py (run as a separate process before uvicorn) is the primary
    owner of schema migrations. This function runs AFTER migrate.py has
    already applied all pending migrations.

    It always calls create_all() as a safety net: for tables that exist
    it is a no-op; for any tables that are genuinely missing (e.g. after
    a stamp-without-tables recovery) it creates them from the current
    models. This makes startup robust against any partially-migrated state
    without replacing the real migration machinery.
    """
    from models import Base  # noqa: F401 — populates Base.metadata

    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
