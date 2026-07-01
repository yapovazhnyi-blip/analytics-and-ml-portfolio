"""
SQL connector for Crucible — Postgres and SQLite.

Promotes the four patterns validated in the async SQLAlchemy spike:
  1. Session scoping: one AsyncSession per operation
  2. Pandas bridge: run_in_executor so pd.read_sql doesn't block the event loop
  3. Server-side cursor streaming: yield chunks without loading full result
  4. Schema inference: reflect column types from DB metadata

The connector saves query results as Parquet so the profiling layer
can read them the same way as uploaded files — no special-casing needed.
"""

from __future__ import annotations

import asyncio
import os
import uuid
from pathlib import Path
from typing import Optional, AsyncIterator

import pandas as pd
import sqlalchemy as sa
from sqlalchemy.ext.asyncio import AsyncEngine, create_async_engine

from connectors.base import (
    BaseConnector,
    ColumnInfo,
    IngestResult,
    infer_columns,
    sha256_of_bytes,
)

# Max rows to stream via server-side cursor per chunk
DEFAULT_CHUNK_SIZE = 5_000


class SQLConnector(BaseConnector):
    """
    Connects to a SQL database and materialises query results as Parquet.

    Supports:
      - PostgreSQL  (db_url prefix: "postgresql+asyncpg://...")
      - SQLite      (db_url prefix: "sqlite+aiosqlite://...")

    Usage:
        connector = SQLConnector(db_url="sqlite+aiosqlite:///./data.db",
                                 storage_dir="/data/datasets")
        result = await connector.ingest(query="SELECT * FROM sales LIMIT 10000")
    """

    def __init__(self, db_url: str, storage_dir: str):
        self.db_url = db_url
        self.sync_url = _to_sync_url(db_url)
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self._engine: Optional[AsyncEngine] = None

    def _get_engine(self) -> AsyncEngine:
        if self._engine is None:
            self._engine = _make_engine(self.db_url)
        return self._engine

    async def close(self) -> None:
        if self._engine:
            await self._engine.dispose()
            self._engine = None

    # ── Ingest: run query, save as Parquet ────────────────────────────────

    async def ingest(self, query: str, **kwargs) -> IngestResult:
        """
        Runs a SQL query and saves the result as a Parquet file.

        Uses the pandas bridge (run_in_executor) so pd.read_sql doesn't
        block the FastAPI event loop — validated in spike.
        """
        try:
            df = await self._read_sql_async(query)
        except Exception as exc:
            return IngestResult(source_type="sql", error=f"Query failed: {exc}")

        if df.empty:
            return IngestResult(source_type="sql", error="Query returned no rows")

        # Save as Parquet — downstream profiling reads Parquet uniformly
        parquet_name = f"{uuid.uuid4().hex}.parquet"
        file_path = self.storage_dir / parquet_name
        try:
            df.to_parquet(file_path, index=False)
        except Exception as exc:
            return IngestResult(source_type="sql", error=f"Failed to save result: {exc}")

        content = file_path.read_bytes()
        return IngestResult(
            source_type="sql",
            file_path=str(file_path),
            content_hash=sha256_of_bytes(content),
            row_count=len(df),
            column_count=len(df.columns),
            columns=infer_columns(df),
        )

    # ── Streaming chunks ──────────────────────────────────────────────────

    async def stream_chunks(
        self,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        query: str = "",
        **kwargs,
    ) -> AsyncIterator[pd.DataFrame]:
        """
        Yields query results in chunks via server-side cursor.

        Validated in spike: stream_results=True + partitions(n) is the
        correct pattern. Without stream_results, SQLAlchemy buffers the
        full result set even when iterating — defeating the purpose.
        """
        engine = self._get_engine()
        stmt = sa.text(query)

        async with engine.connect() as conn:
            await conn.execution_options(stream_results=True)
            result = await conn.stream(stmt)
            columns = list(result.keys())

            async for partition in result.partitions(chunk_size):
                rows = [dict(zip(columns, row)) for row in partition]
                yield pd.DataFrame(rows, columns=columns)

    # ── Schema inference ──────────────────────────────────────────────────

    async def get_table_schema(self, table_name: str) -> list[ColumnInfo]:
        """
        Reflects table schema without loading any data rows.
        Used to surface schema to the UI before running a full query.
        """
        engine = self._get_engine()
        async with engine.connect() as conn:
            table = await conn.run_sync(
                lambda sync_conn: sa.Table(
                    table_name,
                    sa.MetaData(),
                    autoload_with=sync_conn,
                )
            )
        return [
            ColumnInfo(
                name=col.name,
                dtype=str(col.type),
                nullable=bool(col.nullable),
            )
            for col in table.columns
        ]

    async def list_tables(self) -> list[str]:
        """Lists all accessible tables in the database."""
        engine = self._get_engine()
        async with engine.connect() as conn:
            tables = await conn.run_sync(
                lambda sync_conn: sa.inspect(sync_conn).get_table_names()
            )
        return tables

    async def test_connection(self) -> tuple[bool, Optional[str]]:
        """
        Returns (success, error_message).
        Runs a lightweight SELECT 1 to verify the DB is reachable.
        """
        try:
            engine = self._get_engine()
            async with engine.connect() as conn:
                await conn.execute(sa.text("SELECT 1"))
            return True, None
        except Exception as exc:
            return False, str(exc)

    # ── Private: pandas bridge ────────────────────────────────────────────

    async def _read_sql_async(self, query: str) -> pd.DataFrame:
        """
        Runs pd.read_sql in a thread executor.
        pd.read_sql is synchronous — calling it directly would block the
        event loop for the duration of the query. Validated in spike.
        """
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            _sync_read_sql,
            self.sync_url,
            query,
        )


# ── Module-level helpers ───────────────────────────────────────────────────

def _sync_read_sql(sync_url: str, query: str) -> pd.DataFrame:
    """Runs in a thread. Never call from async context directly."""
    engine = sa.create_engine(sync_url)
    try:
        with engine.connect() as conn:
            return pd.read_sql(sa.text(query), conn)
    finally:
        engine.dispose()


def _to_sync_url(async_url: str) -> str:
    """
    Strips async driver prefix so pandas can use the sync driver.
    "sqlite+aiosqlite:///..." → "sqlite:///..."
    "postgresql+asyncpg://..." → "postgresql://..."
    """
    return async_url.replace("+aiosqlite", "").replace("+asyncpg", "")


def _make_engine(url: str) -> AsyncEngine:
    if url.startswith("sqlite"):
        from sqlalchemy.pool import StaticPool
        return create_async_engine(
            url,
            connect_args={"check_same_thread": False},
            poolclass=StaticPool,
            echo=False,
        )
    return create_async_engine(
        url,
        pool_size=3,
        max_overflow=5,
        pool_pre_ping=True,
        pool_recycle=1800,
        echo=False,
    )
