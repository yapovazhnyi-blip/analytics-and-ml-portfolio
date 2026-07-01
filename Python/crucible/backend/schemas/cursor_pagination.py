"""
Cursor-based (keyset) pagination.

WHY KEYSET PAGINATION INSTEAD OF OFFSET/LIMIT
------------------------------------------------
OFFSET/LIMIT pagination ("page 500 of results") requires the database to
scan and discard every row before the offset. `LIMIT 20 OFFSET 10000`
forces a scan of 10,020 rows even though only 20 are returned. As a table
grows, every page past the first gets linearly slower — page 1000 takes
proportionally longer than page 1, even though the same constant amount
of work (20 rows) is conceptually being requested.

Keyset pagination (also called cursor pagination, or "seek method") instead
filters by a value that uniquely identifies the position in a sorted order:

  SELECT * FROM experiments
  WHERE (created_at, id) < (:cursor_created_at, :cursor_id)
  ORDER BY created_at DESC, id DESC
  LIMIT 20

This uses an index on (created_at, id) to seek directly to the right position —
O(log n + page_size) regardless of how deep into the result set you are.

CURSOR FORMAT
-------------
The cursor is an opaque base64-encoded string containing (created_at, id).
Clients should treat it as opaque — they pass it back unchanged to get the
next page. This insulates the API from changes to the underlying encoding.

TRADE-OFFS VS OFFSET PAGINATION
----------------------------------
Cursor pagination CANNOT:
  - Jump to an arbitrary page number ("go to page 47")
  - Show a total page count cheaply (COUNT(*) is still O(n))

Cursor pagination CAN:
  - Page through millions of rows at constant speed
  - Handle concurrent inserts/deletes without skipping or duplicating rows
    (offset pagination shows duplicates/gaps if rows are added/removed
    between page requests — keyset pagination does not have this problem,
    since each page is anchored to a specific row, not a numeric position)

Crucible exposes BOTH offset and cursor pagination:
  - Offset (?page=N) for UI page-number navigation (small datasets, < 10K rows)
  - Cursor (?cursor=X) for programmatic/API clients paging through everything
"""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional, TypeVar

from sqlalchemy import and_, or_, func
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.sql import Select

T = TypeVar("T")


def encode_cursor(created_at: datetime, id_: int) -> str:
    """Encodes a (created_at, id) position into an opaque cursor string."""
    payload = json.dumps({"t": created_at.isoformat(), "i": id_})
    return base64.urlsafe_b64encode(payload.encode()).decode()


def decode_cursor(cursor: str) -> tuple[datetime, int]:
    """Decodes a cursor string back into (created_at, id). Raises ValueError if malformed."""
    try:
        payload = json.loads(base64.urlsafe_b64decode(cursor.encode()).decode())
        return datetime.fromisoformat(payload["t"]), int(payload["i"])
    except Exception as exc:
        raise ValueError(f"Invalid cursor: {exc}") from exc


@dataclass
class CursorPage:
    """A page of results with a cursor for fetching the next page."""
    items: list[Any]
    next_cursor: Optional[str]
    has_more: bool


async def paginate_by_cursor(
    db: AsyncSession,
    stmt: Select,
    model,                          # ORM model class (must have .created_at and .id)
    cursor: Optional[str],
    limit: int = 20,
) -> CursorPage:
    """
    Applies keyset pagination to a SELECT statement and executes it.

    Args:
        db:     The async DB session.
        stmt:   A SELECT statement already filtered (e.g. WHERE dataset_id = X),
                but WITHOUT .order_by() or .limit() applied — this function adds both.
        model:  The ORM model class, used to reference .created_at and .id columns.
        cursor: Opaque cursor string from a previous page, or None for the first page.
        limit:  Page size.

    Returns:
        CursorPage with items, next_cursor (None if no more pages), and has_more flag.

    Ordering is always created_at DESC, id DESC (newest first) — matches the
    existing offset-based endpoints' default ordering for consistency.

    SQLITE PRECISION NOTE
    ----------------------
    SQLite has no native DATETIME type — TimestampMixin's server_default=func.now()
    is evaluated by SQLite's own datetime('now') function, which stores TEXT at
    *second* resolution ("2024-06-15 10:30:00"). But SQLAlchemy's bind parameter
    formatting for a Python datetime includes microseconds ("...10:30:00.000000")
    by default. Comparing these two string representations directly in a WHERE
    clause silently fails to match even identical moments in time, because SQLite
    compares TEXT-affinity columns lexicographically.

    This function detects the SQLite dialect and normalises both sides of the
    comparison to second resolution using strftime() before comparing. PostgreSQL
    (the production target — see .env.example) stores genuine microsecond-precision
    TIMESTAMPTZ values and round-trips exactly, so no normalisation is needed there.
    """
    is_sqlite = db.bind.dialect.name == "sqlite"

    if cursor:
        cursor_created_at, cursor_id = decode_cursor(cursor)

        if is_sqlite:
            # Normalise both sides to "YYYY-MM-DD HH:MM:SS" text before comparing,
            # matching the precision SQLite actually stores.
            created_norm = func.strftime("%Y-%m-%d %H:%M:%S", model.created_at)
            cursor_norm  = cursor_created_at.strftime("%Y-%m-%d %H:%M:%S")
            stmt = stmt.where(
                or_(
                    created_norm < cursor_norm,
                    and_(created_norm == cursor_norm, model.id < cursor_id),
                )
            )
        else:
            stmt = stmt.where(
                or_(
                    model.created_at < cursor_created_at,
                    and_(
                        model.created_at == cursor_created_at,
                        model.id < cursor_id,
                    ),
                )
            )

    stmt = stmt.order_by(model.created_at.desc(), model.id.desc()).limit(limit + 1)

    result = await db.scalars(stmt)
    rows = list(result.all())

    has_more = len(rows) > limit
    items = rows[:limit]

    next_cursor = None
    if has_more and items:
        last = items[-1]
        next_cursor = encode_cursor(last.created_at, last.id)

    return CursorPage(items=items, next_cursor=next_cursor, has_more=has_more)
