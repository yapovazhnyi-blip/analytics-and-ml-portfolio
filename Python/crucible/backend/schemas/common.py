"""
Common Pydantic schemas used across all Crucible API responses.

Every endpoint returns a consistent envelope:
  - Single resource:   {"data": {...}, "meta": {...}}
  - Collection:        {"data": [...], "pagination": {...}, "meta": {...}}
  - Error:             {"error": {"code": "...", "message": "...", "details": [...]}}

This consistency matters because the React frontend can handle all responses
with a single interceptor rather than bespoke parsing per endpoint.
"""

from __future__ import annotations

import datetime
from typing import Any, Generic, Optional, TypeVar
from pydantic import BaseModel, Field


T = TypeVar("T")


# ── Response envelope ──────────────────────────────────────────────────────

class Meta(BaseModel):
    timestamp: datetime.datetime = Field(
        default_factory=lambda: datetime.datetime.now(datetime.timezone.utc)
    )
    version: str = "v1"


class DataResponse(BaseModel, Generic[T]):
    """Single-resource response envelope."""
    data: T
    meta: Meta = Field(default_factory=Meta)


class PaginationMeta(BaseModel):
    page: int
    page_size: int
    total: int
    total_pages: int
    has_next: bool
    has_prev: bool


class PaginatedResponse(BaseModel, Generic[T]):
    """Collection response envelope with pagination."""
    data: list[T]
    pagination: PaginationMeta
    meta: Meta = Field(default_factory=Meta)


# ── Error response ─────────────────────────────────────────────────────────

class ErrorDetail(BaseModel):
    field: Optional[str] = None
    message: str


class ErrorBody(BaseModel):
    code: str        # machine-readable, e.g. "NOT_FOUND", "VALIDATION_ERROR"
    message: str     # human-readable
    details: list[ErrorDetail] = Field(default_factory=list)


class ErrorResponse(BaseModel):
    error: ErrorBody
    meta: Meta = Field(default_factory=Meta)


# ── Pagination query params ────────────────────────────────────────────────

class PaginationParams(BaseModel):
    """Reusable query parameter model for paginated list endpoints."""
    page: int = Field(default=1, ge=1, description="Page number (1-indexed)")
    page_size: int = Field(default=20, ge=1, le=100, description="Items per page")

    @property
    def offset(self) -> int:
        return (self.page - 1) * self.page_size


def make_pagination_meta(page: int, page_size: int, total: int) -> PaginationMeta:
    total_pages = max(1, -(-total // page_size))  # ceiling division
    return PaginationMeta(
        page=page,
        page_size=page_size,
        total=total,
        total_pages=total_pages,
        has_next=page < total_pages,
        has_prev=page > 1,
    )
