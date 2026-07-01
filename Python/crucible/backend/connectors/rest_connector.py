"""
PaginatedRestConnector for Crucible.

This is the layer that sits above the OAuth2 authentication base and
actually fetches data from REST APIs. The key challenge it solves is
that REST APIs do not agree on how to paginate — every provider has
its own convention — so the connector must handle all major strategies
without requiring per-provider subclasses.

Three pagination strategies are supported:

  cursor    — the response body contains a "next cursor" field. The
              client sends cursor=<value> on the next request until
              the cursor field is absent or null. Used by: Stripe,
              Notion, many modern APIs.

  offset    — each request sends offset=N&limit=M. The client
              increments offset by the page size until it receives
              fewer rows than the page size. Used by: most SQL-backed
              APIs, older REST APIs.

  link      — the response has a Link header with rel="next" pointing
              to the full URL of the next page. Used by: GitHub,
              GitLab, many developer-oriented APIs.

Rate limiting is handled by inspecting the response for 429 status
codes and the Retry-After header, with exponential backoff as a
fallback when no header is present.

JSON normalisation handles the common pattern where the actual array
of records is nested inside a wrapper key (e.g., {"data": [...],
"total": 1234}). The connector infers which key contains the array,
or falls back to treating the entire response as a single record.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, AsyncIterator, Optional
from urllib.parse import parse_qs, urlencode, urljoin, urlparse

import httpx
import pandas as pd

from connectors.base import BaseConnector, IngestResult, infer_columns, sha256_of_bytes
from connectors.auth.oauth2 import OAuth2Connector


# ── Pagination strategy config ─────────────────────────────────────────────

@dataclass
class PaginationConfig:
    """
    Describes how a specific API paginates its responses.

    strategy:
        "cursor"  — response body contains a next-cursor field
        "offset"  — use offset + limit query parameters
        "link"    — follow the Link: <url>; rel="next" response header
        "none"    — single-page API, no pagination needed

    cursor_field:
        For "cursor" strategy — the JSON key path to the next cursor
        value. Supports dot-notation for nested keys:
        "pagination.next_cursor" → response["pagination"]["next_cursor"]

    cursor_param:
        For "cursor" strategy — the query parameter name to send the
        cursor value in (default: "cursor").

    records_key:
        The JSON key that contains the array of records. If None, the
        connector will auto-detect by looking for the first list value
        in the response. For APIs that return a top-level array, set
        this to None explicitly.

    offset_param / limit_param:
        For "offset" strategy — the query parameter names.

    page_size:
        Number of records per page (default 100).

    max_pages:
        Safety limit to prevent infinite loops on misbehaving APIs
        (default 500 pages × page_size = 50,000 rows max by default).
    """
    strategy: str = "link"                 # cursor | offset | link | none
    cursor_field: Optional[str] = None     # e.g. "pagination.next_cursor"
    cursor_param: str = "cursor"
    records_key: Optional[str] = None      # e.g. "data", "items", "results"
    offset_param: str = "offset"
    limit_param: str = "limit"
    page_size: int = 100
    max_pages: int = 500


# ── Rate limit config ──────────────────────────────────────────────────────

@dataclass
class RateLimitConfig:
    """
    Controls how the connector responds to rate limiting.

    max_retries:
        How many times to retry a 429 before giving up.

    base_backoff_secs:
        Starting wait time for exponential backoff when no Retry-After
        header is present. Doubles on each retry.

    max_backoff_secs:
        Cap on the exponential backoff (to prevent hour-long waits).

    retry_after_headers:
        Headers to check for a server-specified retry delay, in order
        of preference. Most providers use Retry-After (seconds) or
        X-RateLimit-Reset (unix timestamp).
    """
    max_retries: int = 3
    base_backoff_secs: float = 1.0
    max_backoff_secs: float = 60.0
    retry_after_headers: list[str] = field(default_factory=lambda: [
        "Retry-After",
        "X-RateLimit-Reset",
        "x-rate-limit-reset",
    ])


# ── The connector ──────────────────────────────────────────────────────────

class PaginatedRestConnector(BaseConnector):
    """
    Fetches paginated data from a REST API and materialises it as Parquet.

    Usage with client-credentials OAuth (e.g. Stripe):
        auth = ClientCredentialsConnector(stripe_config, httpx.AsyncClient())
        connector = PaginatedRestConnector(
            auth_connector=auth,
            base_url="https://api.stripe.com",
            storage_dir="/data/datasets",
            pagination=PaginationConfig(
                strategy="cursor",
                cursor_field="data.-1.id",  # last item's id
                cursor_param="starting_after",
                records_key="data",
            ),
        )
        result = await connector.ingest(endpoint="/v1/charges", params={"limit": 100})

    Usage without authentication (public API):
        connector = PaginatedRestConnector(
            base_url="https://api.github.com",
            storage_dir="/data/datasets",
            pagination=PaginationConfig(strategy="link"),
        )
        result = await connector.ingest(
            endpoint="/repos/torvalds/linux/commits",
            params={"per_page": 100},
            headers={"Accept": "application/vnd.github+json"},
        )
    """

    def __init__(
        self,
        base_url: str,
        storage_dir: str,
        auth_connector: Optional[OAuth2Connector] = None,
        pagination: Optional[PaginationConfig] = None,
        rate_limit: Optional[RateLimitConfig] = None,
        timeout_secs: float = 30.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.auth_connector = auth_connector
        self.pagination = pagination or PaginationConfig()
        self.rate_limit = rate_limit or RateLimitConfig()
        self._client: Optional[httpx.AsyncClient] = None

    async def _get_client(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=self.pagination.page_size)
        return self._client

    async def close(self) -> None:
        if self._client:
            await self._client.aclose()
            self._client = None

    async def ingest(
        self,
        endpoint: str,
        params: Optional[dict] = None,
        headers: Optional[dict] = None,
        **kwargs,
    ) -> IngestResult:
        """
        Fetches all pages from the endpoint and saves the result as Parquet.

        Returns an IngestResult with schema, row count, file path, and
        content hash — the same shape as FileConnector and SQLConnector
        so the datasets router doesn't need to know which connector ran.
        """
        try:
            all_records: list[dict] = []
            async for batch in self._paginate(endpoint, params or {}, headers or {}):
                all_records.extend(batch)

            if not all_records:
                return IngestResult(
                    source_type="rest_api",
                    error="API returned no records",
                )

            df = pd.json_normalize(all_records)
            parquet_name = f"{uuid.uuid4().hex}.parquet"
            file_path = self.storage_dir / parquet_name
            df.to_parquet(file_path, index=False)

            content = file_path.read_bytes()
            return IngestResult(
                source_type="rest_api",
                file_path=str(file_path),
                content_hash=sha256_of_bytes(content),
                row_count=len(df),
                column_count=len(df.columns),
                columns=infer_columns(df),
            )
        except Exception as exc:
            return IngestResult(source_type="rest_api", error=str(exc))

    async def test_connection(self) -> tuple[bool, Optional[str]]:
        """
        Lightweight connectivity check — hits the base URL and returns
        (success, error_message). Does not fetch any data.
        """
        try:
            t0 = time.monotonic()
            resp = await self._request("GET", "/", params={}, headers={})
            # 404 is fine — the root may not exist, but the server responded
            if resp.status_code < 500:
                return True, None
            return False, f"Server returned {resp.status_code}"
        except Exception as exc:
            return False, str(exc)

    # ── Pagination engine ──────────────────────────────────────────────────

    async def _paginate(
        self,
        endpoint: str,
        params: dict,
        headers: dict,
    ) -> AsyncIterator[list[dict]]:
        """
        Yields batches of records, handling pagination transparently.

        The strategy is determined by self.pagination.strategy and
        dispatches to the appropriate method. All strategies share the
        same rate-limit-aware _request() method.
        """
        strategy = self.pagination.strategy

        if strategy == "none":
            resp = await self._request("GET", endpoint, params, headers)
            records = self._extract_records(resp.json())
            if records:
                yield records

        elif strategy == "link":
            yield (records := [])
            async for batch in self._paginate_link(endpoint, params, headers):
                yield batch

        elif strategy == "cursor":
            async for batch in self._paginate_cursor(endpoint, params, headers):
                yield batch

        elif strategy == "offset":
            async for batch in self._paginate_offset(endpoint, params, headers):
                yield batch

        else:
            raise ValueError(f"Unknown pagination strategy: {strategy!r}")

    async def _paginate_link(
        self,
        endpoint: str,
        params: dict,
        headers: dict,
    ) -> AsyncIterator[list[dict]]:
        """
        Link-header pagination. The server sends a Link header like:
            Link: <https://api.example.com/items?page=2>; rel="next"

        We follow rel="next" until it's absent. This is the cleanest
        pagination style because the client doesn't need to know the
        URL structure — it just follows the server's instructions.

        Critical: for pages 2+ we pass params=None (not {}) to _request_url.
        httpx treats params={} as "replace the query string with nothing",
        stripping any ?page=N that was embedded in the Link header URL.
        params=None preserves the URL's existing query string unchanged.
        """
        url = f"{self.base_url}{endpoint}"
        page_count = 0

        while url and page_count < self.pagination.max_pages:
            # First page uses caller-supplied params; subsequent pages use
            # the full URL from the Link header (params must be None to
            # preserve the query string httpx would otherwise strip)
            request_params = params if page_count == 0 else None
            resp = await self._request_url("GET", url, request_params, headers)
            records = self._extract_records(resp.json())
            if records:
                yield records
            page_count += 1
            url = _parse_link_header(resp.headers.get("Link", ""))

    async def _paginate_cursor(
        self, endpoint: str, params: dict, headers: dict
    ) -> AsyncIterator[list[dict]]:
        """
        Cursor-based pagination. After each page, we extract a cursor
        value from the response body and send it as a query parameter
        on the next request. When the cursor is absent or null, we stop.

        This is popular with modern APIs (Stripe, Notion, Linear) because
        it works correctly even when records are inserted between pages.
        """
        current_params = dict(params)
        page_count = 0

        while page_count < self.pagination.max_pages:
            resp = await self._request("GET", endpoint, current_params, headers)
            body = resp.json()
            records = self._extract_records(body)
            if records:
                yield records
            page_count += 1

            cursor = _extract_nested(body, self.pagination.cursor_field)
            if not cursor:
                break
            current_params[self.pagination.cursor_param] = cursor

    async def _paginate_offset(
        self, endpoint: str, params: dict, headers: dict
    ) -> AsyncIterator[list[dict]]:
        """
        Offset/limit pagination. We increment the offset by page_size
        after each request and stop when we receive fewer records than
        the page size (meaning we've reached the last page).

        Simple and reliable, but has a subtle bug: if records are
        deleted between pages, you can miss rows. Acceptable for
        data ingestion where consistency is better than perfection.
        """
        offset = 0
        page_size = self.pagination.page_size
        current_params = {
            **params,
            self.pagination.limit_param: page_size,
        }
        page_count = 0

        while page_count < self.pagination.max_pages:
            current_params[self.pagination.offset_param] = offset
            resp = await self._request("GET", endpoint, current_params, headers)
            records = self._extract_records(resp.json())
            if records:
                yield records
            page_count += 1

            # Stop when we get fewer records than the page size
            if len(records) < page_size:
                break
            offset += page_size

    # ── Rate-limit-aware HTTP ──────────────────────────────────────────────

    async def _request(
        self, method: str, endpoint: str, params: Optional[dict], headers: dict
    ) -> httpx.Response:
        url = f"{self.base_url}{endpoint}"
        return await self._request_url(method, url, params, headers)

    async def _request_url(
        self, method: str, url: str, params: Optional[dict], headers: dict
    ) -> httpx.Response:
        """
        Makes a single HTTP request with authentication and rate-limit
        handling. Retries on 429 with exponential backoff.

        params=None preserves the URL's existing query string (important
        for link-header pagination where the next-page URL is self-contained).
        params={} would strip the query string — always pass None when
        following a Link header URL.
        """
        request_headers = dict(headers)
        if self.auth_connector:
            token = await self.auth_connector.get_token()
            request_headers["Authorization"] = f"{token.token_type} {token.access_token}"

        client = httpx.AsyncClient(timeout=30.0)
        backoff = self.rate_limit.base_backoff_secs
        last_exc: Optional[Exception] = None

        for attempt in range(self.rate_limit.max_retries + 1):
            try:
                resp = await client.request(
                    method, url,
                    params=params,
                    headers=request_headers,
                )

                if resp.status_code == 429:
                    wait = _parse_retry_after(resp.headers, self.rate_limit.retry_after_headers)
                    if wait is None:
                        wait = min(backoff, self.rate_limit.max_backoff_secs)
                        backoff *= 2
                    if attempt < self.rate_limit.max_retries:
                        await asyncio.sleep(wait)
                        continue
                    resp.raise_for_status()

                resp.raise_for_status()
                await client.aclose()
                return resp

            except httpx.HTTPStatusError:
                raise
            except Exception as exc:
                last_exc = exc
                if attempt < self.rate_limit.max_retries:
                    await asyncio.sleep(min(backoff, self.rate_limit.max_backoff_secs))
                    backoff *= 2

        await client.aclose()
        raise last_exc or RuntimeError("Request failed after retries")

    # ── JSON normalisation ─────────────────────────────────────────────────

    def _extract_records(self, body: Any) -> list[dict]:
        """
        Extracts the array of records from a JSON response body.

        Handles three common shapes:
          1. Top-level array: [...] → return as-is
          2. Known wrapper key: {"data": [...], "total": 42} → return body["data"]
          3. Auto-detect: find the first list value in the object
          4. Single record: {} → wrap in a list

        The records_key config takes priority over auto-detection.
        """
        if isinstance(body, list):
            return [r for r in body if isinstance(r, dict)]

        if not isinstance(body, dict):
            return []

        # Use configured key if specified
        if self.pagination.records_key:
            val = body.get(self.pagination.records_key)
            if isinstance(val, list):
                return [r for r in val if isinstance(r, dict)]
            return []

        # Auto-detect: find the first list value
        for value in body.values():
            if isinstance(value, list) and value:
                if isinstance(value[0], dict):
                    return value

        # Single-record response
        return [body]


# ── Helpers ────────────────────────────────────────────────────────────────

def _parse_link_header(header: str) -> Optional[str]:
    """
    Parses GitHub-style Link headers:
        <https://api.github.com/repos?page=2>; rel="next", <...>; rel="last"

    Returns the URL for rel="next", or None if not present.
    """
    if not header:
        return None
    for part in header.split(","):
        part = part.strip()
        if 'rel="next"' in part or "rel='next'" in part:
            url_part = part.split(";")[0].strip()
            if url_part.startswith("<") and url_part.endswith(">"):
                return url_part[1:-1]
    return None


def _parse_retry_after(
    headers: httpx.Headers,
    header_names: list[str],
) -> Optional[float]:
    """
    Extracts a retry delay from rate-limit response headers.

    Handles two common formats:
      - Retry-After: 30          (seconds to wait)
      - X-RateLimit-Reset: 1699000000  (unix timestamp of when limit resets)
    """
    for name in header_names:
        value = headers.get(name)
        if value is None:
            continue
        try:
            v = float(value)
            # If it looks like a unix timestamp (> 1 billion), convert to wait time
            if v > 1_000_000_000:
                wait = v - time.time()
                return max(0.0, wait)
            return v
        except (ValueError, TypeError):
            continue
    return None


def _extract_nested(obj: Any, path: Optional[str]) -> Any:
    """
    Extracts a value from a nested dict using dot-notation.
    "_extract_nested({"a": {"b": "x"}}, "a.b") → "x"
    Returns None if any key in the path is missing.
    """
    if not path or not isinstance(obj, dict):
        return None
    parts = path.split(".")
    current = obj
    for part in parts:
        if isinstance(current, dict):
            current = current.get(part)
        elif isinstance(current, list):
            # Numeric index into a list
            try:
                current = current[int(part)]
            except (ValueError, IndexError):
                return None
        else:
            return None
    return current
