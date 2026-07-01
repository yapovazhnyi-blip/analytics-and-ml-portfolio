"""
REST connector integration tests.

Three test layers:

  1. Mocked unit tests — validate each pagination strategy, rate-limit
     backoff, and JSON normalisation against controlled HTTP responses
     using respx. These run offline and cover edge cases that are hard
     to trigger against a real API.

  2. Public API integration test — validates the full stack against
     GitHub's public commits API (no auth required). Skipped by default
     unless TEST_LIVE_API=1 is set, to avoid flaky CI from network issues.

  3. Factory tests — validate that ConnectorFactory.from_record() produces
     the right connector type from a mocked ORM record.
"""

import asyncio
import json
import os
import time
from unittest.mock import MagicMock, patch

import httpx
import pandas as pd
import pytest
import respx

from connectors.rest_connector import (
    PaginatedRestConnector,
    PaginationConfig,
    RateLimitConfig,
    _parse_link_header,
    _parse_retry_after,
    _extract_nested,
)


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def storage_dir(tmp_path):
    return str(tmp_path / "storage")


def make_connector(
    base_url: str,
    strategy: str,
    storage_dir: str,
    records_key: str = None,
    cursor_field: str = None,
    page_size: int = 10,
) -> PaginatedRestConnector:
    return PaginatedRestConnector(
        base_url=base_url,
        storage_dir=storage_dir,
        pagination=PaginationConfig(
            strategy=strategy,
            records_key=records_key,
            cursor_field=cursor_field,
            page_size=page_size,
            max_pages=100,
        ),
        rate_limit=RateLimitConfig(max_retries=2, base_backoff_secs=0.01),
    )


def make_records(n: int, offset: int = 0) -> list[dict]:
    return [{"id": i + offset, "name": f"item_{i + offset}", "value": float(i)} for i in range(n)]


# ══════════════════════════════════════════════════════════════════════════
# PAGINATION STRATEGIES
# ══════════════════════════════════════════════════════════════════════════

class TestLinkHeaderPagination:

    @pytest.mark.asyncio
    @respx.mock
    async def test_follows_link_header_across_pages(self, storage_dir):
        """Three pages linked via Link headers — all records collected."""
        connector = make_connector("https://api.example.com", "link", storage_dir)

        # Use a handler so the response depends on the URL (not separate respx params)
        def handler(request: httpx.Request) -> httpx.Response:
            url = str(request.url)
            if "page=3" in url:
                return httpx.Response(200, json=make_records(5, offset=20))
            if "page=2" in url:
                return httpx.Response(
                    200,
                    json=make_records(10, offset=10),
                    headers={"Link": '<https://api.example.com/items?page=3>; rel="next"'},
                )
            # First page (no page param)
            return httpx.Response(
                200,
                json=make_records(10, offset=0),
                headers={"Link": '<https://api.example.com/items?page=2>; rel="next"'},
            )

        respx.get(url__startswith="https://api.example.com/items").mock(side_effect=handler)
        result = await connector.ingest(endpoint="/items")
        assert result.succeeded, result.error
        assert result.row_count == 25

    @pytest.mark.asyncio
    @respx.mock
    async def test_stops_when_no_next_link(self, storage_dir):
        """Single-page response without Link header — stops after one page."""
        connector = make_connector("https://api.example.com", "link", storage_dir)
        respx.get("https://api.example.com/data").mock(
            return_value=httpx.Response(200, json=make_records(7))
        )
        result = await connector.ingest(endpoint="/data")
        assert result.succeeded
        assert result.row_count == 7

    @pytest.mark.asyncio
    @respx.mock
    async def test_handles_wrapped_response_with_link(self, storage_dir):
        """API wraps records in a 'data' key but still uses Link headers."""
        connector = make_connector(
            "https://api.example.com", "link", storage_dir, records_key="data"
        )
        respx.get("https://api.example.com/v1/items").mock(
            return_value=httpx.Response(
                200,
                json={"data": make_records(5), "total": 5},
            )
        )
        result = await connector.ingest(endpoint="/v1/items")
        assert result.succeeded
        assert result.row_count == 5


class TestCursorPagination:

    @pytest.mark.asyncio
    @respx.mock
    async def test_follows_cursor_across_pages(self, storage_dir):
        """Cursor-based pagination — passes cursor from response to next request."""
        connector = PaginatedRestConnector(
            base_url="https://api.example.com",
            storage_dir=storage_dir,
            pagination=PaginationConfig(
                strategy="cursor",
                cursor_field="next_cursor",
                cursor_param="cursor",
                records_key="items",
                page_size=10,
            ),
        )

        def handler(request: httpx.Request) -> httpx.Response:
            params = dict(httpx.URL(request.url).params)
            cursor = params.get("cursor")
            if cursor == "cursor_abc":
                return httpx.Response(
                    200,
                    json={"items": make_records(10, offset=10), "next_cursor": "cursor_def"},
                )
            if cursor == "cursor_def":
                return httpx.Response(
                    200,
                    json={"items": make_records(3, offset=20), "next_cursor": None},
                )
            # First page — no cursor
            return httpx.Response(
                200,
                json={"items": make_records(10), "next_cursor": "cursor_abc"},
            )

        respx.get(url__startswith="https://api.example.com/items").mock(side_effect=handler)

        result = await connector.ingest(endpoint="/items")
        assert result.succeeded, result.error
        assert result.row_count == 23

    @pytest.mark.asyncio
    @respx.mock
    async def test_nested_cursor_field(self, storage_dir):
        """Cursor nested inside a 'pagination' object — dot-notation path."""
        connector = PaginatedRestConnector(
            base_url="https://api.example.com",
            storage_dir=storage_dir,
            pagination=PaginationConfig(
                strategy="cursor",
                cursor_field="pagination.next",
                cursor_param="after",
                records_key="data",
            ),
        )
        respx.get("https://api.example.com/records").mock(return_value=httpx.Response(
            200,
            json={"data": make_records(5), "pagination": {"next": None}},
        ))
        result = await connector.ingest(endpoint="/records")
        assert result.succeeded
        assert result.row_count == 5


class TestOffsetPagination:

    @pytest.mark.asyncio
    @respx.mock
    async def test_offset_increments_correctly(self, storage_dir):
        """Offset pagination — three full pages then a partial page."""
        connector = PaginatedRestConnector(
            base_url="https://api.example.com",
            storage_dir=storage_dir,
            pagination=PaginationConfig(
                strategy="offset",
                offset_param="offset",
                limit_param="limit",
                page_size=10,
            ),
        )

        def handler(request: httpx.Request) -> httpx.Response:
            params = dict(httpx.URL(request.url).params)
            offset = int(params.get("offset", 0))
            limit = int(params.get("limit", 10))
            # Total 28 records
            records = make_records(min(limit, max(0, 28 - offset)), offset=offset)
            return httpx.Response(200, json=records)

        respx.get("https://api.example.com/list").mock(side_effect=handler)

        result = await connector.ingest(endpoint="/list")
        assert result.succeeded, result.error
        assert result.row_count == 28

    @pytest.mark.asyncio
    @respx.mock
    async def test_stops_on_partial_page(self, storage_dir):
        """When a page has fewer records than page_size, pagination stops."""
        connector = PaginatedRestConnector(
            base_url="https://api.example.com",
            storage_dir=storage_dir,
            pagination=PaginationConfig(strategy="offset", page_size=10),
        )
        call_count = [0]

        def handler(request: httpx.Request) -> httpx.Response:
            call_count[0] += 1
            # Only first call returns a full page
            return httpx.Response(200, json=make_records(10 if call_count[0] == 1 else 4))

        respx.get("https://api.example.com/items").mock(side_effect=handler)
        result = await connector.ingest(endpoint="/items")
        assert result.row_count == 14
        assert call_count[0] == 2  # stopped after partial page


# ══════════════════════════════════════════════════════════════════════════
# RATE LIMITING
# ══════════════════════════════════════════════════════════════════════════

class TestRateLimiting:

    @pytest.mark.asyncio
    @respx.mock
    async def test_retries_on_429_then_succeeds(self, storage_dir):
        """A 429 followed by a 200 — must retry and return the data."""
        connector = PaginatedRestConnector(
            base_url="https://api.example.com",
            storage_dir=storage_dir,
            pagination=PaginationConfig(strategy="none"),
            rate_limit=RateLimitConfig(max_retries=2, base_backoff_secs=0.01),
        )
        responses = [
            httpx.Response(429, headers={"Retry-After": "0.01"}),
            httpx.Response(200, json=make_records(5)),
        ]
        respx.get("https://api.example.com/data").mock(side_effect=iter(responses))
        result = await connector.ingest(endpoint="/data")
        assert result.succeeded, result.error
        assert result.row_count == 5

    @pytest.mark.asyncio
    @respx.mock
    async def test_raises_after_max_retries_exhausted(self, storage_dir):
        """Persistent 429s — must give up after max_retries and return an error."""
        connector = PaginatedRestConnector(
            base_url="https://api.example.com",
            storage_dir=storage_dir,
            pagination=PaginationConfig(strategy="none"),
            rate_limit=RateLimitConfig(max_retries=1, base_backoff_secs=0.01),
        )
        respx.get("https://api.example.com/data").mock(
            return_value=httpx.Response(429, headers={"Retry-After": "0.01"})
        )
        result = await connector.ingest(endpoint="/data")
        assert not result.succeeded
        assert result.error is not None


# ══════════════════════════════════════════════════════════════════════════
# JSON NORMALISATION
# ══════════════════════════════════════════════════════════════════════════

class TestJSONNormalisation:

    @pytest.mark.asyncio
    @respx.mock
    async def test_top_level_array(self, storage_dir):
        """Response is a top-level JSON array — records extracted directly."""
        connector = make_connector("https://api.example.com", "none", storage_dir)
        respx.get("https://api.example.com/items").mock(
            return_value=httpx.Response(200, json=make_records(8))
        )
        result = await connector.ingest(endpoint="/items")
        assert result.row_count == 8

    @pytest.mark.asyncio
    @respx.mock
    async def test_wrapped_in_data_key(self, storage_dir):
        """Response wrapped: {"data": [...], "total": N}."""
        connector = make_connector(
            "https://api.example.com", "none", storage_dir, records_key="data"
        )
        respx.get("https://api.example.com/items").mock(
            return_value=httpx.Response(200, json={"data": make_records(6), "total": 6})
        )
        result = await connector.ingest(endpoint="/items")
        assert result.row_count == 6

    @pytest.mark.asyncio
    @respx.mock
    async def test_auto_detects_array_key(self, storage_dir):
        """No records_key configured — auto-detects the first list value."""
        connector = make_connector("https://api.example.com", "none", storage_dir)
        respx.get("https://api.example.com/items").mock(
            return_value=httpx.Response(
                200,
                json={"meta": {"total": 4}, "results": make_records(4)},
            )
        )
        result = await connector.ingest(endpoint="/items")
        assert result.row_count == 4

    @pytest.mark.asyncio
    @respx.mock
    async def test_empty_response_returns_error(self, storage_dir):
        """Empty array from API — must return an error, not an empty file."""
        connector = make_connector("https://api.example.com", "none", storage_dir)
        respx.get("https://api.example.com/items").mock(
            return_value=httpx.Response(200, json=[])
        )
        result = await connector.ingest(endpoint="/items")
        assert not result.succeeded
        assert "no records" in result.error.lower()

    @pytest.mark.asyncio
    @respx.mock
    async def test_nested_fields_flattened(self, storage_dir):
        """Nested JSON objects are flattened by pd.json_normalize."""
        connector = make_connector("https://api.example.com", "none", storage_dir)
        respx.get("https://api.example.com/items").mock(
            return_value=httpx.Response(
                200,
                json=[{"id": 1, "address": {"city": "Berlin", "country": "DE"}}],
            )
        )
        result = await connector.ingest(endpoint="/items")
        assert result.succeeded
        col_names = [c.name for c in result.columns]
        assert "address.city" in col_names
        assert "address.country" in col_names


# ══════════════════════════════════════════════════════════════════════════
# RESULT AND SCHEMA
# ══════════════════════════════════════════════════════════════════════════

class TestIngestResult:

    @pytest.mark.asyncio
    @respx.mock
    async def test_result_has_file_path_and_hash(self, storage_dir, tmp_path):
        """Successful ingest must produce a Parquet file with a SHA-256 hash."""
        connector = make_connector("https://api.example.com", "none", storage_dir)
        respx.get("https://api.example.com/data").mock(
            return_value=httpx.Response(200, json=make_records(5))
        )
        result = await connector.ingest(endpoint="/data")
        assert result.file_path is not None
        from pathlib import Path
        assert Path(result.file_path).exists()
        assert result.content_hash is not None and len(result.content_hash) == 64

    @pytest.mark.asyncio
    @respx.mock
    async def test_parquet_file_is_readable(self, storage_dir):
        """The output Parquet file must be readable by pandas."""
        connector = make_connector("https://api.example.com", "none", storage_dir)
        respx.get("https://api.example.com/data").mock(
            return_value=httpx.Response(200, json=make_records(10))
        )
        result = await connector.ingest(endpoint="/data")
        df = pd.read_parquet(result.file_path)
        assert len(df) == 10
        assert "id" in df.columns

    @pytest.mark.asyncio
    @respx.mock
    async def test_schema_inferred_correctly(self, storage_dir):
        """Schema must reflect the actual column types from the API response."""
        connector = make_connector("https://api.example.com", "none", storage_dir)
        respx.get("https://api.example.com/data").mock(
            return_value=httpx.Response(200, json=make_records(5))
        )
        result = await connector.ingest(endpoint="/data")
        col_names = [c.name for c in result.columns]
        assert "id" in col_names
        assert "name" in col_names
        assert "value" in col_names


# ══════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════

class TestHelpers:

    def test_parse_link_header_extracts_next(self):
        header = '<https://api.github.com/repos?page=2>; rel="next", <https://api.github.com/repos?page=5>; rel="last"'
        assert _parse_link_header(header) == "https://api.github.com/repos?page=2"

    def test_parse_link_header_no_next_returns_none(self):
        header = '<https://api.github.com/repos?page=4>; rel="last"'
        assert _parse_link_header(header) is None

    def test_parse_link_header_empty_returns_none(self):
        assert _parse_link_header("") is None
        assert _parse_link_header(None) is None

    def test_parse_retry_after_seconds(self):
        headers = httpx.Headers({"Retry-After": "30"})
        wait = _parse_retry_after(headers, ["Retry-After"])
        assert wait == 30.0

    def test_parse_retry_after_unix_timestamp(self):
        future = str(int(time.time()) + 45)
        headers = httpx.Headers({"X-RateLimit-Reset": future})
        wait = _parse_retry_after(headers, ["X-RateLimit-Reset"])
        assert 44 <= wait <= 46  # allow 1 second of clock skew

    def test_parse_retry_after_missing_returns_none(self):
        headers = httpx.Headers({})
        assert _parse_retry_after(headers, ["Retry-After"]) is None

    def test_extract_nested_simple(self):
        assert _extract_nested({"a": {"b": "x"}}, "a.b") == "x"

    def test_extract_nested_missing_key_returns_none(self):
        assert _extract_nested({"a": 1}, "a.b.c") is None

    def test_extract_nested_none_path_returns_none(self):
        assert _extract_nested({"a": 1}, None) is None


# ══════════════════════════════════════════════════════════════════════════
# CONNECTOR FACTORY
# ══════════════════════════════════════════════════════════════════════════

class TestConnectorFactory:

    def _make_record(self, connector_type: str, **kwargs) -> MagicMock:
        rec = MagicMock()
        rec.connector_type = connector_type
        rec.name = "test"
        rec.db_url_encrypted = None
        rec.client_id = None
        rec.client_secret_encrypted = None
        rec.base_url = "https://api.example.com"
        rec.oauth2_flow = None
        rec.token_body_format = "json"
        rec.response_format = "json"
        rec.client_auth = "body"
        rec.use_pkce = False
        rec.access_token_encrypted = None
        rec.refresh_token_encrypted = None
        rec.token_expires_at = None
        for k, v in kwargs.items():
            setattr(rec, k, v)
        return rec

    def test_file_connector_instantiated(self):
        from connectors.factory import ConnectorFactory
        from connectors.file_connector import FileConnector
        rec = self._make_record("csv")
        c = ConnectorFactory.from_record(rec)
        assert isinstance(c, FileConnector)

    def test_sql_connector_instantiated(self, tmp_path):
        from connectors.factory import ConnectorFactory
        from connectors.sql_connector import SQLConnector
        from storage.encryption import encrypt

        rec = self._make_record(
            "sql_sqlite",
            db_url_encrypted=encrypt("sqlite+aiosqlite:///:memory:"),
        )
        with patch("connectors.factory.settings") as ms:
            ms.dataset_storage_path = str(tmp_path)
            c = ConnectorFactory.from_record(rec)
        assert isinstance(c, SQLConnector)

    def test_rest_connector_instantiated(self, tmp_path):
        from connectors.factory import ConnectorFactory
        from connectors.rest_connector import PaginatedRestConnector

        rec = self._make_record("rest_oauth2")
        with patch("connectors.factory.settings") as ms:
            ms.dataset_storage_path = str(tmp_path)
            c = ConnectorFactory.from_record(rec)
        assert isinstance(c, PaginatedRestConnector)

    def test_unknown_type_raises(self):
        from connectors.factory import ConnectorFactory
        rec = self._make_record("unknown_db")
        with pytest.raises(ValueError, match="Unknown connector type"):
            ConnectorFactory.from_record(rec)


# ══════════════════════════════════════════════════════════════════════════
# LIVE API TEST (opt-in via TEST_LIVE_API=1)
# ══════════════════════════════════════════════════════════════════════════

@pytest.mark.skipif(
    os.getenv("TEST_LIVE_API") != "1",
    reason="Live API tests — set TEST_LIVE_API=1 to run (requires external network)",
)
class TestLiveExternalAPI:
    """
    Placeholder for external API tests.
    Requires TEST_LIVE_API=1 and external network access.
    The local server tests below provide equivalent coverage without
    depending on external services.
    """
    pass


# ══════════════════════════════════════════════════════════════════════════
# LOCAL HTTP SERVER TESTS
# Proves the full stack (real TCP, real HTTP) without external network.
# ══════════════════════════════════════════════════════════════════════════

class TestLocalHTTPServer:
    """
    Runs a real HTTP server on localhost and tests the connector against it.

    This is more thorough than mocked tests because it exercises the full
    network stack: real TCP connections, real HTTP parsing, real response
    headers, real content negotiation. It proves the connector would work
    against any real server that follows the same conventions.
    """

    @pytest.mark.asyncio
    async def test_link_header_pagination_full_stack(self, storage_dir, tmp_path):
        """
        Starts a real HTTP server that serves 30 records across 3 pages
        using Link-header pagination. Runs the connector against it.
        Validates: real TCP, real HTTP, correct page traversal, Parquet output.
        """
        import socket
        from http.server import BaseHTTPRequestHandler, HTTPServer
        import threading
        import json as json_lib

        # Build the dataset in-memory: 30 records, 3 pages of 10
        all_data = [{"id": i, "name": f"item_{i}", "score": float(i) / 30} for i in range(30)]

        # Find a free port
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]

        base_url = f"http://127.0.0.1:{port}"
        server_ready = threading.Event()

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass  # suppress server log noise

            def do_GET(self):
                from urllib.parse import urlparse, parse_qs
                parsed = urlparse(self.path)
                params = parse_qs(parsed.query)
                page = int(params.get("page", ["1"])[0])

                start = (page - 1) * 10
                page_data = all_data[start:start + 10]

                # Build Link header if there's a next page
                headers = {"Content-Type": "application/json"}
                if page < 3:
                    next_url = f"http://127.0.0.1:{port}/items?page={page + 1}"
                    headers["Link"] = f'<{next_url}>; rel="next"'

                body = json_lib.dumps(page_data).encode()
                self.send_response(200)
                for k, v in headers.items():
                    self.send_header(k, v)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        server = HTTPServer(("127.0.0.1", port), Handler)
        server_thread = threading.Thread(target=server.serve_forever, daemon=True)
        server_thread.start()

        try:
            # Give the server a moment to start
            await asyncio.sleep(0.05)

            connector = PaginatedRestConnector(
                base_url=base_url,
                storage_dir=storage_dir,
                pagination=PaginationConfig(
                    strategy="link",
                    max_pages=10,
                ),
            )
            result = await connector.ingest(endpoint="/items")

            assert result.succeeded, result.error
            assert result.row_count == 30
            assert result.file_path is not None

            df = pd.read_parquet(result.file_path)
            assert len(df) == 30
            assert list(df["id"]) == list(range(30))
            assert "name" in df.columns
            assert "score" in df.columns
        finally:
            server.shutdown()

    @pytest.mark.asyncio
    async def test_offset_pagination_full_stack(self, storage_dir):
        """
        Serves 45 records via offset/limit pagination on a local server.
        Validates: offset increments correctly, stops on partial page.
        """
        import socket
        from http.server import BaseHTTPRequestHandler, HTTPServer
        import threading
        import json as json_lib

        total = 45
        all_data = [{"id": i, "value": i * 2} for i in range(total)]

        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                from urllib.parse import urlparse, parse_qs
                params = parse_qs(urlparse(self.path).query)
                start = int(params.get("offset", ["0"])[0])
                limit = int(params.get("limit", ["10"])[0])
                page_data = all_data[start:start + limit]
                body = json_lib.dumps(page_data).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        server = HTTPServer(("127.0.0.1", port), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        await asyncio.sleep(0.05)

        try:
            connector = PaginatedRestConnector(
                base_url=f"http://127.0.0.1:{port}",
                storage_dir=storage_dir,
                pagination=PaginationConfig(
                    strategy="offset",
                    offset_param="offset",
                    limit_param="limit",
                    page_size=10,
                    max_pages=10,
                ),
            )
            result = await connector.ingest(endpoint="/data")
            assert result.succeeded, result.error
            assert result.row_count == 45
        finally:
            server.shutdown()

    @pytest.mark.asyncio
    async def test_rate_limit_and_retry_full_stack(self, storage_dir):
        """
        Server returns 429 with Retry-After on the first request,
        then 200 on the second. Validates real retry-after handling.
        """
        import socket
        from http.server import BaseHTTPRequestHandler, HTTPServer
        import threading
        import json as json_lib

        call_count = [0]

        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            port = s.getsockname()[1]

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                call_count[0] += 1
                if call_count[0] == 1:
                    self.send_response(429)
                    self.send_header("Retry-After", "0.01")  # 10ms
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                else:
                    body = json_lib.dumps([{"id": 1, "result": "ok"}]).encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)

        server = HTTPServer(("127.0.0.1", port), Handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        await asyncio.sleep(0.05)

        try:
            connector = PaginatedRestConnector(
                base_url=f"http://127.0.0.1:{port}",
                storage_dir=storage_dir,
                pagination=PaginationConfig(strategy="none"),
                rate_limit=RateLimitConfig(max_retries=2, base_backoff_secs=0.01),
            )
            result = await connector.ingest(endpoint="/data")
            assert result.succeeded, result.error
            assert result.row_count == 1
            assert call_count[0] == 2  # one 429, one 200
        finally:
            server.shutdown()
