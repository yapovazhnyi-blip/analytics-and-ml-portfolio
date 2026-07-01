"""
FastAPI skeleton integration tests.

Uses an in-memory SQLite database so tests are fully isolated — no
file system pollution, no shared state between test runs.

The TestClient from Starlette runs the full ASGI stack synchronously,
including lifespan (startup/shutdown), so these tests cover the real
request/response cycle, not just unit logic.
"""

import io
import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

# ── Test DB setup ──────────────────────────────────────────────────────────

TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


def get_test_engine():
    return create_async_engine(
        TEST_DATABASE_URL,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )


# ── App fixture with DB override ───────────────────────────────────────────

@pytest.fixture
def client():
    """
    Returns a TestClient wired to an in-memory SQLite database.

    Uses importlib to reload main so module-level state doesn't leak
    between test files when the full suite runs together.
    """
    import sys
    import importlib
    import database

    test_engine = get_test_engine()
    test_factory = async_sessionmaker(test_engine, expire_on_commit=False)

    database.engine = test_engine
    database.SessionFactory = test_factory
    database.AsyncSessionLocal = test_factory

    # Reload main to pick up fresh engine/factory references
    if "main" in sys.modules:
        importlib.reload(sys.modules["main"])
    import main as app_module

    with TestClient(app_module.app, raise_server_exceptions=True) as c:
        yield c

    # Clean teardown: asyncio.run() creates a fresh event loop,
    # which is safer than get_event_loop() in Python 3.10+
    import asyncio
    asyncio.run(test_engine.dispose())


# ══════════════════════════════════════════════════════════════════════════
# HEALTH CHECKS
# ══════════════════════════════════════════════════════════════════════════

class TestHealth:

    def test_liveness_returns_ok(self, client):
        resp = client.get("/api/v1/health/live")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    def test_readiness_returns_ok_when_db_reachable(self, client):
        resp = client.get("/api/v1/health/ready")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
        assert resp.json()["database"] == "reachable"

    def test_root_redirects_to_docs(self, client):
        resp = client.get("/", follow_redirects=False)
        assert resp.status_code in (301, 302, 307, 308)
        assert "/docs" in resp.headers["location"]


# ══════════════════════════════════════════════════════════════════════════
# DATASETS
# ══════════════════════════════════════════════════════════════════════════

class TestDatasets:

    def _make_csv(self, content: str = "a,b,target\n1,2,0\n3,4,1\n5,6,0\n") -> bytes:
        return content.encode()

    def test_list_datasets_empty(self, client):
        resp = client.get("/api/v1/datasets")
        assert resp.status_code == 200
        body = resp.json()
        assert body["data"] == []
        assert body["pagination"]["total"] == 0

    def test_upload_csv(self, client, tmp_path, monkeypatch):
        """Upload a CSV file — must return a ready dataset with schema."""
        # Patch storage path to use tmp_path
        import config
        monkeypatch.setattr(config.settings, "dataset_storage_path", str(tmp_path))

        csv_bytes = self._make_csv()
        resp = client.post(
            "/api/v1/datasets/upload",
            files={"file": ("test.csv", io.BytesIO(csv_bytes), "text/csv")},
            data={"name": "test_dataset"},
        )
        assert resp.status_code == 201, resp.text
        body = resp.json()["data"]
        assert body["name"] == "test_dataset"
        assert body["source_type"] == "csv"
        assert body["status"] == "ready"
        assert body["row_count"] == 3
        assert body["column_count"] == 3

    def test_upload_unsupported_format_returns_422(self, client):
        resp = client.post(
            "/api/v1/datasets/upload",
            files={"file": ("model.pkl", io.BytesIO(b"data"), "application/octet-stream")},
        )
        assert resp.status_code == 422

    def test_get_nonexistent_dataset_returns_404(self, client):
        resp = client.get("/api/v1/datasets/9999")
        assert resp.status_code == 404

    def test_upload_then_get(self, client, tmp_path, monkeypatch):
        import config
        monkeypatch.setattr(config.settings, "dataset_storage_path", str(tmp_path))

        csv_bytes = self._make_csv()
        upload_resp = client.post(
            "/api/v1/datasets/upload",
            files={"file": ("data.csv", io.BytesIO(csv_bytes), "text/csv")},
            data={"name": "my_data"},
        )
        dataset_id = upload_resp.json()["data"]["id"]

        get_resp = client.get(f"/api/v1/datasets/{dataset_id}")
        assert get_resp.status_code == 200
        assert get_resp.json()["data"]["id"] == dataset_id

    def test_upload_then_delete(self, client, tmp_path, monkeypatch):
        import config
        monkeypatch.setattr(config.settings, "dataset_storage_path", str(tmp_path))

        csv_bytes = self._make_csv()
        upload_resp = client.post(
            "/api/v1/datasets/upload",
            files={"file": ("del_me.csv", io.BytesIO(csv_bytes), "text/csv")},
        )
        dataset_id = upload_resp.json()["data"]["id"]

        del_resp = client.delete(f"/api/v1/datasets/{dataset_id}")
        assert del_resp.status_code == 204

        get_resp = client.get(f"/api/v1/datasets/{dataset_id}")
        assert get_resp.status_code == 404

    def test_list_datasets_after_upload(self, client, tmp_path, monkeypatch):
        import config
        monkeypatch.setattr(config.settings, "dataset_storage_path", str(tmp_path))

        csv_bytes = self._make_csv()
        for name in ["ds_a", "ds_b"]:
            client.post(
                "/api/v1/datasets/upload",
                files={"file": (f"{name}.csv", io.BytesIO(csv_bytes), "text/csv")},
                data={"name": name},
            )

        resp = client.get("/api/v1/datasets")
        assert resp.json()["pagination"]["total"] == 2


# ══════════════════════════════════════════════════════════════════════════
# CONNECTORS
# ══════════════════════════════════════════════════════════════════════════

class TestConnectors:

    def test_list_connectors_empty(self, client):
        resp = client.get("/api/v1/connectors")
        assert resp.status_code == 200
        assert resp.json()["data"] == []

    def test_create_sql_connector(self, client):
        resp = client.post("/api/v1/connectors/sql", json={
            "name": "dev_postgres",
            "connector_type": "sql_postgres",
            "db_url": "postgresql://user:pass@localhost/devdb",
        })
        assert resp.status_code == 201
        body = resp.json()["data"]
        assert body["name"] == "dev_postgres"
        assert body["connector_type"] == "sql_postgres"
        # Sensitive field must not be returned
        assert "db_url" not in body
        assert "db_url_encrypted" not in body

    def test_create_oauth_connector(self, client):
        resp = client.post("/api/v1/connectors/oauth", json={
            "name": "github_api",
            "base_url": "https://api.github.com",
            "client_id": "gh_client_id",
            "client_secret": "gh_secret",
            "oauth2_flow": "authorization_code",
            "scopes": ["read:user", "repo"],
            "token_url": "https://github.com/login/oauth/access_token",
            "auth_url": "https://github.com/login/oauth/authorize",
            "token_body_format": "form",
            "response_format": "form",
            "client_auth": "body",
            "use_pkce": False,
        })
        assert resp.status_code == 201
        body = resp.json()["data"]
        assert body["name"] == "github_api"
        assert body["use_pkce"] is False
        # Secret must not appear in response
        assert "client_secret" not in body
        # token_url and auth_url must be stored and returned — this was the bug:
        # they were previously discarded and base_url was used as a fallback,
        # which would cause token requests to go to the wrong host.
        assert body["token_url"] == "https://github.com/login/oauth/access_token"
        assert body["auth_url"] == "https://github.com/login/oauth/authorize"

    def test_get_nonexistent_connector_returns_404(self, client):
        resp = client.get("/api/v1/connectors/9999")
        assert resp.status_code == 404

    def test_delete_connector(self, client):
        create = client.post("/api/v1/connectors/sql", json={
            "name": "to_delete",
            "connector_type": "sql_sqlite",
            "db_url": "sqlite:///tmp.db",
        })
        cid = create.json()["data"]["id"]

        del_resp = client.delete(f"/api/v1/connectors/{cid}")
        assert del_resp.status_code == 204

        get_resp = client.get(f"/api/v1/connectors/{cid}")
        assert get_resp.status_code == 404

    def test_test_connector_endpoint(self, client):
        create = client.post("/api/v1/connectors/sql", json={
            "name": "testable",
            "connector_type": "sql_sqlite",
            "db_url": "sqlite:///tmp.db",
        })
        cid = create.json()["data"]["id"]

        resp = client.post(f"/api/v1/connectors/{cid}/test")
        assert resp.status_code == 200
        assert "success" in resp.json()["data"]


# ══════════════════════════════════════════════════════════════════════════
# RESPONSE ENVELOPE
# ══════════════════════════════════════════════════════════════════════════

class TestResponseEnvelope:

    def test_single_resource_has_data_and_meta(self, client, tmp_path, monkeypatch):
        """Every single-resource response must have data + meta fields."""
        import config
        monkeypatch.setattr(config.settings, "dataset_storage_path", str(tmp_path))

        import io
        csv_bytes = b"a,b\n1,2\n3,4\n"
        resp = client.post(
            "/api/v1/datasets/upload",
            files={"file": ("env_test.csv", io.BytesIO(csv_bytes), "text/csv")},
        )
        body = resp.json()
        assert "data" in body
        assert "meta" in body
        assert "timestamp" in body["meta"]
        assert "version" in body["meta"]

    def test_list_response_has_pagination(self, client):
        """Every list response must have data + pagination + meta."""
        resp = client.get("/api/v1/datasets")
        body = resp.json()
        assert "data" in body
        assert "pagination" in body
        assert "total" in body["pagination"]
        assert "has_next" in body["pagination"]
        assert "meta" in body
