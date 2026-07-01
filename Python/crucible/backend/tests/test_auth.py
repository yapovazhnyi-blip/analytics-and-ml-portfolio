"""
Authentication tests.

Tests cover: registration, login, token validation, refresh, role enforcement,
timing-safe login (no email enumeration), and duplicate email rejection.

Uses the same in-memory SQLite + TestClient pattern as the rest of the test suite.
Auth is forcibly enabled for these tests via monkeypatching settings.disable_auth=False.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.pool import StaticPool

TEST_DB = "sqlite+aiosqlite:///:memory:"


@pytest.fixture
def auth_client(monkeypatch):
    """
    TestClient with in-memory DB and DISABLE_AUTH=False.
    Patching must happen BEFORE the app is imported so the lifespan
    startup sees the correct disable_auth value.
    """
    import sys, importlib, database
    import config as cfg

    # Enable auth for these tests
    monkeypatch.setattr(cfg.settings, "disable_auth", False)

    engine = create_async_engine(TEST_DB, connect_args={"check_same_thread": False}, poolclass=StaticPool)
    factory = async_sessionmaker(engine, expire_on_commit=False)
    database.engine = engine
    database.SessionFactory = factory
    database.AsyncSessionLocal = factory

    if "main" in sys.modules:
        importlib.reload(sys.modules["main"])
    import main as app_module
    import auth.dependencies as dep_mod
    monkeypatch.setattr(dep_mod.settings, "disable_auth", False)

    with TestClient(app_module.app, raise_server_exceptions=True) as c:
        yield c


def _register(client, email="admin@test.com", password="adminpass123"):
    return client.post("/api/v1/auth/register", json={"email": email, "password": password})


def _login(client, email="admin@test.com", password="adminpass123"):
    return client.post("/api/v1/auth/login", json={"email": email, "password": password})


# ══════════════════════════════════════════════════════════════════════════
# REGISTRATION
# ══════════════════════════════════════════════════════════════════════════

class TestRegistration:

    def test_first_user_gets_admin_role(self, auth_client):
        resp = _register(auth_client)
        assert resp.status_code == 201
        data = resp.json()["data"]
        assert data["role"] == "admin"
        assert "access_token" in data
        assert "refresh_token" in data

    def test_second_user_gets_contributor_role(self, auth_client):
        _register(auth_client)
        resp = _register(auth_client, email="second@test.com")
        assert resp.status_code == 201
        assert resp.json()["data"]["role"] == "contributor"

    def test_duplicate_email_rejected(self, auth_client):
        _register(auth_client)
        resp = _register(auth_client)   # same email
        assert resp.status_code == 409

    def test_short_password_rejected(self, auth_client):
        resp = _register(auth_client, password="abc")
        assert resp.status_code == 422

    def test_invalid_email_rejected(self, auth_client):
        resp = auth_client.post("/api/v1/auth/register", json={
            "email": "not-an-email", "password": "password123"
        })
        assert resp.status_code == 422


# ══════════════════════════════════════════════════════════════════════════
# LOGIN
# ══════════════════════════════════════════════════════════════════════════

class TestLogin:

    def test_valid_credentials_return_tokens(self, auth_client):
        _register(auth_client)
        resp = _login(auth_client)
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "access_token" in data
        assert data["role"] == "admin"

    def test_wrong_password_rejected(self, auth_client):
        _register(auth_client)
        resp = _login(auth_client, password="wrongpassword")
        assert resp.status_code == 401

    def test_unknown_email_returns_401(self, auth_client):
        resp = _login(auth_client, email="nobody@test.com")
        assert resp.status_code == 401

    def test_error_message_does_not_reveal_email_existence(self, auth_client):
        """Both wrong-email and wrong-password return the same error message."""
        _register(auth_client)
        r1 = _login(auth_client, email="doesnotexist@test.com", password="pass12345")
        r2 = _login(auth_client, password="wrongpassword")
        assert r1.json()["detail"] == r2.json()["detail"]


# ══════════════════════════════════════════════════════════════════════════
# TOKEN VALIDATION
# ══════════════════════════════════════════════════════════════════════════

class TestTokens:

    def _tokens(self, client):
        _register(client)
        return _login(client).json()["data"]

    def test_me_endpoint_requires_token(self, auth_client):
        resp = auth_client.get("/api/v1/auth/me")
        assert resp.status_code == 401

    def test_me_returns_user_with_valid_token(self, auth_client):
        tokens = self._tokens(auth_client)
        resp = auth_client.get("/api/v1/auth/me",
                               headers={"Authorization": f"Bearer {tokens['access_token']}"})
        assert resp.status_code == 200
        assert resp.json()["data"]["email"] == "admin@test.com"

    def test_invalid_token_rejected(self, auth_client):
        resp = auth_client.get("/api/v1/auth/me",
                               headers={"Authorization": "Bearer not.a.real.token"})
        assert resp.status_code == 401

    def test_refresh_token_issues_new_access_token(self, auth_client):
        tokens = self._tokens(auth_client)
        resp = auth_client.post("/api/v1/auth/refresh",
                                json={"refresh_token": tokens["refresh_token"]})
        assert resp.status_code == 200
        assert resp.json()["data"]["access_token"] != tokens["access_token"]

    def test_access_token_cannot_be_used_as_refresh_token(self, auth_client):
        tokens = self._tokens(auth_client)
        resp = auth_client.post("/api/v1/auth/refresh",
                                json={"refresh_token": tokens["access_token"]})
        assert resp.status_code == 401

    def test_protected_route_blocked_without_token(self, auth_client):
        resp = auth_client.get("/api/v1/datasets")
        assert resp.status_code == 401

    def test_protected_route_accessible_with_token(self, auth_client):
        tokens = self._tokens(auth_client)
        resp = auth_client.get("/api/v1/datasets",
                               headers={"Authorization": f"Bearer {tokens['access_token']}"})
        assert resp.status_code == 200


# ══════════════════════════════════════════════════════════════════════════
# ROLE-BASED ACCESS CONTROL
# ══════════════════════════════════════════════════════════════════════════

class TestRBAC:

    def test_admin_can_list_users(self, auth_client):
        _register(auth_client)
        tokens = _login(auth_client).json()["data"]
        resp = auth_client.get("/api/v1/auth/users",
                               headers={"Authorization": f"Bearer {tokens['access_token']}"})
        assert resp.status_code == 200

    def test_contributor_cannot_list_users(self, auth_client):
        _register(auth_client)                                   # creates admin
        _register(auth_client, email="contrib@test.com")        # creates contributor
        tokens = _login(auth_client, email="contrib@test.com").json()["data"]
        resp = auth_client.get("/api/v1/auth/users",
                               headers={"Authorization": f"Bearer {tokens['access_token']}"})
        assert resp.status_code == 403


# ══════════════════════════════════════════════════════════════════════════
# JWT UNIT TESTS (no HTTP — pure logic)
# ══════════════════════════════════════════════════════════════════════════

class TestJWT:

    def test_access_token_decodes_correctly(self):
        from auth.jwt import create_access_token, decode_access_token
        token = create_access_token(user_id=42, role="admin")
        payload = decode_access_token(token)
        assert payload is not None
        assert payload["sub"] == "42"
        assert payload["role"] == "admin"
        assert payload["type"] == "access"

    def test_refresh_token_decodes_correctly(self):
        from auth.jwt import create_refresh_token, decode_refresh_token
        token = create_refresh_token(user_id=7)
        assert decode_refresh_token(token) == 7

    def test_tampered_token_rejected(self):
        from auth.jwt import create_access_token, decode_access_token
        token = create_access_token(1, "admin")
        assert decode_access_token(token[:-5] + "XXXXX") is None

    def test_access_token_not_valid_as_refresh(self):
        from auth.jwt import create_access_token, decode_refresh_token
        assert decode_refresh_token(create_access_token(1, "admin")) is None

    def test_refresh_token_not_valid_as_access(self):
        from auth.jwt import create_refresh_token, decode_access_token
        assert decode_access_token(create_refresh_token(1)) is None

    def test_password_hash_and_verify(self):
        from auth.passwords import hash_password, verify_password
        hashed = hash_password("my-secret-password")
        assert hashed != "my-secret-password"
        assert verify_password("my-secret-password", hashed) is True
        assert verify_password("wrong-password", hashed) is False

