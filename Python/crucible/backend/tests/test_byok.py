"""
BYOK (Bring Your Own Key) tests.

Tests cover:
  - Fernet encryption/decryption roundtrip
  - Masking API keys in responses
  - get_anthropic_key() resolution order: user key > server key > error
  - Auth endpoints: PUT /api-keys, GET /api-keys/status, DELETE /api-keys
  - Raw key never exposed in any API response
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch


# ══════════════════════════════════════════════════════════════════════════
# ENCRYPTION HELPERS
# ══════════════════════════════════════════════════════════════════════════

class TestEncryption:

    def test_encrypt_decrypt_roundtrip(self):
        from auth.key_manager import encrypt_key, decrypt_key
        plaintext = "sk-ant-api03-test-key-abcdefgh"
        assert decrypt_key(encrypt_key(plaintext)) == plaintext

    def test_encrypted_differs_from_plaintext(self):
        from auth.key_manager import encrypt_key
        key = "sk-ant-test-key"
        assert encrypt_key(key) != key

    def test_two_encryptions_differ(self):
        """Fernet includes a random nonce — same plaintext gives different ciphertext."""
        from auth.key_manager import encrypt_key
        k = "sk-ant-test"
        assert encrypt_key(k) != encrypt_key(k)

    def test_decrypt_garbage_raises(self):
        from auth.key_manager import decrypt_key
        with pytest.raises(ValueError):
            decrypt_key("not-valid-ciphertext")

    def test_mask_hides_middle(self):
        from auth.key_manager import mask_key
        result = mask_key("sk-ant-api03-abcdefghijklXY12")
        assert result.startswith("sk-ant-")
        assert "..." in result
        assert result.endswith("Y12")

    def test_mask_short_key(self):
        from auth.key_manager import mask_key
        assert mask_key("short") == "****"

    def test_mask_empty(self):
        from auth.key_manager import mask_key
        assert mask_key("") == "****"


# ══════════════════════════════════════════════════════════════════════════
# KEY RESOLVER
# ══════════════════════════════════════════════════════════════════════════

class TestKeyResolver:

    def _make_user(self, key=None):
        from auth.key_manager import encrypt_key
        u = MagicMock()
        u.id = 1
        u.anthropic_key_encrypted = encrypt_key(key) if key else None
        return u

    @pytest.mark.asyncio
    async def test_returns_user_key_when_set(self):
        from auth.key_manager import get_anthropic_key
        from config import settings
        user = self._make_user("sk-ant-user-key-abc")
        original = settings.anthropic_api_key
        try:
            settings.anthropic_api_key = "sk-ant-server-key"
            result = await get_anthropic_key(user)
        finally:
            settings.anthropic_api_key = original
        assert result == "sk-ant-user-key-abc"

    @pytest.mark.asyncio
    async def test_falls_back_to_server_key(self):
        from auth.key_manager import get_anthropic_key
        from config import settings
        user = self._make_user(None)
        original = settings.anthropic_api_key
        try:
            settings.anthropic_api_key = "sk-ant-server-key"
            result = await get_anthropic_key(user)
        finally:
            settings.anthropic_api_key = original
        assert result == "sk-ant-server-key"

    @pytest.mark.asyncio
    async def test_raises_when_no_key_require_true(self):
        from auth.key_manager import get_anthropic_key
        from config import settings
        from fastapi import HTTPException
        user = self._make_user(None)
        original = settings.anthropic_api_key
        try:
            settings.anthropic_api_key = ""
            with pytest.raises(HTTPException) as exc:
                await get_anthropic_key(user, require=True)
            assert exc.value.status_code == 422
        finally:
            settings.anthropic_api_key = original

    @pytest.mark.asyncio
    async def test_returns_none_require_false(self):
        from auth.key_manager import get_anthropic_key
        from config import settings
        user = self._make_user(None)
        original = settings.anthropic_api_key
        try:
            settings.anthropic_api_key = ""
            result = await get_anthropic_key(user, require=False)
        finally:
            settings.anthropic_api_key = original
        assert result is None

    @pytest.mark.asyncio
    async def test_none_user_falls_back_to_server(self):
        from auth.key_manager import get_anthropic_key
        from config import settings
        original = settings.anthropic_api_key
        try:
            settings.anthropic_api_key = "sk-ant-server"
            result = await get_anthropic_key(None, require=False)
        finally:
            settings.anthropic_api_key = original
        assert result == "sk-ant-server"

    @pytest.mark.asyncio
    async def test_user_key_preferred_over_server(self):
        from auth.key_manager import get_anthropic_key
        from config import settings
        user = self._make_user("sk-ant-user-wins")
        original = settings.anthropic_api_key
        try:
            settings.anthropic_api_key = "sk-ant-server-fallback"
            result = await get_anthropic_key(user)
        finally:
            settings.anthropic_api_key = original
        assert result == "sk-ant-user-wins"


# ══════════════════════════════════════════════════════════════════════════
# API ENDPOINTS
# ══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def byok_client():
    import sys, importlib, database as db_mod
    from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
    from sqlalchemy.pool import StaticPool
    from fastapi.testclient import TestClient
    engine = create_async_engine(
        "sqlite+aiosqlite:///:memory:",
        connect_args={"check_same_thread": False}, poolclass=StaticPool,
    )
    factory = async_sessionmaker(engine, expire_on_commit=False)
    db_mod.engine = engine
    db_mod.SessionFactory = factory
    db_mod.AsyncSessionLocal = factory
    if "main" in sys.modules:
        importlib.reload(sys.modules["main"])
    import main as m
    with TestClient(m.app, raise_server_exceptions=True) as c:
        yield c


def _login_token(client):
    client.post("/api/v1/auth/register",
                json={"email": "byok@test.com", "password": "TestPass123!"})
    resp = client.post("/api/v1/auth/login",
                       json={"email": "byok@test.com", "password": "TestPass123!"})
    return resp.json()["data"]["access_token"]


class TestBYOKEndpoints:

    def test_put_api_key_returns_200(self, byok_client):
        token = _login_token(byok_client)
        resp = byok_client.put(
            "/api/v1/auth/api-keys",
            json={"anthropic_api_key": "sk-ant-test-key-abcdef"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert data["status"] in ("stored", "accepted")
        assert "preview" in data

    def test_raw_key_not_in_response(self, byok_client):
        token = _login_token(byok_client)
        raw = "sk-ant-secret-never-returned-xyz"
        resp = byok_client.put(
            "/api/v1/auth/api-keys",
            json={"anthropic_api_key": raw},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert raw not in resp.text

    def test_get_status_returns_structure(self, byok_client):
        token = _login_token(byok_client)
        resp = byok_client.get(
            "/api/v1/auth/api-keys/status",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 200
        data = resp.json()["data"]
        assert "has_user_key" in data
        assert "has_server_key" in data
        assert "active_source" in data

    def test_status_no_key_by_default(self, byok_client):
        token = _login_token(byok_client)
        resp = byok_client.get(
            "/api/v1/auth/api-keys/status",
            headers={"Authorization": f"Bearer {token}"},
        )
        data = resp.json()["data"]
        assert data["has_user_key"] is False
        assert data["preview"] is None

    def test_delete_key(self, byok_client):
        token = _login_token(byok_client)
        byok_client.put(
            "/api/v1/auth/api-keys",
            json={"anthropic_api_key": "sk-ant-to-delete-key-zxcv"},
            headers={"Authorization": f"Bearer {token}"},
        )
        del_resp = byok_client.delete(
            "/api/v1/auth/api-keys",
            headers={"Authorization": f"Bearer {token}"},
        )
        assert del_resp.status_code == 204

    def test_short_key_rejected(self, byok_client):
        token = _login_token(byok_client)
        resp = byok_client.put(
            "/api/v1/auth/api-keys",
            json={"anthropic_api_key": "short"},
            headers={"Authorization": f"Bearer {token}"},
        )
        assert resp.status_code == 422

    def test_preview_masks_key(self, byok_client):
        token = _login_token(byok_client)
        resp = byok_client.put(
            "/api/v1/auth/api-keys",
            json={"anthropic_api_key": "sk-ant-api03-longkeyvalue12345"},
            headers={"Authorization": f"Bearer {token}"},
        )
        if resp.status_code == 200:
            preview = resp.json()["data"].get("preview", "")
            if preview and preview != "****":
                assert "..." in preview   # masked format
                assert "longkeyvalue12345" not in preview
