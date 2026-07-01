"""
API Key Manager — encrypted storage and per-request resolution of user API keys.

RESOLUTION ORDER
----------------
Every Claude call in Crucible resolves the API key via get_anthropic_key():

  1. User's own key (stored encrypted in DB)      ← preferred: user pays, user controls rate limits
  2. Server-level key (settings.anthropic_api_key) ← fallback: useful for demos / shared instances
  3. None → caller raises a clear 422             ← no silent failures

WHY BYOK MATTERS
-----------------
Without BYOK the platform operator pays for every Claude call across all users.
At $1/MTok for Haiku this adds up fast in demos or with multiple engineers using
the platform simultaneously. More importantly, enterprise users won't route their
data through a stranger's Anthropic account — they have their own contracts with
data processing agreements. BYOK is a hard requirement for any B2B deployment.

ENCRYPTION
----------
Keys are encrypted with Fernet (symmetric, AES-128-CBC + HMAC) before storage.
The Fernet key is derived from settings.secret_key via SHA-256. This means:
  - Changing secret_key rotates all stored keys (they become unreadable)
  - Exporting the DB without the application secret is useless to an attacker
  - The raw key is never logged or returned in API responses

Fernet is already a dependency (cryptography package) and already used for SQL
connector credentials — consistent pattern throughout the codebase.
"""

from __future__ import annotations

import base64
import hashlib
from typing import Optional

from fastapi import HTTPException


# ── Encryption helpers ────────────────────────────────────────────────────────

def _fernet():
    """Creates a Fernet instance derived from the application secret key."""
    from cryptography.fernet import Fernet
    from config import settings
    raw = settings.secret_key.encode("utf-8")
    key = base64.urlsafe_b64encode(hashlib.sha256(raw).digest())
    return Fernet(key)


def encrypt_key(plaintext: str) -> str:
    """Encrypts an API key for storage. Returns a URL-safe base64 string."""
    return _fernet().encrypt(plaintext.encode("utf-8")).decode("utf-8")


def decrypt_key(ciphertext: str) -> str:
    """Decrypts a stored API key. Raises ValueError on corrupt/invalid data."""
    try:
        return _fernet().decrypt(ciphertext.encode("utf-8")).decode("utf-8")
    except Exception as exc:
        raise ValueError(f"Failed to decrypt API key: {exc}") from exc


def mask_key(key: str) -> str:
    """Returns a masked version for display: sk-ant-...XY12"""
    if not key or len(key) < 8:
        return "****"
    return key[:8] + "..." + key[-4:]


# ── Key resolver ──────────────────────────────────────────────────────────────

async def get_anthropic_key(user=None, require: bool = True) -> Optional[str]:
    """
    Resolves the Anthropic API key for the current request.

    Resolution order:
      1. User's own stored key (decrypted from DB)
      2. Server-level key from settings
      3. None (raises HTTP 422 if require=True)

    Args:
        user:    The authenticated User ORM object (may be None if auth disabled).
        require: If True, raises HTTP 422 when no key is found. Set to False for
                 endpoints that are optional (e.g. advisor can run without Claude).

    Returns:
        The plaintext API key, or None if require=False and no key is found.
    """
    from config import settings

    # ── 1. User's own key ─────────────────────────────────────────────────────
    if user is not None:
        encrypted = getattr(user, "anthropic_key_encrypted", None)
        if encrypted:
            try:
                return decrypt_key(encrypted)
            except ValueError:
                # Corrupted key — don't silently fail, fall through to server key
                pass

    # ── 2. Server-level key ───────────────────────────────────────────────────
    server_key = getattr(settings, "anthropic_api_key", "") or ""
    if server_key:
        return server_key

    # ── 3. No key available ───────────────────────────────────────────────────
    if require:
        raise HTTPException(
            status_code=422,
            detail=(
                "No Anthropic API key configured. "
                "Add your key via PUT /api/v1/auth/api-keys, "
                "or ask the platform admin to set ANTHROPIC_API_KEY."
            ),
        )
    return None


def get_anthropic_key_sync(user=None) -> Optional[str]:
    """Synchronous version for use outside async context (e.g. background tasks)."""
    from config import settings

    if user is not None:
        encrypted = getattr(user, "anthropic_key_encrypted", None)
        if encrypted:
            try:
                return decrypt_key(encrypted)
            except ValueError:
                pass

    return getattr(settings, "anthropic_api_key", "") or None
