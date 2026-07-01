"""
LLM key resolver — the single source of truth for which API key to use.

RESOLUTION ORDER
-----------------
1. User's stored key for the requested provider (decrypted from DB)
2. Server-side key from settings (ANTHROPIC_API_KEY env var)
3. Empty string (callers should detect this and return a clear error)

WHY A SINGLE RESOLVER
----------------------
Before BYOK, six different callers each read `settings.anthropic_api_key`
directly. Adding BYOK to each caller individually would mean six changes
and six places where the logic could diverge.

The resolver is the single authoritative function. Every caller gets the
right key by calling `await resolve_api_key(user, db)` — the resolution
logic lives in exactly one place.

PROVIDER SUPPORT
-----------------
Currently "anthropic" only. When Bedrock or Groq support is added, the
provider parameter controls which stored key is retrieved. The caller
(e.g. the agent runner) knows which provider it's calling and passes that
string — the resolver handles the DB lookup transparently.

DEV MODE
---------
When `settings.disable_auth=True`, all requests use a synthetic dev user
with no real user_id. The resolver skips the DB lookup and returns the
server key directly to avoid a meaningless DB query.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from config import settings
from storage.encryption import decrypt


async def resolve_api_key(
    user,                         # User ORM object or None (dev mode)
    db: AsyncSession,
    provider: str = "anthropic",
) -> str:
    """
    Returns the best available API key for the given user and provider.

    Args:
        user:     The authenticated User ORM object. May be None in dev mode
                  or when auth is disabled.
        db:       Active async DB session for the key lookup.
        provider: Provider identifier ("anthropic", "openai", "groq").

    Returns:
        The API key string, or "" if no key is configured anywhere.
        Callers should check for empty string and raise a clear HTTP error.
    """
    # ── 1. Try user's own key ─────────────────────────────────────────────
    if user and getattr(user, "id", None):
        from models.user_api_key import UserAPIKey
        stored = await db.scalar(
            select(UserAPIKey).where(
                UserAPIKey.user_id == user.id,
                UserAPIKey.provider == provider,
            )
        )
        if stored and stored.encrypted_key:
            try:
                key = decrypt(stored.encrypted_key)
                if key:
                    # Update last_used_at asynchronously (best-effort)
                    stored.last_used_at = datetime.now(timezone.utc)
                    return key
            except Exception:
                # Decryption failure (e.g. key rotation) — fall through
                pass

    # ── 2. Fall back to server key ────────────────────────────────────────
    server_key = getattr(settings, f"{provider}_api_key", None) or \
                 getattr(settings, "anthropic_api_key", None) or ""
    return server_key


def key_hint(plaintext: str) -> str:
    """Returns the last 4 characters of a key for display purposes."""
    return plaintext[-4:] if len(plaintext) >= 4 else plaintext
