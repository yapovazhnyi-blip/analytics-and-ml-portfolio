"""
API key management router — /api/v1/auth/api-keys

Endpoints:
  GET    /auth/api-keys           — list stored providers (hints only, never full keys)
  PUT    /auth/api-keys/{provider} — store or update a key
  POST   /auth/api-keys/{provider}/validate — test a key with a live API call
  DELETE /auth/api-keys/{provider} — remove a stored key
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from auth.dependencies import get_current_user
from database import get_db
from models.user_api_key import UserAPIKey
from schemas.common import DataResponse
from storage.encryption import encrypt, decrypt
from llm.resolver import key_hint

router = APIRouter(
    prefix="/auth/api-keys",
    tags=["api-keys"],
    dependencies=[Depends(get_current_user)],
)

SUPPORTED_PROVIDERS = {"anthropic", "openai", "groq"}


# ── Schemas ───────────────────────────────────────────────────────────────────

class APIKeyIn(BaseModel):
    key: str = Field(
        ..., min_length=8,
        description="The API key to store. It will be encrypted immediately and never returned.",
    )
    label: Optional[str] = Field(
        None, max_length=128,
        description="Optional label to identify this key (e.g. 'personal account').",
    )


class APIKeyOut(BaseModel):
    provider: str
    key_hint: str        # e.g. "k2bx" — last 4 chars only
    label: Optional[str]
    validated: bool
    created_at: str
    last_used_at: Optional[str]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _key_out(k: UserAPIKey) -> dict:
    return {
        "provider":     k.provider,
        "key_hint":     f"...{k.key_hint}",
        "label":        k.label,
        "validated":    k.validated,
        "created_at":   k.created_at.isoformat(),
        "last_used_at": k.last_used_at.isoformat() if k.last_used_at else None,
    }


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.get("")
async def list_api_keys(
    user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Lists all API key providers the user has configured.
    Returns key hints only — the actual keys are never returned after storage.
    """
    rows = await db.scalars(
        select(UserAPIKey).where(UserAPIKey.user_id == user.id)
    )
    keys = rows.all()
    return DataResponse(data={"keys": [_key_out(k) for k in keys]})


@router.put("/{provider}", status_code=201)
async def upsert_api_key(
    provider: str,
    body: APIKeyIn,
    user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Stores or updates an API key for the given provider.

    The key is encrypted with Fernet immediately — only the last 4 characters
    are stored in plaintext as a display hint.

    Supported providers: anthropic, openai, groq.
    """
    if provider not in SUPPORTED_PROVIDERS:
        raise HTTPException(
            422,
            f"Unsupported provider '{provider}'. "
            f"Supported: {sorted(SUPPORTED_PROVIDERS)}",
        )

    plaintext   = body.key.strip()
    ciphertext  = encrypt(plaintext)
    hint        = key_hint(plaintext)

    # Upsert: update if exists, insert if not
    existing = await db.scalar(
        select(UserAPIKey).where(
            UserAPIKey.user_id == user.id,
            UserAPIKey.provider == provider,
        )
    )

    if existing:
        existing.encrypted_key = ciphertext
        existing.key_hint      = hint
        existing.label         = body.label
        existing.validated     = False   # reset validation after key change
        row = existing
    else:
        row = UserAPIKey(
            user_id=user.id,
            provider=provider,
            encrypted_key=ciphertext,
            key_hint=hint,
            label=body.label,
        )
        db.add(row)

    await db.flush()
    await db.refresh(row)
    return DataResponse(data=_key_out(row))


@router.post("/{provider}/validate")
async def validate_api_key(
    provider: str,
    user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Tests the stored API key by making a minimal live API call.

    For Anthropic: sends a single-token request to verify the key is valid
    and has API access. Updates the `validated` flag on the stored key.

    Returns: {"valid": true/false, "message": "..."}
    """
    if provider not in SUPPORTED_PROVIDERS:
        raise HTTPException(422, f"Unsupported provider: {provider}")

    stored = await db.scalar(
        select(UserAPIKey).where(
            UserAPIKey.user_id == user.id,
            UserAPIKey.provider == provider,
        )
    )
    if not stored:
        raise HTTPException(
            404,
            f"No {provider} key stored. Use PUT /auth/api-keys/{provider} first."
        )

    try:
        key = decrypt(stored.encrypted_key)
    except Exception:
        raise HTTPException(422, "Failed to decrypt stored key. Please re-enter it.")

    valid, message = await _test_key(provider, key)

    stored.validated = valid
    return DataResponse(data={"valid": valid, "message": message})


@router.delete("/{provider}", status_code=204)
async def delete_api_key(
    provider: str,
    user=Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """Removes the stored API key for the given provider."""
    stored = await db.scalar(
        select(UserAPIKey).where(
            UserAPIKey.user_id == user.id,
            UserAPIKey.provider == provider,
        )
    )
    if stored:
        await db.delete(stored)


# ── Live key validation ───────────────────────────────────────────────────────

async def _test_key(provider: str, key: str) -> tuple[bool, str]:
    """Makes a minimal live API call to verify the key works."""
    import httpx

    if provider == "anthropic":
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json={
                        "model":      "claude-haiku-4-5-20251001",
                        "max_tokens": 1,
                        "messages":   [{"role": "user", "content": "hi"}],
                    },
                )
            if resp.status_code == 200:
                return True, "Key is valid and has API access."
            if resp.status_code == 401:
                return False, "Key is invalid or has been revoked."
            if resp.status_code == 403:
                return False, "Key is valid but lacks API access (check your Anthropic account)."
            return False, f"Unexpected response: HTTP {resp.status_code}"
        except httpx.TimeoutException:
            return False, "Validation timed out — Anthropic API may be unreachable."
        except Exception as exc:
            return False, f"Validation failed: {exc}"

    if provider == "groq":
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                resp = await client.get(
                    "https://api.groq.com/openai/v1/models",
                    headers={"Authorization": f"Bearer {key}"},
                )
            return resp.status_code == 200, (
                "Groq key valid." if resp.status_code == 200
                else f"Groq key invalid (HTTP {resp.status_code})."
            )
        except Exception as exc:
            return False, f"Groq validation failed: {exc}"

    return False, f"Validation not implemented for provider '{provider}'."
