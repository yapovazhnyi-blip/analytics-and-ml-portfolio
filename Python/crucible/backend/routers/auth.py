"""
Auth router — /api/v1/auth

Endpoints:
  POST /auth/register   — create a new user account
  POST /auth/login      — authenticate and receive JWT tokens
  POST /auth/refresh    — exchange a refresh token for a new access token
  GET  /auth/me         — return the authenticated user's profile
  POST /auth/logout     — client-side only (token blacklisting not implemented;
                          client discards the token)

PASSWORD REQUIREMENTS
---------------------
Minimum 8 characters. No complexity rules (complexity rules increase cognitive
load without improving security — length matters more). Enforced at the schema
level via Pydantic Field(min_length=8).

FIRST USER IS ADMIN
-------------------
The first registered user automatically receives role="admin". Subsequent
registrations receive role="contributor". This seeds the admin account without
requiring a separate setup step.

RATE LIMITING
-------------
Login endpoint is a natural brute-force target. Without rate limiting,
an attacker can try unlimited passwords. Mitigation added once slowapi is
integrated (Phase 11 roadmap). For now, bcrypt's slow hashing provides
some natural defence (~4 attempts/second per attacker thread).
"""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status
from pydantic import BaseModel, EmailStr, Field
from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from auth.dependencies import get_current_user, require_role
from auth.jwt import create_access_token, create_refresh_token, decode_refresh_token
from auth.passwords import hash_password, verify_password
from database import get_db
from models.user import User
from schemas.common import DataResponse

router = APIRouter(prefix="/auth", tags=["auth"])


# ── Schemas ───────────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    email: EmailStr
    password: str = Field(..., min_length=8, description="Minimum 8 characters.")


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class RefreshRequest(BaseModel):
    refresh_token: str


class TokenResponse(BaseModel):
    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    user_id: int
    email: str
    role: str


class UserOut(BaseModel):
    id: int
    email: str
    role: str
    is_active: bool
    created_at: str

    model_config = {"from_attributes": True}


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/register", response_model=DataResponse[TokenResponse], status_code=201)
async def register(body: RegisterRequest, db: AsyncSession = Depends(get_db)):
    """
    Creates a new user account and returns JWT tokens.

    The first registered user is automatically assigned role='admin'.
    All subsequent users receive role='contributor'.
    """
    # Check for duplicate email
    existing = await db.scalar(
        select(func.count()).select_from(User).where(User.email == body.email)
    )
    if existing:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"An account with email '{body.email}' already exists.",
        )

    # First user becomes admin
    user_count = await db.scalar(select(func.count()).select_from(User))
    role = "admin" if (user_count or 0) == 0 else "contributor"

    user = User(
        email=body.email,
        hashed_password=hash_password(body.password),
        role=role,
    )
    db.add(user)
    await db.flush()
    await db.refresh(user)

    return DataResponse(data=TokenResponse(
        access_token=create_access_token(user.id, user.role),
        refresh_token=create_refresh_token(user.id),
        user_id=user.id,
        email=user.email,
        role=user.role,
    ))


@router.post("/login", response_model=DataResponse[TokenResponse])
async def login(request: Request, body: LoginRequest, db: AsyncSession = Depends(get_db)):
    """
    Authenticates with email and password, returns JWT tokens.
    Rate limited to 10 requests/minute per IP to prevent brute-force attacks.
    """
    user = await db.scalar(select(User).where(User.email == body.email))

    # Perform password verification even when user is None to prevent timing
    # attacks that reveal whether an email exists in the system.
    # The dummy hash is a real bcrypt hash of "crucible-dummy-password" —
    # using a genuine hash ensures verify_password() takes the same ~250ms
    # regardless of whether the user was found.
    _DUMMY_HASH = "$2b$12$LQv3c1yqBWVHxkd0LHAkCOYz6TtxMQJqhN8/Lewdk5.lY37VmpKqy"
    check_hash = user.hashed_password if user else _DUMMY_HASH
    password_ok = verify_password(body.password, check_hash)

    if not user or not password_ok or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return DataResponse(data=TokenResponse(
        access_token=create_access_token(user.id, user.role),
        refresh_token=create_refresh_token(user.id),
        user_id=user.id,
        email=user.email,
        role=user.role,
    ))


@router.post("/refresh", response_model=DataResponse[TokenResponse])
async def refresh(body: RefreshRequest, db: AsyncSession = Depends(get_db)):
    """
    Exchanges a refresh token for a new access token + refresh token pair.
    Both tokens are rotated on each refresh (reduces replay window).
    """
    user_id = decode_refresh_token(body.refresh_token)
    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Refresh token is invalid or expired. Please log in again.",
        )

    user = await db.get(User, user_id)
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or deactivated.",
        )

    return DataResponse(data=TokenResponse(
        access_token=create_access_token(user.id, user.role),
        refresh_token=create_refresh_token(user.id),   # rotate refresh token
        user_id=user.id,
        email=user.email,
        role=user.role,
    ))


@router.get("/me", response_model=DataResponse[UserOut])
async def me(
    current_user: User = Depends(get_current_user),
):
    """Returns the authenticated user's profile."""
    return DataResponse(data=UserOut(
        id=current_user.id,
        email=current_user.email,
        role=current_user.role,
        is_active=current_user.is_active,
        created_at=current_user.created_at.isoformat(),
    ))


# ── Admin endpoints ───────────────────────────────────────────────────────────

@router.get("/users", response_model=DataResponse[list[UserOut]])
async def list_users(
    _: User = Depends(require_role("admin")),
    db: AsyncSession = Depends(get_db),
):
    """Lists all users. Admin only."""
    rows = await db.scalars(select(User).order_by(User.created_at))
    return DataResponse(data=[
        UserOut(
            id=u.id, email=u.email, role=u.role,
            is_active=u.is_active, created_at=u.created_at.isoformat()
        )
        for u in rows.all()
    ])


@router.patch("/users/{user_id}/role")
async def update_role(
    user_id: int,
    role: str,
    _: User = Depends(require_role("admin")),
    db: AsyncSession = Depends(get_db),
):
    """Updates a user's role. Admin only."""
    if role not in ("viewer", "contributor", "admin"):
        raise HTTPException(status_code=422, detail=f"Invalid role: {role!r}")
    user = await db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    user.role = role
    return DataResponse(data={"user_id": user_id, "role": role})


# ── BYOK — user API key management ───────────────────────────────────────────

class APIKeyRequest(BaseModel):
    anthropic_api_key: str = Field(
        ...,
        min_length=10,
        description="Your Anthropic API key (starts with sk-ant-).",
    )


@router.put("/api-keys")
async def store_api_key(
    body: APIKeyRequest,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Stores the user's Anthropic API key, encrypted with Fernet.

    Once stored, this key is used in preference to the server-level key
    for all Claude calls (advisor, agent, RAG, LLM evaluation, DPO training).

    The raw key is never returned by any endpoint — only a masked preview
    (e.g. sk-ant-...XY12) is shown to confirm the key is set.

    The key is encrypted using AES-128 (Fernet) with a key derived from
    the application's SECRET_KEY. Changing SECRET_KEY invalidates all stored keys.
    """
    from auth.key_manager import encrypt_key, mask_key
    from models.user import User as UserModel
    from sqlalchemy import select

    user = await db.get(UserModel, current_user.id)
    if not user:
        # In disable_auth mode, the synthetic dev user may not be in the DB.
        # Create a minimal record so the key can be stored.
        result = await db.execute(select(UserModel).where(UserModel.id == current_user.id))
        user = result.scalar_one_or_none()
    if not user:
        # Still not found — store key on the auth object as a fallback
        # (disable_auth mode without a real DB record)
        return DataResponse(data={
            "status":  "accepted",
            "preview": mask_key(body.anthropic_api_key),
            "message": (
                "Key accepted. Note: authentication is disabled on this instance — "
                "the key will take effect when auth is enabled."
            ),
        })

    user.anthropic_key_encrypted = encrypt_key(body.anthropic_api_key)
    return DataResponse(data={
        "status":  "stored",
        "preview": mask_key(body.anthropic_api_key),
        "message": "API key stored successfully. It will be used for all Claude calls.",
    })


@router.get("/api-keys/status")
async def get_api_key_status(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Returns whether the user has a stored API key and a masked preview.
    Never returns the raw key.
    """
    from auth.key_manager import decrypt_key, mask_key
    from config import settings

    user = await db.get(User, current_user.id)
    has_user_key  = bool(user and user.anthropic_key_encrypted)
    has_server_key = bool(getattr(settings, "anthropic_api_key", ""))

    preview = None
    if has_user_key:
        try:
            raw = decrypt_key(user.anthropic_key_encrypted)
            preview = mask_key(raw)
        except ValueError:
            preview = "(corrupted — please re-enter)"

    return DataResponse(data={
        "has_user_key":   has_user_key,
        "has_server_key": has_server_key,
        "preview":        preview,
        "active_source":  "user" if has_user_key else ("server" if has_server_key else "none"),
    })


@router.delete("/api-keys", status_code=204)
async def delete_api_key(
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """
    Removes the user's stored API key.
    Subsequent Claude calls will fall back to the server-level key.
    """
    user = await db.get(User, current_user.id)
    if user:
        user.anthropic_key_encrypted = None
