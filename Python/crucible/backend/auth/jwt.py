"""
JWT token operations — create, verify, refresh.

TOKEN DESIGN
------------
Access token (24h): carries user_id, role, and expiry. Short-lived so a
stolen token has a limited window. Used in every API request header.

Refresh token (30d): carries only user_id and a "refresh" type flag.
Used once to obtain a new access token without re-entering credentials.
Longer-lived because it is only sent to the /auth/refresh endpoint, not
every request. Store in an httpOnly cookie in production.

WHY HS256
---------
HS256 (HMAC-SHA256) uses a single shared secret. Simple, fast, and
appropriate when the same service signs and verifies tokens. RS256
(RSA) would be needed in multi-service architectures where a separate
auth service signs and each microservice verifies independently.

PAYLOAD FIELDS
--------------
sub  — subject (user id as string, JWT standard)
role — user role (avoids a DB lookup on every request)
type — "access" | "refresh" (prevents refresh tokens being used as access tokens)
exp  — expiry timestamp (JWT standard, handled by python-jose automatically)
iat  — issued-at timestamp (JWT standard)
"""

from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional

from jose import jwt, JWTError

from config import settings


_ALGORITHM = "HS256"


def create_access_token(user_id: int, role: str) -> str:
    """
    Creates a signed JWT access token valid for ACCESS_TOKEN_EXPIRE_MINUTES.

    Args:
        user_id: Database primary key of the authenticated user.
        role:    User's role string ("viewer" | "contributor" | "admin").

    Returns:
        Signed JWT string ready to be returned as Bearer token.
    """
    now = datetime.now(timezone.utc)
    payload = {
        "sub":  str(user_id),
        "role": role,
        "type": "access",
        "jti":  uuid.uuid4().hex,
        "iat":  now,
        "exp":  now + timedelta(minutes=settings.access_token_expire_minutes),
    }
    return jwt.encode(payload, settings.secret_key, algorithm=_ALGORITHM)


def create_refresh_token(user_id: int) -> str:
    """
    Creates a long-lived refresh token.
    Contains only user_id and type — does NOT carry role (avoids stale role data).
    """
    now = datetime.now(timezone.utc)
    payload = {
        "sub":  str(user_id),
        "type": "refresh",
        "jti":  uuid.uuid4().hex,
        "iat":  now,
        "exp":  now + timedelta(days=settings.refresh_token_expire_days),
    }
    return jwt.encode(payload, settings.secret_key, algorithm=_ALGORITHM)


def decode_access_token(token: str) -> Optional[dict]:
    """
    Decodes and validates an access token.

    Returns the payload dict on success, None on any failure
    (expired, invalid signature, wrong type, malformed).

    Callers should treat None as "unauthenticated" and return 401.
    """
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[_ALGORITHM])
        if payload.get("type") != "access":
            return None
        return payload
    except JWTError:
        return None


def decode_refresh_token(token: str) -> Optional[int]:
    """
    Decodes a refresh token and returns the user_id (int) on success,
    or None if the token is invalid or expired.
    """
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[_ALGORITHM])
        if payload.get("type") != "refresh":
            return None
        return int(payload["sub"])
    except (JWTError, KeyError, ValueError):
        return None
