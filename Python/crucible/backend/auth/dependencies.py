"""
FastAPI authentication dependencies.

USAGE IN ROUTES
---------------
# Protect an entire router (all endpoints require login):
router = APIRouter(dependencies=[Depends(get_current_user)])

# Protect one endpoint:
@router.get("/admin-only")
async def admin_endpoint(user: User = Depends(require_role("admin"))):
    ...

# Get the current user inside a handler:
@router.get("/me")
async def me(user: User = Depends(get_current_user)):
    return {"email": user.email}

DEV MODE (DISABLE_AUTH=true)
-----------------------------
When settings.disable_auth is True, get_current_user returns a synthetic
User object without touching the database or requiring a token. This makes
local development frictionless while the real auth infrastructure is in place.

The synthetic user has role="admin" so all endpoints are accessible.
Switch disable_auth to False before any public deployment.

WEBSOCKET AUTH
--------------
WebSocket connections cannot send Authorization headers from browsers.
Use the token query parameter instead and call validate_ws_token():

    @router.websocket("/ws/{id}")
    async def ws(websocket: WebSocket, token: str = Query(None)):
        user = await validate_ws_token(token, db)
        if not user:
            await websocket.close(code=1008)  # 1008 = Policy Violation
            return
        await websocket.accept()
        ...
"""

from __future__ import annotations

from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy.ext.asyncio import AsyncSession

from auth.jwt import decode_access_token, decode_refresh_token
from config import settings
from database import get_db
from models.user import User

# HTTPBearer extracts the token from "Authorization: Bearer <token>" header.
# auto_error=False so we can return a clean 401 instead of FastAPI's default 403.
_bearer = HTTPBearer(auto_error=False)

# Synthetic user returned in dev mode (DISABLE_AUTH=true)
_DEV_USER = User(id=0, email="dev@crucible.local", role="admin", is_active=True)


async def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(_bearer),
    db: AsyncSession = Depends(get_db),
) -> User:
    """
    Extracts and validates the JWT from the Authorization header.
    Returns the authenticated User object.
    Raises HTTP 401 if the token is missing, expired, or invalid.
    """
    if settings.disable_auth:
        return _DEV_USER

    if not credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authorization header missing. Use: Authorization: Bearer <token>",
            headers={"WWW-Authenticate": "Bearer"},
        )

    payload = decode_access_token(credentials.credentials)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token is invalid or expired.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    user_id = int(payload["sub"])
    user = await db.get(User, user_id)
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or deactivated.",
        )

    return user


def require_role(*roles: str):
    """
    Dependency factory that requires the current user to have one of the given roles.

    Usage:
        @router.delete("/{id}", dependencies=[Depends(require_role("admin"))])
        async def delete_something(...): ...
    """
    async def _check(user: User = Depends(get_current_user)) -> User:
        if settings.disable_auth:
            return user
        if user.role not in roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"This action requires one of these roles: {list(roles)}. "
                       f"Your role is: {user.role!r}",
            )
        return user
    return _check


async def validate_ws_token(
    token: Optional[str],
    db: AsyncSession,
) -> Optional[User]:
    """
    Validates a JWT token passed as a WebSocket query parameter.
    Returns the User on success, None on failure (caller should close the socket).

    In dev mode (DISABLE_AUTH=true) always returns the synthetic dev user.
    """
    if settings.disable_auth:
        return _DEV_USER

    if not token:
        return None

    payload = decode_access_token(token)
    if not payload:
        return None

    user = await db.get(User, int(payload["sub"]))
    return user if (user and user.is_active) else None
