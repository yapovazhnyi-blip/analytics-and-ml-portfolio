"""
OAuth2 callback router for Crucible.

Handles the two endpoints needed to complete the authorization-code flow:

  GET /connectors/{id}/authorize
      Generates the provider's authorization URL (with state + optional PKCE)
      and redirects the user to it. The state is stored server-side in a
      short-lived in-memory dict so we can validate it on callback.

  GET /oauth/callback
      Receives the authorization code from the provider, exchanges it for
      tokens, encrypts the tokens, and persists them against the Connector
      record. Returns a simple success page the user can close.

State management:
  In a production multi-instance deployment, state would live in Redis.
  For Phase 1 (single-process), an in-memory dict is correct and simple.
  The state entries expire after 10 minutes to prevent accumulation.

Token storage:
  access_token and refresh_token are encrypted with Fernet before
  being written to the database. The token_expires_at field stores
  the unix timestamp of expiry so the factory can detect expired tokens
  and trigger a refresh before use.
"""

from __future__ import annotations

import time
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import HTMLResponse
from sqlalchemy.ext.asyncio import AsyncSession

from database import get_db
from models.connector import Connector
from connectors.auth.oauth2 import AuthCodeConnector, OAuth2Config
from connectors.factory import ConnectorFactory
from storage.encryption import encrypt

import httpx

router = APIRouter(tags=["oauth_callback"])

# ── Short-lived state store — (state_token → connector_id) ────────────────
# Entries expire after OAUTH_STATE_TTL_SECS to prevent accumulation.
_STATE_STORE: dict[str, tuple[int, float]] = {}  # state → (connector_id, created_at)
OAUTH_STATE_TTL_SECS = 600  # 10 minutes


def _store_state(state: str, connector_id: int) -> None:
    _STATE_STORE[state] = (connector_id, time.time())
    # Expire old entries
    cutoff = time.time() - OAUTH_STATE_TTL_SECS
    expired = [k for k, (_, t) in _STATE_STORE.items() if t < cutoff]
    for k in expired:
        del _STATE_STORE[k]


def _pop_state(state: str) -> Optional[int]:
    """Returns the connector_id for a valid state, or None if expired/invalid."""
    entry = _STATE_STORE.pop(state, None)
    if entry is None:
        return None
    connector_id, created_at = entry
    if time.time() - created_at > OAUTH_STATE_TTL_SECS:
        return None
    return connector_id


# ── Authorize endpoint ────────────────────────────────────────────────────

@router.get("/connectors/{connector_id}/authorize")
async def authorize_connector(
    connector_id: int,
    redirect_uri: Optional[str] = Query(
        default=None,
        description="Override the default callback URL (useful for local dev)",
    ),
    db: AsyncSession = Depends(get_db),
):
    """
    Step 1 of the authorization-code flow.

    Generates the provider's auth URL (including state, PKCE challenge if
    configured, and the requested scopes), stores the state token server-side,
    and redirects the user to the provider's login/consent page.

    After the user authorises the app, the provider redirects back to
    GET /oauth/callback?code=...&state=...
    """
    connector = await db.get(Connector, connector_id)
    if not connector:
        raise HTTPException(status_code=404, detail=f"Connector {connector_id} not found")
    if connector.oauth2_flow != "authorization_code":
        raise HTTPException(
            status_code=422,
            detail="This connector does not use the authorization_code flow"
        )
    if not connector.client_id or not connector.client_secret_encrypted:
        raise HTTPException(
            status_code=422,
            detail="Connector is missing client_id or client_secret"
        )

    # Validate redirect_uri against an allowlist to prevent open redirect attacks.
    # A malicious actor could craft a URL like:
    #   /connectors/1/authorize?redirect_uri=https://evil.com/steal?code=
    # which would cause the OAuth provider to send the auth code to evil.com.
    DEFAULT_CALLBACK = "http://localhost:8000/oauth/callback"
    ALLOWED_CALLBACK_HOSTS = {"localhost", "127.0.0.1"}

    if redirect_uri:
        from urllib.parse import urlparse
        parsed = urlparse(redirect_uri)
        if parsed.hostname not in ALLOWED_CALLBACK_HOSTS:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"redirect_uri host '{parsed.hostname}' is not in the allowlist. "
                    f"Allowed hosts: {sorted(ALLOWED_CALLBACK_HOSTS)}"
                ),
            )
    effective_redirect_uri = redirect_uri or DEFAULT_CALLBACK

    from storage.encryption import decrypt
    client_secret = decrypt(connector.client_secret_encrypted) or ""

    # Use the explicitly stored token_url and auth_url — the old code used
    # base_url as a fallback for both, which would construct a broken
    # OAuth2Config (GitHub's auth URL is github.com/login/oauth/authorize,
    # not api.github.com which is the base_url for API calls).
    oauth_cfg = OAuth2Config(
        client_id=connector.client_id,
        client_secret=client_secret,
        token_url=connector.token_url or "",
        auth_url=connector.auth_url or "",
        scopes=[],
        token_body_format=connector.token_body_format or "json",
        response_format=connector.response_format or "json",
        client_auth=connector.client_auth or "body",
        use_pkce=bool(connector.use_pkce),
        redirect_uri=effective_redirect_uri,
    )

    http_client = httpx.AsyncClient(timeout=30.0)
    auth_connector = AuthCodeConnector(oauth_cfg, http_client)
    auth_url, state = auth_connector.build_auth_url()

    # Store state + the connector's code_verifier (if PKCE) for the callback
    _store_state(state, connector_id)

    # Also cache the auth_connector instance so the callback can use the
    # same code_verifier (PKCE requires the verifier that was used to build
    # the challenge)
    _STATE_STORE[f"connector_{state}"] = (auth_connector, time.time())

    from fastapi.responses import RedirectResponse
    return RedirectResponse(url=auth_url)


# ── Callback endpoint ─────────────────────────────────────────────────────

@router.get("/oauth/callback")
async def oauth_callback(
    code: str = Query(..., description="Authorization code from the provider"),
    state: str = Query(..., description="State token for CSRF validation"),
    error: Optional[str] = Query(default=None),
    db: AsyncSession = Depends(get_db),
):
    """
    Step 2 of the authorization-code flow.

    Receives the authorization code from the provider, validates the state
    token (CSRF protection), exchanges the code for access + refresh tokens,
    encrypts them, and persists them against the Connector record.

    Returns a simple HTML page the user can close — in a real app this
    would redirect to the frontend's connector management page.
    """
    if error:
        return HTMLResponse(
            content=_result_html(
                success=False,
                message=f"Provider returned an error: {error}",
            ),
            status_code=400,
        )

    connector_id = _pop_state(state)
    if connector_id is None:
        return HTMLResponse(
            content=_result_html(
                success=False,
                message="Invalid or expired state token. Please try authorising again.",
            ),
            status_code=400,
        )

    connector = await db.get(Connector, connector_id)
    if not connector:
        return HTMLResponse(
            content=_result_html(success=False, message="Connector not found"),
            status_code=404,
        )

    # Retrieve the cached auth connector (preserves the PKCE code_verifier)
    cached = _STATE_STORE.pop(f"connector_{state}", None)
    auth_connector = cached[0] if cached else None

    if auth_connector is None:
        # State was valid but the connector instance was lost from the cache
        # (can happen if the server restarted between authorize and callback).
        # Rebuild it using the stored URLs from the database.
        from storage.encryption import decrypt
        client_secret = decrypt(connector.client_secret_encrypted) or ""
        oauth_cfg = OAuth2Config(
            client_id=connector.client_id or "",
            client_secret=client_secret,
            token_url=connector.token_url or "",
            auth_url=connector.auth_url or "",
            scopes=[],
            token_body_format=connector.token_body_format or "json",
            response_format=connector.response_format or "json",
            client_auth=connector.client_auth or "body",
            use_pkce=bool(connector.use_pkce),
        )
        auth_connector = AuthCodeConnector(oauth_cfg, httpx.AsyncClient(timeout=30.0))
        # Inject state so exchange_code's CSRF check passes
        auth_connector._state = state

    try:
        token = await auth_connector.exchange_code(code=code, state=state)
    except ValueError as exc:
        return HTMLResponse(
            content=_result_html(success=False, message=str(exc)),
            status_code=400,
        )
    except Exception as exc:
        return HTMLResponse(
            content=_result_html(
                success=False,
                message=f"Token exchange failed: {exc}",
            ),
            status_code=500,
        )

    # Persist the tokens encrypted
    connector.access_token_encrypted = encrypt(token.access_token)
    connector.refresh_token_encrypted = encrypt(token.refresh_token)
    connector.token_expires_at = (
        time.time() + token.expires_in if token.expires_in else None
    )
    connector.status = "active"

    await db.commit()

    return HTMLResponse(
        content=_result_html(
            success=True,
            message=f"Successfully connected. You can close this tab.",
            connector_name=connector.name,
        )
    )


# ── Success/error HTML page ────────────────────────────────────────────────

def _result_html(
    success: bool,
    message: str,
    connector_name: Optional[str] = None,
) -> str:
    import html as html_mod  # stdlib — no dep
    # Escape all user-controlled strings before interpolating into HTML.
    # connector_name comes from the database (set by the user at connector creation).
    # message may include exception text which could contain provider-supplied content.
    safe_message = html_mod.escape(str(message))
    safe_name = html_mod.escape(str(connector_name)) if connector_name else ""

    colour = "#00C2A8" if success else "#F25555"
    icon = "✓" if success else "✗"
    title = "Connected" if success else "Connection failed"
    name_line = f"<p style='color:#9BA3AA;margin:0 0 16px'>{safe_name}</p>" if safe_name else ""

    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'">
  <title>Crucible — {html_mod.escape(title)}</title>
  <style>
    body {{ font-family: 'IBM Plex Sans', system-ui, sans-serif;
            background: #0D0F10; color: #E8EAEC;
            display: flex; align-items: center; justify-content: center;
            min-height: 100vh; margin: 0; }}
    .card {{ background: #141618; border: 1px solid #2A2D30;
             border-radius: 10px; padding: 40px; max-width: 420px;
             text-align: center; }}
    .icon {{ font-size: 40px; color: {colour}; margin-bottom: 16px; }}
    h1 {{ font-size: 20px; margin: 0 0 8px; }}
    p  {{ font-size: 14px; color: #9BA3AA; line-height: 1.6; margin: 0; }}
    .close {{ margin-top: 24px; display: inline-block; padding: 8px 20px;
              background: {colour}20; border: 1px solid {colour};
              border-radius: 6px; color: {colour}; font-size: 13px;
              cursor: pointer; text-decoration: none; }}
  </style>
</head>
<body>
  <div class="card">
    <div class="icon">{icon}</div>
    <h1>{html_mod.escape(title)}</h1>
    {name_line}
    <p>{safe_message}</p>
    <a class="close" href="javascript:window.close()">Close tab</a>
  </div>
</body>
</html>"""
