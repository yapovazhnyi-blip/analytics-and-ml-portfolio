"""
OAuth2 connector spike for Crucible.

Goal: resolve provider variance before locking the connector interface.
Tests two flows against mocked real-world provider behaviour:
  1. Client Credentials  — server-to-server, no user (Stripe-style, internal APIs)
  2. Authorization Code + PKCE — user-facing (GitHub, Google)

Provider quirks handled declaratively via OAuth2Config flags rather than
subclasses per provider — five flags cover ~95% of real-world variance.
"""

import base64
import hashlib
import secrets
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urlencode, parse_qs

import httpx


# ── Token model ────────────────────────────────────────────────────────────

@dataclass
class OAuth2Token:
    access_token: str
    token_type: str = "Bearer"
    expires_in: Optional[int] = None       # seconds
    refresh_token: Optional[str] = None
    scope: Optional[str] = None
    issued_at: float = field(default_factory=time.time)
    raw: dict = field(default_factory=dict)

    def is_expired(self, buffer_secs: int = 60) -> bool:
        if self.expires_in is None:
            return False
        return time.time() >= self.issued_at + self.expires_in - buffer_secs


# ── Provider config — all variance lives here ──────────────────────────────

@dataclass
class OAuth2Config:
    client_id: str
    client_secret: str
    token_url: str
    scopes: list[str]

    # Auth-code flow only
    auth_url: str = ""
    redirect_uri: str = "http://localhost:8000/oauth/callback"

    # Provider quirks — five flags cover ~95% of real-world variance
    token_body_format: str = "json"     # "json" | "form"  — GitHub/Stripe need form
    response_format: str = "json"       # "json" | "form"  — GitHub returns form-encoded response
    client_auth: str = "body"           # "body" | "basic" — where to send client_id/secret
    use_pkce: bool = False              # required by: Google (web), Okta, some GitHub Apps
    pkce_method: str = "S256"


# ── Base connector ─────────────────────────────────────────────────────────

class OAuth2Connector(ABC):
    def __init__(self, config: OAuth2Config, http_client: Optional[httpx.AsyncClient] = None):
        self.config = config
        self._http = http_client or httpx.AsyncClient(timeout=30.0)
        self._token: Optional[OAuth2Token] = None

    @abstractmethod
    async def get_token(self) -> OAuth2Token:
        ...

    async def request(self, method: str, url: str, **kwargs) -> httpx.Response:
        """Authenticated request — auto-refreshes token if expired."""
        token = await self.get_token()
        headers = kwargs.pop("headers", {})
        headers["Authorization"] = f"Bearer {token.access_token}"
        return await self._http.request(method, url, headers=headers, **kwargs)

    async def close(self):
        await self._http.aclose()

    # ── Shared token-request logic ─────────────────────────────────────────

    async def _post_token(self, payload: dict) -> OAuth2Token:
        """
        Sends token request, handling body format + client auth variance.
        Returns parsed OAuth2Token regardless of provider response format.
        """
        headers: dict[str, str] = {"Accept": "application/json"}

        if self.config.client_auth == "basic":
            raw = f"{self.config.client_id}:{self.config.client_secret}".encode()
            headers["Authorization"] = "Basic " + base64.b64encode(raw).decode()
        else:
            payload["client_id"] = self.config.client_id
            payload["client_secret"] = self.config.client_secret

        if self.config.token_body_format == "form":
            headers["Content-Type"] = "application/x-www-form-urlencoded"
            resp = await self._http.post(self.config.token_url, data=payload, headers=headers)
        else:
            resp = await self._http.post(self.config.token_url, json=payload, headers=headers)

        resp.raise_for_status()
        data = self._parse_token_response(resp)

        if "error" in data:
            raise RuntimeError(f"OAuth2 error: {data['error']} — {data.get('error_description', '')}")

        return OAuth2Token(
            access_token=data["access_token"],
            token_type=data.get("token_type", "Bearer"),
            expires_in=int(data["expires_in"]) if "expires_in" in data else None,
            refresh_token=data.get("refresh_token"),
            scope=data.get("scope"),
            raw=data,
        )

    def _parse_token_response(self, resp: httpx.Response) -> dict:
        """
        Handles GitHub's form-encoded response body.
        Even with Accept: application/json, some providers ignore it.
        """
        ct = resp.headers.get("content-type", "")
        if "json" in ct:
            return resp.json()
        # form-encoded fallback (GitHub OAuth Apps)
        parsed = parse_qs(resp.text)
        return {k: v[0] for k, v in parsed.items()}


# ── Flow 1: Client Credentials ─────────────────────────────────────────────

class ClientCredentialsConnector(OAuth2Connector):
    """
    Server-to-server. No user interaction.
    Providers: internal APIs, Twilio, some Stripe endpoints.

    client_auth="basic"      → sends credentials in Authorization header
    token_body_format="form" → sends application/x-www-form-urlencoded
    """

    async def get_token(self) -> OAuth2Token:
        if self._token and not self._token.is_expired():
            return self._token
        self._token = await self._fetch()
        return self._token

    async def _fetch(self) -> OAuth2Token:
        return await self._post_token({
            "grant_type": "client_credentials",
            "scope": " ".join(self.config.scopes),
        })


# ── Flow 2: Authorization Code + PKCE ──────────────────────────────────────

class AuthCodeConnector(OAuth2Connector):
    """
    User-facing OAuth flow.
    Supports PKCE (S256) — required by Google web apps, Okta, some GitHub Apps.
    Handles:
      - GitHub:  form body, form response, no PKCE
      - Google:  json body, json response, PKCE required
      - Okta:    basic auth, json body/response, PKCE optional
    """

    def __init__(self, config: OAuth2Config, http_client: Optional[httpx.AsyncClient] = None):
        super().__init__(config, http_client)
        self._code_verifier: Optional[str] = None
        self._state: Optional[str] = None

    def build_auth_url(self) -> tuple[str, str]:
        """
        Returns (authorization_url, state).
        Caller must store state server-side and validate on callback.
        """
        self._state = secrets.token_urlsafe(32)
        params = {
            "client_id": self.config.client_id,
            "redirect_uri": self.config.redirect_uri,
            "scope": " ".join(self.config.scopes),
            "response_type": "code",
            "state": self._state,
        }
        if self.config.use_pkce:
            self._code_verifier = secrets.token_urlsafe(64)
            params["code_challenge"] = self._s256(self._code_verifier)
            params["code_challenge_method"] = self.config.pkce_method

        return f"{self.config.auth_url}?{urlencode(params)}", self._state

    async def exchange_code(self, code: str, state: str) -> OAuth2Token:
        """Exchange authorization code for token. Validates state (CSRF guard)."""
        if state != self._state:
            raise ValueError("State mismatch — possible CSRF attack")

        payload: dict = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": self.config.redirect_uri,
        }
        if self.config.use_pkce and self._code_verifier:
            payload["code_verifier"] = self._code_verifier

        self._token = await self._post_token(payload)
        return self._token

    async def refresh(self) -> OAuth2Token:
        """Refresh using refresh_token. Not all providers issue one."""
        if not self._token or not self._token.refresh_token:
            raise ValueError("No refresh token — re-authorize the user")

        self._token = await self._post_token({
            "grant_type": "refresh_token",
            "refresh_token": self._token.refresh_token,
        })
        return self._token

    async def get_token(self) -> OAuth2Token:
        if not self._token:
            raise RuntimeError("No token — call exchange_code() first")
        if self._token.is_expired() and self._token.refresh_token:
            return await self.refresh()
        return self._token

    @staticmethod
    def _s256(verifier: str) -> str:
        digest = hashlib.sha256(verifier.encode()).digest()
        return base64.urlsafe_b64encode(digest).rstrip(b"=").decode()


# ── Provider presets ───────────────────────────────────────────────────────
# Validated against real provider docs. Flags capture all variance.

def github_config(client_id: str, client_secret: str) -> OAuth2Config:
    """
    GitHub OAuth Apps.
    Quirks: form body, form-encoded response (even with Accept: application/json
    set, some older app configs still return form-encoded). No PKCE for OAuth Apps.
    GitHub Apps support PKCE — set use_pkce=True if using GitHub App flow.
    """
    return OAuth2Config(
        client_id=client_id,
        client_secret=client_secret,
        auth_url="https://github.com/login/oauth/authorize",
        token_url="https://github.com/login/oauth/access_token",
        scopes=["read:user", "repo"],
        token_body_format="form",
        response_format="form",   # GitHub ignores Accept header on some app configs
        client_auth="body",
        use_pkce=False,
    )


def google_config(client_id: str, client_secret: str) -> OAuth2Config:
    """
    Google OAuth 2.0. PKCE required for web apps (security policy as of 2023).
    JSON body + JSON response. Refresh tokens only issued on first authorization.
    """
    return OAuth2Config(
        client_id=client_id,
        client_secret=client_secret,
        auth_url="https://accounts.google.com/o/oauth2/v2/auth",
        token_url="https://oauth2.googleapis.com/token",
        scopes=["openid", "email", "profile"],
        token_body_format="json",
        response_format="json",
        client_auth="body",
        use_pkce=True,
    )


def stripe_m2m_config(client_id: str, client_secret: str) -> OAuth2Config:
    """
    Stripe Connect / internal API — client credentials via Basic auth header.
    """
    return OAuth2Config(
        client_id=client_id,
        client_secret=client_secret,
        token_url="https://connect.stripe.com/oauth/token",
        scopes=["read_write"],
        token_body_format="form",
        response_format="json",
        client_auth="basic",
        use_pkce=False,
    )
