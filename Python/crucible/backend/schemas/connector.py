"""
Connector schemas — Pydantic models for the /connectors API.

Note that sensitive fields (client_secret, tokens) are write-only —
they are accepted on create/update but never returned in responses.
This prevents credential leakage through the API.
"""

from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


# ── Response schemas ───────────────────────────────────────────────────────

class ConnectorOut(BaseModel):
    """Connector representation — sensitive fields deliberately excluded."""
    id: int
    name: str
    connector_type: str
    status: str
    base_url: Optional[str] = None
    client_id: Optional[str] = None
    oauth2_flow: Optional[str] = None
    # These are the fields that were previously missing — non-sensitive URL
    # fields that must be stored and returned so the factory can construct
    # a correct OAuth2Config. Without them, token requests went to base_url
    # which is almost never the correct token endpoint host.
    token_url: Optional[str] = None
    auth_url: Optional[str] = None
    use_pkce: bool = False
    last_used_at: Optional[float] = None
    created_at: str

    model_config = {"from_attributes": True}


class ConnectorSummary(BaseModel):
    id: int
    name: str
    connector_type: str
    status: str
    created_at: str

    model_config = {"from_attributes": True}


# ── Request schemas ────────────────────────────────────────────────────────

class SQLConnectorCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    connector_type: str = Field(
        ..., description="sql_postgres | sql_sqlite"
    )
    db_url: str = Field(
        ...,
        description="Full connection string. Will be encrypted at rest.",
        examples=["postgresql://user:pass@host:5432/db"],
    )


class OAuthConnectorCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    base_url: str = Field(..., description="API base URL (must be a public HTTPS endpoint)")
    client_id: str
    client_secret: str = Field(..., description="Write-only. Encrypted at rest.")
    oauth2_flow: str = Field(..., description="client_credentials | authorization_code")
    scopes: list[str] = Field(default_factory=list)

    # Provider variance flags (from spike)
    token_url: str
    auth_url: Optional[str] = None        # Required for authorization_code flow
    token_body_format: str = "json"       # json | form
    response_format: str = "json"         # json | form
    client_auth: str = "body"             # body | basic
    use_pkce: bool = False

    def model_post_init(self, __context) -> None:
        """
        SSRF protection: block connectors that point at private or loopback
        addresses. An attacker could use a connector to probe internal
        services unreachable from the public internet.

        Override in dev by setting ALLOW_PRIVATE_URLS=true.
        """
        import os, ipaddress
        from urllib.parse import urlparse

        if os.getenv("ALLOW_PRIVATE_URLS", "").lower() == "true":
            return

        def _is_private(url: Optional[str]) -> bool:
            if not url:
                return False
            lo = url.lower()
            if any(lo.startswith(p) for p in (
                "http://localhost", "https://localhost",
                "http://127.", "https://127.",
                "http://0.",  "https://0.",
                "http://[::1]", "https://[::1]",
            )):
                return True
            try:
                host = urlparse(url).hostname or ""
                addr = ipaddress.ip_address(host)
                return addr.is_private or addr.is_loopback or addr.is_link_local
            except ValueError:
                pass
            return False

        for fname, val in [
            ("base_url", self.base_url),
            ("token_url", self.token_url),
            ("auth_url", self.auth_url),
        ]:
            if _is_private(val):
                raise ValueError(
                    f"'{fname}' points to a private or loopback address ({val!r}). "
                    "Set ALLOW_PRIVATE_URLS=true to override in development."
                )


class ConnectorTestResult(BaseModel):
    success: bool
    message: str
    latency_ms: Optional[float] = None
