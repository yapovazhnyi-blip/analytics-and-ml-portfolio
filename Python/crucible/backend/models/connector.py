"""
Connector model — stores the configuration for a data source connection.

Each connector record represents a configured, reusable connection to a
data source. For OAuth connectors, encrypted token storage lives here
rather than in memory (validated in the OAuth2 spike conclusions).

Sensitive fields (client_secret, access_token, refresh_token) are stored
as Text and must be encrypted at the application layer before saving.
Phase 1 uses environment-variable-based encryption key. Phase 2+ can
swap to a secret manager without changing this model.
"""

from typing import Optional
import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column
from .base import Base, TimestampMixin


class Connector(Base, TimestampMixin):
    __tablename__ = "connectors"

    id: Mapped[int] = mapped_column(primary_key=True)

    name: Mapped[str] = mapped_column(sa.String(255), nullable=False)

    connector_type: Mapped[str] = mapped_column(
        sa.String(50), nullable=False
    )  # "csv" | "parquet" | "sql_postgres" | "sql_sqlite" | "rest_oauth2" | "rest_api_key"

    # ── SQL connector fields ───────────────────────────────────────────────
    # Connection string stored encrypted. Null for non-SQL connectors.
    db_url_encrypted: Mapped[Optional[str]] = mapped_column(sa.Text, nullable=True)

    # ── REST / OAuth2 connector fields ────────────────────────────────────
    base_url: Mapped[Optional[str]] = mapped_column(sa.String(1024), nullable=True)
    client_id: Mapped[Optional[str]] = mapped_column(sa.String(255), nullable=True)
    client_secret_encrypted: Mapped[Optional[str]] = mapped_column(sa.Text, nullable=True)

    # OAuth2 flow type — drives which connector class is instantiated
    oauth2_flow: Mapped[Optional[str]] = mapped_column(
        sa.String(50), nullable=True
    )  # "client_credentials" | "authorization_code"

    # OAuth2 config flags (from spike: the five flags that cover ~95% of variance)
    token_body_format: Mapped[Optional[str]] = mapped_column(sa.String(10), nullable=True)
    response_format: Mapped[Optional[str]] = mapped_column(sa.String(10), nullable=True)
    client_auth: Mapped[Optional[str]] = mapped_column(sa.String(10), nullable=True)
    use_pkce: Mapped[bool] = mapped_column(sa.Boolean, default=False, nullable=False)

    # OAuth2 endpoint URLs — stored separately from base_url because they
    # are almost never the same host. For GitHub: base_url is api.github.com
    # but token_url is github.com/login/oauth/access_token.
    # These were the missing columns that caused the token_url bug.
    token_url: Mapped[Optional[str]] = mapped_column(sa.String(1024), nullable=True)
    auth_url: Mapped[Optional[str]] = mapped_column(sa.String(1024), nullable=True)

    # Token storage (encrypted) — persisted across restarts
    access_token_encrypted: Mapped[Optional[str]] = mapped_column(sa.Text, nullable=True)
    refresh_token_encrypted: Mapped[Optional[str]] = mapped_column(sa.Text, nullable=True)
    token_expires_at: Mapped[Optional[float]] = mapped_column(sa.Float, nullable=True)

    # Connection health
    status: Mapped[str] = mapped_column(
        sa.String(20), nullable=False, default="unconfigured"
    )  # "unconfigured" | "active" | "error" | "token_expired"

    last_used_at: Mapped[Optional[float]] = mapped_column(sa.Float, nullable=True)

    def __repr__(self) -> str:
        return f"<Connector id={self.id} name={self.name!r} type={self.connector_type}>"
