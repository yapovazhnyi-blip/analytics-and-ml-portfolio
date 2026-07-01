"""
UserAPIKey — stores encrypted third-party API keys on behalf of users.

WHY PER-USER KEYS
-----------------
A single server-side API key shared by all users has three problems:

  Cost: every advisor query, agent run, and RAG call bills to the
  platform operator's account. At scale this becomes the dominant
  operating cost.

  Rate limits: Anthropic's rate limits apply per API key. A single key
  shared across many concurrent users hits the limit far sooner than
  each user having their own key with their own quota.

  Enterprise compliance: companies using Crucible may have negotiated
  enterprise Anthropic contracts with data residency requirements. They
  cannot route data through the platform operator's key — they need
  their own key in the request.

BYOK solves all three: users bring their own key, pay their own bills,
use their own rate limits, and satisfy their own compliance requirements.

SECURITY PROPERTIES
--------------------
Keys are stored encrypted using Fernet (AES-128-CBC + HMAC-SHA256).
The key is never returned in any API response — only a hint showing the
last 4 characters (e.g. "...k2bx") is exposed, enough for the user to
identify which key is stored without revealing the full secret.

The key_hint is computed at write time and stored in plaintext — it's
not sensitive (the last 4 chars of a 40+ char key provide no useful
information to an attacker) and must be readable without decryption for
display purposes.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column

from models.base import Base


class UserAPIKey(Base):
    """Encrypted API key for a user + provider combination."""

    __tablename__ = "user_api_keys"

    id: Mapped[int] = mapped_column(sa.Integer, primary_key=True, autoincrement=True)

    user_id: Mapped[int] = mapped_column(
        sa.Integer,
        sa.ForeignKey("users.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )

    # Provider identifier — "anthropic" | "openai" | "groq" | "bedrock"
    provider: Mapped[str] = mapped_column(
        sa.String(32), nullable=False, default="anthropic"
    )

    # Fernet-encrypted API key
    encrypted_key: Mapped[str] = mapped_column(sa.Text, nullable=False)

    # Last 4 characters of the plaintext key — shown in UI for identification
    # (never the full key — this is intentionally low-entropy for display only)
    key_hint: Mapped[str] = mapped_column(sa.String(8), nullable=False, default="")

    # Optional label the user assigns
    label: Mapped[Optional[str]] = mapped_column(sa.String(128), nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        sa.DateTime(timezone=True),
        nullable=False,
        server_default=sa.func.now(),
    )

    last_used_at: Mapped[Optional[datetime]] = mapped_column(
        sa.DateTime(timezone=True), nullable=True
    )

    # Soft validation flag — set after the key passes a live API test
    validated: Mapped[bool] = mapped_column(
        sa.Boolean, nullable=False, default=False
    )

    __table_args__ = (
        # One key per user per provider
        sa.UniqueConstraint("user_id", "provider", name="uq_user_provider_key"),
    )

    def __repr__(self) -> str:
        return f"<UserAPIKey user={self.user_id} provider={self.provider!r} hint=...{self.key_hint}>"
