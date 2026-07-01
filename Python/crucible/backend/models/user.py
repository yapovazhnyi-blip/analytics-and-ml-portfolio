"""
User ORM model for JWT authentication.

ROLES
-----
viewer      — read-only access to all resources (GET endpoints only)
contributor — can create and run experiments, upload datasets, add connectors
admin       — full access, including user management and delete operations

The role check is enforced at the dependency level (auth/dependencies.py),
not in individual route handlers, so adding a new endpoint is safe by default.
"""

from __future__ import annotations

from typing import Optional

import sqlalchemy as sa
from sqlalchemy.orm import Mapped, mapped_column

from models.base import Base, TimestampMixin


class User(Base, TimestampMixin):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(
        sa.Integer, primary_key=True, autoincrement=True
    )
    email: Mapped[str] = mapped_column(
        sa.String(255), unique=True, nullable=False, index=True
    )
    # bcrypt hash — never store plain text
    hashed_password: Mapped[str] = mapped_column(
        sa.String(255), nullable=False
    )
    # viewer | contributor | admin
    role: Mapped[str] = mapped_column(
        sa.String(20), nullable=False, default="contributor"
    )
    is_active: Mapped[bool] = mapped_column(
        sa.Boolean, nullable=False, default=True
    )
    # Per-user Anthropic API key, Fernet-encrypted at rest.
    # When set, used in preference to the server-level ANTHROPIC_API_KEY.
    # The raw key is NEVER stored or returned in any API response.
    anthropic_key_encrypted: Mapped[Optional[str]] = mapped_column(
        sa.Text, nullable=True, default=None
    )

    def __repr__(self) -> str:
        return f"<User id={self.id} email={self.email!r} role={self.role!r}>"
