"""
Password hashing using the bcrypt library directly.

WHY BCRYPT
----------
bcrypt is deliberately slow (configurable work factor) making brute-force
attacks expensive. Work factor 12 (default) takes ~250ms per hash.
This is imperceptible to users but limits attackers to ~4 attempts/second
versus millions per second with SHA-256.

WHY NOT PASSLIB
---------------
passlib 1.7.4 (last release 2020) has a compatibility bug with bcrypt 4.x:
its internal `detect_wrap_bug()` test uses a 73-char string, which bcrypt 4.x
correctly rejects with ValueError instead of silently truncating.
Using the bcrypt library directly avoids this entirely.
"""

import bcrypt as _bcrypt


def hash_password(plain: str) -> str:
    """Returns a bcrypt hash of the plain-text password."""
    return _bcrypt.hashpw(plain.encode("utf-8"), _bcrypt.gensalt(rounds=12)).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    """Returns True if plain matches the stored bcrypt hash."""
    try:
        return _bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
    except Exception:
        return False

