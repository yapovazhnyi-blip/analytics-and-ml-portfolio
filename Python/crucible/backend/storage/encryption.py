"""
Encryption service for Crucible — protects sensitive fields at rest.

Uses Fernet symmetric encryption (AES-128-CBC + HMAC-SHA256) from the
cryptography library. Fernet is the right choice here because:
  - It's authenticated: a tampered ciphertext raises InvalidToken,
    so we detect corruption rather than silently decrypting garbage.
  - It's simple: one key, one function call to encrypt/decrypt.
  - It produces URL-safe base64 output, safe to store in a TEXT column.

Key management:
  - The key is read from ENCRYPTION_KEY env var (set in .env).
  - If no key is configured, encrypt/decrypt are no-ops that return
    the plaintext — this lets Phase 1 work without forcing a key setup,
    while making the secure path available when the key is configured.
  - In production, rotate the key by re-encrypting all stored values.

Generate a key:
    python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
"""

from __future__ import annotations

from typing import Optional

from config import settings

# Module-level Fernet instance — initialised once at import time.
# If no key is set, _fernet stays None and we fall back to plaintext.
_fernet = None

def _get_fernet():
    """Lazy-initialise Fernet so tests that don't set a key aren't broken."""
    global _fernet
    if _fernet is not None:
        return _fernet

    if not settings.encryption_key:
        return None

    try:
        from cryptography.fernet import Fernet
        _fernet = Fernet(settings.encryption_key.encode())
        return _fernet
    except Exception as exc:
        # Bad key format — log and fall through to plaintext mode.
        import structlog
        structlog.get_logger().warning(
            "crucible.encryption.init_failed",
            error=str(exc),
            note="Falling back to plaintext storage — set a valid ENCRYPTION_KEY.",
        )
        return None


def encrypt(plaintext: Optional[str]) -> Optional[str]:
    """
    Encrypts a string and returns the ciphertext as a base64 string.

    Returns None if input is None (no-op for optional fields).
    Falls back to the original string if no ENCRYPTION_KEY is configured
    — this allows Phase 1 development without mandatory key setup, while
    the secure path activates automatically when the key is present.
    """
    if plaintext is None:
        return None
    f = _get_fernet()
    if f is None:
        return plaintext   # plaintext fallback — warn in logs, not here
    return f.encrypt(plaintext.encode()).decode()


def decrypt(ciphertext: Optional[str]) -> Optional[str]:
    """
    Decrypts a Fernet ciphertext back to plaintext.

    Returns None if input is None.
    Falls back to returning the input unchanged if no key is configured
    (assumes the value was stored as plaintext in that case).

    Raises cryptography.fernet.InvalidToken if the ciphertext was
    tampered with or encrypted with a different key — callers must
    handle this at the service layer.
    """
    if ciphertext is None:
        return None
    f = _get_fernet()
    if f is None:
        return ciphertext   # plaintext fallback
    return f.decrypt(ciphertext.encode()).decode()


def is_encryption_enabled() -> bool:
    """Returns True if a valid encryption key is configured."""
    return _get_fernet() is not None
