"""
Encryption service tests.

Validates Fernet encrypt/decrypt round-trip, None handling, and the
graceful plaintext fallback when no key is configured.
"""

import pytest
from unittest.mock import patch
from cryptography.fernet import Fernet

# Generate a valid key for tests
VALID_KEY = Fernet.generate_key().decode()


class TestEncryption:

    def test_encrypt_decrypt_roundtrip(self):
        """Encrypting then decrypting returns the original plaintext."""
        with patch("storage.encryption.settings") as mock_settings:
            mock_settings.encryption_key = VALID_KEY
            # Reset the module-level cache so our key is picked up
            import storage.encryption as enc
            enc._fernet = None

            ciphertext = enc.encrypt("postgresql://user:pass@host/db")
            assert ciphertext != "postgresql://user:pass@host/db"  # must not be plaintext
            assert enc.decrypt(ciphertext) == "postgresql://user:pass@host/db"

            enc._fernet = None  # reset after test

    def test_none_input_returns_none(self):
        """None in, None out — no error."""
        import storage.encryption as enc
        assert enc.encrypt(None) is None
        assert enc.decrypt(None) is None

    def test_plaintext_fallback_when_no_key(self):
        """Without an encryption key, encrypt/decrypt are identity functions."""
        import storage.encryption as enc
        enc._fernet = None

        with patch("storage.encryption.settings") as mock_settings:
            mock_settings.encryption_key = None
            enc._fernet = None

            result = enc.encrypt("sensitive_value")
            assert result == "sensitive_value"  # returned unchanged

            decrypted = enc.decrypt("sensitive_value")
            assert decrypted == "sensitive_value"

            enc._fernet = None  # reset

    def test_ciphertext_is_not_plaintext(self):
        """Encrypted output must not contain the original string."""
        import storage.encryption as enc
        enc._fernet = None

        with patch("storage.encryption.settings") as mock_settings:
            mock_settings.encryption_key = VALID_KEY
            enc._fernet = None

            original = "my_secret_password_12345"
            ciphertext = enc.encrypt(original)
            assert original not in (ciphertext or "")

            enc._fernet = None

    def test_is_encryption_enabled_true_with_key(self):
        import storage.encryption as enc
        enc._fernet = None

        with patch("storage.encryption.settings") as mock_settings:
            mock_settings.encryption_key = VALID_KEY
            enc._fernet = None
            assert enc.is_encryption_enabled() is True
            enc._fernet = None

    def test_is_encryption_enabled_false_without_key(self):
        import storage.encryption as enc
        enc._fernet = None

        with patch("storage.encryption.settings") as mock_settings:
            mock_settings.encryption_key = None
            enc._fernet = None
            assert enc.is_encryption_enabled() is False
            enc._fernet = None
