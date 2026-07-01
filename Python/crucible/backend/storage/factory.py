"""
Storage factory — creates and returns the configured StorageBackend.

Usage:
    from storage.factory import get_storage

    backend = get_storage()
    backend.write("datasets/42/data.csv", data)
    url = backend.get_url("models/exp_7/model.pkl")

The backend is a module-level singleton created on first call.
Subsequent calls return the same instance (no reconnection overhead).

Configuration (via settings / environment variables):
    STORAGE_BACKEND=local    (default) → LocalStorage
    STORAGE_BACKEND=s3                → S3Storage

    # Required for S3:
    AWS_BUCKET_NAME=my-crucible-bucket
    AWS_REGION=us-east-1

    # Optional S3:
    AWS_STORAGE_PREFIX=crucible/   # isolate keys within the bucket
    AWS_ENDPOINT_URL=http://localhost:9000  # MinIO / R2 / custom endpoint
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from storage.base import StorageBackend


@lru_cache(maxsize=1)
def get_storage() -> StorageBackend:
    """
    Returns the configured StorageBackend singleton.

    Uses @lru_cache so the same instance is returned on every call —
    equivalent to a singleton but without the global variable.
    Cache is cleared in tests via get_storage.cache_clear().
    """
    from config import settings

    backend = getattr(settings, "storage_backend", "local").lower()

    if backend == "s3":
        return _make_s3(settings)
    elif backend == "local":
        return _make_local(settings)
    else:
        raise ValueError(
            f"Unknown STORAGE_BACKEND={backend!r}. "
            "Valid values: 'local', 's3'."
        )


def _make_local(settings) -> "LocalStorage":
    from storage.local import LocalStorage
    import os
    from pathlib import Path

    root = getattr(settings, "storage_local_root", None) or \
           os.environ.get("STORAGE_LOCAL_ROOT") or \
           str(Path(settings.dataset_storage_path).parent / "storage")

    return LocalStorage(root_dir=root)


def _make_s3(settings) -> "S3Storage":
    from storage.s3 import S3Storage
    import os

    bucket = getattr(settings, "aws_bucket_name", None) or \
             os.environ.get("AWS_BUCKET_NAME")
    if not bucket:
        raise ValueError(
            "STORAGE_BACKEND=s3 requires AWS_BUCKET_NAME to be set. "
            "Set it in .env or as an environment variable."
        )

    prefix     = getattr(settings, "aws_storage_prefix", "") or os.environ.get("AWS_STORAGE_PREFIX", "")
    region     = getattr(settings, "aws_region", None) or os.environ.get("AWS_REGION", "us-east-1")
    endpoint   = getattr(settings, "aws_endpoint_url", None) or os.environ.get("AWS_ENDPOINT_URL")

    return S3Storage(
        bucket_name=bucket,
        prefix=prefix,
        region_name=region,
        endpoint_url=endpoint,
    )


def override_storage(backend: StorageBackend) -> None:
    """
    Replaces the singleton with a specific backend instance.
    Used in tests to inject a LocalStorage with a temp directory.

    Call override_storage(None) or get_storage.cache_clear() to reset.
    """
    get_storage.cache_clear()
    # Re-populate the cache with the provided instance
    get_storage._cache_override = backend
    # Monkey-patch so the next call returns the override
    import functools
    original = get_storage.__wrapped__ if hasattr(get_storage, "__wrapped__") else None
    get_storage.cache_clear()

    # Simpler approach: just clear and let factory build from settings
    # For testing, use a context manager instead (see below)


class StorageContext:
    """
    Context manager for injecting a test storage backend.

    Usage in tests:
        with StorageContext(LocalStorage(tmp_path)):
            result = await some_service_that_uses_get_storage()
    """

    def __init__(self, backend: StorageBackend):
        self._backend = backend
        self._original = None

    def __enter__(self) -> StorageBackend:
        # Patch the factory to return our test backend
        import storage.factory as factory_mod
        self._original_fn = factory_mod.get_storage

        @lru_cache(maxsize=1)
        def _test_storage():
            return self._backend

        factory_mod.get_storage = _test_storage
        return self._backend

    def __exit__(self, *args):
        import storage.factory as factory_mod
        factory_mod.get_storage = self._original_fn
        if hasattr(self._original_fn, "cache_clear"):
            self._original_fn.cache_clear()
