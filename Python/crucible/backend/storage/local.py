"""
LocalStorage — StorageBackend backed by the local filesystem.

This is the default backend. All files are stored under a configurable
root directory (settings.storage_root, default: ./data/storage/).

Key format:
  "datasets/42/data.csv"  →  {root}/datasets/42/data.csv
  "models/exp_7/model.pkl" →  {root}/models/exp_7/model.pkl

Suitable for:
  - Development on a single machine
  - Single-pod production deployments with persistent volumes
  - Testing (use a tempdir as the root)

NOT suitable for:
  - Multi-pod deployments (each pod sees only its own filesystem)
  - Ephemeral container environments (files lost on restart)
  - → Switch to S3Storage in those cases
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

from storage.base import StorageBackend, StorageError


class LocalStorage(StorageBackend):
    """
    Filesystem-backed storage.

    All keys are resolved relative to `root_dir`. Parent directories
    are created automatically on write. Directory traversal attacks
    (keys containing '..') are rejected.
    """

    def __init__(self, root_dir: str):
        self._root = Path(root_dir).resolve()
        self._root.mkdir(parents=True, exist_ok=True)

    @property
    def root(self) -> Path:
        return self._root

    def _resolve(self, key: str) -> Path:
        """
        Resolves a storage key to an absolute local path.

        Raises ValueError if the resolved path escapes the root directory
        (path traversal protection).
        """
        # Normalise: strip leading slash, replace backslashes
        key = key.lstrip("/").replace("\\", "/")
        resolved = (self._root / key).resolve()
        try:
            resolved.relative_to(self._root)
        except ValueError:
            raise ValueError(
                f"Key '{key}' resolves outside the storage root — "
                "path traversal is not allowed."
            )
        return resolved

    # ── Core operations ───────────────────────────────────────────────────────

    def write(self, key: str, data: bytes) -> None:
        path = self._resolve(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Write atomically via a temp file + rename to prevent partial reads
        tmp = path.with_suffix(path.suffix + ".tmp")
        try:
            tmp.write_bytes(data)
            tmp.rename(path)
        except Exception as exc:
            tmp.unlink(missing_ok=True)
            raise StorageError(f"LocalStorage.write({key!r}) failed: {exc}") from exc

    def read(self, key: str) -> bytes:
        path = self._resolve(key)
        if not path.exists():
            raise FileNotFoundError(f"Storage key not found: {key!r}")
        try:
            return path.read_bytes()
        except Exception as exc:
            raise StorageError(f"LocalStorage.read({key!r}) failed: {exc}") from exc

    def exists(self, key: str) -> bool:
        try:
            return self._resolve(key).exists()
        except ValueError:
            return False

    def delete(self, key: str) -> None:
        path = self._resolve(key)
        if path.exists():
            if path.is_dir():
                shutil.rmtree(path, ignore_errors=True)
            else:
                path.unlink(missing_ok=True)

    def list(self, prefix: str = "") -> list[str]:
        search_root = self._resolve(prefix) if prefix else self._root
        if not search_root.exists():
            return []
        keys = []
        for p in search_root.rglob("*"):
            if p.is_file():
                rel = p.relative_to(self._root)
                keys.append(str(rel).replace("\\", "/"))
        return sorted(keys)

    # ── File-path operations ──────────────────────────────────────────────────

    def upload_file(self, local_path: str, key: str) -> str:
        """
        Copies a local file into the storage root.

        Uses shutil.copy2 which preserves metadata and is more efficient
        than read() + write() for large files.
        """
        src = Path(local_path)
        if not src.exists():
            raise FileNotFoundError(f"Source file not found: {local_path!r}")
        dst = self._resolve(key)
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(src, dst)
        except Exception as exc:
            raise StorageError(f"LocalStorage.upload_file({key!r}) failed: {exc}") from exc
        return key

    def download_file(self, key: str, local_path: str) -> None:
        """
        Copies a file from the storage root to a local destination.
        """
        src = self._resolve(key)
        if not src.exists():
            raise FileNotFoundError(f"Storage key not found: {key!r}")
        dst = Path(local_path)
        dst.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(src, dst)
        except Exception as exc:
            raise StorageError(f"LocalStorage.download_file({key!r}) failed: {exc}") from exc

    def get_url(self, key: str, expires_in: int = 3600) -> str:
        """Returns a file:// URL for the local path (useful for debugging)."""
        return self._resolve(key).as_uri()

    # ── Server-side copy optimisation ─────────────────────────────────────────

    def copy(self, src_key: str, dst_key: str) -> None:
        """Server-side copy using shutil — avoids loading file into memory."""
        src = self._resolve(src_key)
        dst = self._resolve(dst_key)
        if not src.exists():
            raise FileNotFoundError(f"Source key not found: {src_key!r}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)

    def __repr__(self) -> str:
        return f"LocalStorage(root={self._root!r})"
