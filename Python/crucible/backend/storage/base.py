"""
StorageBackend — abstract interface for file storage operations.

WHY THIS ABSTRACTION EXISTS
----------------------------
Crucible currently writes every file — dataset uploads, model artifacts,
ONNX exports, RAG embeddings — to the local filesystem using pathlib.
This works in development but breaks in any production environment that:

  - Runs multiple API pods (each pod sees its own filesystem)
  - Scales to zero and restarts (filesystem is ephemeral)
  - Requires auditable, versioned, replicated storage
  - Needs to share files between the training job and the inference server

The StorageBackend abstraction decouples where files are stored from
how they are used. All business logic receives a `StorageBackend` object
and calls `backend.upload()`, `backend.read()`, `backend.get_url()`.
The concrete implementation (local disk or S3) is injected from config.

SWITCHING BACKENDS
-------------------
In .env or environment variables:

  STORAGE_BACKEND=local    → LocalStorage (default, uses local filesystem)
  STORAGE_BACKEND=s3       → S3Storage    (requires AWS_BUCKET_NAME)

No code changes are required to switch. The factory returns the correct
implementation based on settings.

METHOD CONTRACT
---------------
All methods must be:
  - Idempotent: calling write() twice with the same key and content is safe
  - Consistent: read() after write() must return the same bytes
  - Atomic: partial writes must not be visible to read()

Error handling:
  - Raise FileNotFoundError for missing keys in read() and download()
  - Raise StorageError (defined below) for backend-specific failures
  - Never swallow exceptions silently
"""

from __future__ import annotations

from abc import ABC, abstractmethod


class StorageError(Exception):
    """Raised for backend-specific storage failures (e.g. S3 access denied)."""


class StorageBackend(ABC):
    """
    Abstract file storage interface.

    All file I/O in Crucible should go through this interface.
    Implementations: LocalStorage (development) and S3Storage (production).
    """

    # ── Core operations ───────────────────────────────────────────────────────

    @abstractmethod
    def write(self, key: str, data: bytes) -> None:
        """
        Writes bytes to the given key (path or S3 object key).

        Args:
            key:  Relative path within the storage root, e.g.
                  "datasets/42/data.csv" or "models/exp_7/model.pkl"
            data: Raw bytes to write.

        Raises:
            StorageError: on any backend failure.
        """

    @abstractmethod
    def read(self, key: str) -> bytes:
        """
        Reads and returns the bytes at the given key.

        Raises:
            FileNotFoundError: if the key does not exist.
            StorageError:      on backend failure.
        """

    @abstractmethod
    def exists(self, key: str) -> bool:
        """Returns True if the key exists in the backend."""

    @abstractmethod
    def delete(self, key: str) -> None:
        """
        Deletes the object at the given key.
        Does not raise if the key does not exist (idempotent delete).
        """

    @abstractmethod
    def list(self, prefix: str = "") -> list[str]:
        """
        Lists all keys with the given prefix.

        Args:
            prefix: e.g. "datasets/42/" lists all files under that dataset.

        Returns:
            List of full keys (not just the suffix after the prefix).
        """

    # ── File-path helpers ─────────────────────────────────────────────────────

    @abstractmethod
    def upload_file(self, local_path: str, key: str) -> str:
        """
        Uploads a local file to the backend.

        More efficient than read() + write() for large files because
        implementations can use streaming multipart upload (S3) or
        os.rename() (local) instead of loading the whole file into memory.

        Args:
            local_path: Absolute path to the file on the local filesystem.
            key:        Destination key in the backend.

        Returns:
            The key (for chaining / logging).

        Raises:
            FileNotFoundError: if local_path does not exist.
            StorageError:      on backend failure.
        """

    @abstractmethod
    def download_file(self, key: str, local_path: str) -> None:
        """
        Downloads the object at key to a local file.

        Creates parent directories if needed.

        Args:
            key:        Source key in the backend.
            local_path: Absolute destination path on the local filesystem.

        Raises:
            FileNotFoundError: if key does not exist.
            StorageError:      on backend failure.
        """

    @abstractmethod
    def get_url(self, key: str, expires_in: int = 3600) -> str:
        """
        Returns a URL to access the object.

        For LocalStorage: a file:// path (useful for debugging).
        For S3Storage:    a presigned HTTPS URL that expires in expires_in seconds.

        Args:
            key:        The object key.
            expires_in: URL lifetime in seconds (S3 only).

        Returns:
            A string URL.
        """

    # ── Convenience helpers ───────────────────────────────────────────────────

    def write_text(self, key: str, text: str, encoding: str = "utf-8") -> None:
        """Writes a string to the backend."""
        self.write(key, text.encode(encoding))

    def read_text(self, key: str, encoding: str = "utf-8") -> str:
        """Reads and decodes a string from the backend."""
        return self.read(key).decode(encoding)

    def copy(self, src_key: str, dst_key: str) -> None:
        """
        Copies an object within the backend.
        Default implementation: read + write (inefficient for large files).
        Subclasses should override with server-side copy where available.
        """
        self.write(dst_key, self.read(src_key))
