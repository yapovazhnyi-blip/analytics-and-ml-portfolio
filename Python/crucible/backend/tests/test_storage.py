"""
Storage backend tests.

LocalStorage tests: run unconditionally (no AWS required).
S3Storage tests: use moto's @mock_aws decorator to mock the AWS API in-process.
  Moto intercepts boto3 calls and returns realistic responses without
  making any real network requests. This lets us test S3 semantics
  (multipart upload, presigned URLs, pagination) without AWS credentials.

StorageContext tests: verify the test injection helper works.
Factory tests: verify settings→backend mapping.
"""

from __future__ import annotations

import pytest
import os
import tempfile
from pathlib import Path


# ══════════════════════════════════════════════════════════════════════════
# LOCAL STORAGE
# ══════════════════════════════════════════════════════════════════════════

class TestLocalStorage:

    @pytest.fixture
    def storage(self, tmp_path):
        from storage.local import LocalStorage
        return LocalStorage(root_dir=str(tmp_path))

    # ── write / read ──────────────────────────────────────────────────────

    def test_write_and_read_roundtrip(self, storage):
        storage.write("test/hello.txt", b"hello world")
        assert storage.read("test/hello.txt") == b"hello world"

    def test_write_creates_parent_dirs(self, storage):
        storage.write("a/b/c/deep.bin", b"deep content")
        assert storage.exists("a/b/c/deep.bin")

    def test_write_overwrites_existing(self, storage):
        storage.write("file.txt", b"v1")
        storage.write("file.txt", b"v2")
        assert storage.read("file.txt") == b"v2"

    def test_read_missing_key_raises_file_not_found(self, storage):
        with pytest.raises(FileNotFoundError):
            storage.read("nonexistent/key.txt")

    def test_write_text_and_read_text(self, storage):
        storage.write_text("greeting.txt", "Hello, Crucible!")
        assert storage.read_text("greeting.txt") == "Hello, Crucible!"

    def test_write_text_unicode(self, storage):
        storage.write_text("unicode.txt", "日本語テスト")
        assert storage.read_text("unicode.txt") == "日本語テスト"

    def test_empty_bytes_write_and_read(self, storage):
        storage.write("empty.bin", b"")
        assert storage.read("empty.bin") == b""

    def test_large_file(self, storage):
        data = b"x" * (5 * 1024 * 1024)   # 5 MB
        storage.write("large.bin", data)
        assert storage.read("large.bin") == data

    # ── exists ────────────────────────────────────────────────────────────

    def test_exists_true_for_written_key(self, storage):
        storage.write("exists.txt", b"yes")
        assert storage.exists("exists.txt") is True

    def test_exists_false_for_missing_key(self, storage):
        assert storage.exists("missing.txt") is False

    # ── delete ────────────────────────────────────────────────────────────

    def test_delete_removes_file(self, storage):
        storage.write("to_delete.txt", b"bye")
        storage.delete("to_delete.txt")
        assert not storage.exists("to_delete.txt")

    def test_delete_missing_key_is_idempotent(self, storage):
        """Deleting a non-existent key must not raise."""
        storage.delete("never_existed.txt")   # should not raise

    # ── list ──────────────────────────────────────────────────────────────

    def test_list_all_keys(self, storage):
        storage.write("a/1.txt", b"1")
        storage.write("a/2.txt", b"2")
        storage.write("b/3.txt", b"3")
        keys = storage.list()
        assert "a/1.txt" in keys
        assert "a/2.txt" in keys
        assert "b/3.txt" in keys

    def test_list_with_prefix(self, storage):
        storage.write("datasets/1/data.csv", b"data")
        storage.write("datasets/2/data.csv", b"data")
        storage.write("models/1/model.pkl", b"model")
        keys = storage.list("datasets/")
        assert all(k.startswith("datasets/") for k in keys)
        assert not any("models" in k for k in keys)

    def test_list_empty_prefix_returns_all(self, storage):
        storage.write("x.txt", b"x")
        storage.write("y/z.txt", b"z")
        keys = storage.list("")
        assert len(keys) == 2

    def test_list_missing_prefix_returns_empty(self, storage):
        assert storage.list("nonexistent/prefix/") == []

    # ── upload_file / download_file ───────────────────────────────────────

    def test_upload_file(self, storage, tmp_path):
        local = tmp_path / "src.csv"
        local.write_bytes(b"col1,col2\n1,2\n")
        storage.upload_file(str(local), "uploads/src.csv")
        assert storage.exists("uploads/src.csv")
        assert storage.read("uploads/src.csv") == b"col1,col2\n1,2\n"

    def test_upload_file_missing_source_raises(self, storage):
        with pytest.raises(FileNotFoundError):
            storage.upload_file("/nonexistent/file.csv", "dest.csv")

    def test_download_file(self, storage, tmp_path):
        storage.write("data.csv", b"a,b\n1,2\n")
        dest = tmp_path / "downloaded.csv"
        storage.download_file("data.csv", str(dest))
        assert dest.read_bytes() == b"a,b\n1,2\n"

    def test_download_file_missing_key_raises(self, storage, tmp_path):
        with pytest.raises(FileNotFoundError):
            storage.download_file("missing.csv", str(tmp_path / "out.csv"))

    def test_download_creates_parent_dirs(self, storage, tmp_path):
        storage.write("file.txt", b"data")
        dest = tmp_path / "a" / "b" / "c" / "out.txt"
        storage.download_file("file.txt", str(dest))
        assert dest.read_bytes() == b"data"

    # ── get_url ───────────────────────────────────────────────────────────

    def test_get_url_returns_file_uri(self, storage):
        storage.write("model.pkl", b"pkl")
        url = storage.get_url("model.pkl")
        assert url.startswith("file://")

    # ── copy ──────────────────────────────────────────────────────────────

    def test_copy(self, storage):
        storage.write("src.txt", b"content")
        storage.copy("src.txt", "dst.txt")
        assert storage.read("dst.txt") == b"content"
        assert storage.read("src.txt") == b"content"   # original unchanged

    def test_copy_missing_source_raises(self, storage):
        with pytest.raises(FileNotFoundError):
            storage.copy("ghost.txt", "dest.txt")

    # ── path traversal protection ─────────────────────────────────────────

    def test_path_traversal_rejected(self, storage):
        with pytest.raises((ValueError, FileNotFoundError)):
            storage.write("../../etc/passwd", b"evil")

    def test_absolute_path_rejected(self, storage):
        with pytest.raises((ValueError, FileNotFoundError)):
            storage.read("/etc/hosts")


# ══════════════════════════════════════════════════════════════════════════
# S3 STORAGE (moto mock)
# ══════════════════════════════════════════════════════════════════════════

@pytest.fixture
def s3_storage(tmp_path):
    """S3Storage backed by a moto-mocked S3 bucket."""
    from moto import mock_aws
    import boto3

    with mock_aws():
        # Create the mock bucket
        s3 = boto3.client("s3", region_name="us-east-1")
        s3.create_bucket(Bucket="test-bucket")

        from storage.s3 import S3Storage
        backend = S3Storage(
            bucket_name="test-bucket",
            prefix="test/",
            region_name="us-east-1",
        )
        yield backend


class TestS3Storage:
    """All tests run inside the moto mock context via the s3_storage fixture."""

    def test_write_and_read_roundtrip(self, s3_storage):
        s3_storage.write("hello.txt", b"hello from S3")
        assert s3_storage.read("hello.txt") == b"hello from S3"

    def test_exists_true_after_write(self, s3_storage):
        s3_storage.write("check.bin", b"data")
        assert s3_storage.exists("check.bin") is True

    def test_exists_false_for_missing(self, s3_storage):
        assert s3_storage.exists("ghost.bin") is False

    def test_delete_removes_object(self, s3_storage):
        s3_storage.write("del.txt", b"bye")
        s3_storage.delete("del.txt")
        assert not s3_storage.exists("del.txt")

    def test_delete_missing_is_idempotent(self, s3_storage):
        s3_storage.delete("never_written.txt")   # must not raise

    def test_list_returns_keys(self, s3_storage):
        s3_storage.write("a/1.txt", b"1")
        s3_storage.write("a/2.txt", b"2")
        keys = s3_storage.list("a/")
        assert "a/1.txt" in keys
        assert "a/2.txt" in keys

    def test_list_with_prefix_filters(self, s3_storage):
        s3_storage.write("datasets/data.csv", b"d")
        s3_storage.write("models/model.pkl", b"m")
        ds_keys = s3_storage.list("datasets/")
        assert all("datasets" in k for k in ds_keys)
        assert not any("models" in k for k in ds_keys)

    def test_read_missing_raises_file_not_found(self, s3_storage):
        with pytest.raises(FileNotFoundError):
            s3_storage.read("missing/key.bin")

    def test_upload_and_download_file(self, s3_storage, tmp_path):
        src = tmp_path / "upload.csv"
        src.write_bytes(b"id,value\n1,100\n")
        s3_storage.upload_file(str(src), "uploads/data.csv")

        dest = tmp_path / "downloaded.csv"
        s3_storage.download_file("uploads/data.csv", str(dest))
        assert dest.read_bytes() == b"id,value\n1,100\n"

    def test_get_url_returns_https_string(self, s3_storage):
        s3_storage.write("artifact.pkl", b"model")
        url = s3_storage.get_url("artifact.pkl", expires_in=300)
        assert "http" in url.lower()

    def test_copy_server_side(self, s3_storage):
        s3_storage.write("original.txt", b"copy me")
        s3_storage.copy("original.txt", "copy.txt")
        assert s3_storage.read("copy.txt") == b"copy me"
        assert s3_storage.read("original.txt") == b"copy me"

    def test_write_text_roundtrip(self, s3_storage):
        s3_storage.write_text("notes.txt", "Hello S3!")
        assert s3_storage.read_text("notes.txt") == "Hello S3!"


# ══════════════════════════════════════════════════════════════════════════
# STORAGE FACTORY
# ══════════════════════════════════════════════════════════════════════════

class TestStorageFactory:

    def test_local_backend_returned_by_default(self, tmp_path, monkeypatch):
        from storage import factory as f
        f.get_storage.cache_clear()
        monkeypatch.setenv("STORAGE_BACKEND", "local")

        import config
        monkeypatch.setattr(config.settings, "storage_backend", "local")
        monkeypatch.setattr(config.settings, "storage_local_root", str(tmp_path))

        backend = f.get_storage()
        from storage.local import LocalStorage
        assert isinstance(backend, LocalStorage)
        f.get_storage.cache_clear()

    def test_unknown_backend_raises(self, monkeypatch):
        from storage import factory as f
        f.get_storage.cache_clear()

        import config
        monkeypatch.setattr(config.settings, "storage_backend", "dropbox")
        with pytest.raises(ValueError, match="Unknown STORAGE_BACKEND"):
            f.get_storage()
        f.get_storage.cache_clear()

    def test_storage_context_injects_backend(self, tmp_path):
        from storage.factory import StorageContext, get_storage
        from storage.local import LocalStorage

        test_backend = LocalStorage(str(tmp_path))
        with StorageContext(test_backend):
            from storage import factory as f
            backend = f.get_storage()
            assert backend is test_backend
