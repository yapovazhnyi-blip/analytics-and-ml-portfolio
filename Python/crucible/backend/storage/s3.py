"""
S3Storage — StorageBackend backed by AWS S3 (or any S3-compatible API).

HOW S3 KEYS WORK
-----------------
S3 has no real directory hierarchy — all objects live in a flat key namespace
within a bucket. The "/" character is just a naming convention that the AWS
console renders as folder icons.

So "datasets/42/data.csv" is a valid S3 key that means "the object named
datasets/42/data.csv in bucket <BUCKET_NAME>".

CREDENTIALS
-----------
Boto3 uses the standard AWS credential chain automatically:
  1. Environment variables: AWS_ACCESS_KEY_ID + AWS_SECRET_ACCESS_KEY
  2. ~/.aws/credentials file (from `aws configure`)
  3. IAM instance role (for EC2/ECS/Lambda)
  4. IAM task role (for ECS tasks / Kubernetes ServiceAccounts with IRSA)

For local development, set environment variables or run `aws configure`.
For production on AWS, use IAM roles — never hard-code credentials.

S3-COMPATIBLE STORES
---------------------
MinIO, DigitalOcean Spaces, Cloudflare R2, and others expose the S3 API.
Set endpoint_url in settings to point to the custom endpoint:
  AWS_ENDPOINT_URL=http://localhost:9000  # MinIO

PRESIGNED URLS
--------------
get_url() returns a presigned HTTPS URL that allows temporary access to
a private object without AWS credentials. The URL expires after expires_in
seconds (default 1 hour). This is the standard pattern for serving model
artifacts and downloaded exports to authenticated frontend users.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

from storage.base import StorageBackend, StorageError


class S3Storage(StorageBackend):
    """
    S3-backed storage using boto3.

    Constructor args:
        bucket_name:   S3 bucket name.
        prefix:        Optional key prefix prepended to all keys
                       (e.g. "crucible/" to isolate from other data in the bucket).
        region_name:   AWS region (defaults to AWS_DEFAULT_REGION env var).
        endpoint_url:  Override for S3-compatible APIs (MinIO, R2, etc.).
    """

    def __init__(
        self,
        bucket_name: str,
        prefix: str = "",
        region_name: Optional[str] = None,
        endpoint_url: Optional[str] = None,
    ):
        import boto3

        self._bucket = bucket_name
        self._prefix = prefix.rstrip("/") + "/" if prefix else ""
        self._client = boto3.client(
            "s3",
            region_name=region_name or os.environ.get("AWS_DEFAULT_REGION", "us-east-1"),
            endpoint_url=endpoint_url or os.environ.get("AWS_ENDPOINT_URL"),
        )

    def _full_key(self, key: str) -> str:
        """Prepends the bucket prefix to the storage key."""
        return self._prefix + key.lstrip("/")

    def _strip_prefix(self, full_key: str) -> str:
        """Removes the bucket prefix to return a plain storage key."""
        if self._prefix and full_key.startswith(self._prefix):
            return full_key[len(self._prefix):]
        return full_key

    # ── Core operations ───────────────────────────────────────────────────────

    def write(self, key: str, data: bytes) -> None:
        try:
            self._client.put_object(
                Bucket=self._bucket,
                Key=self._full_key(key),
                Body=data,
            )
        except Exception as exc:
            raise StorageError(f"S3Storage.write({key!r}) failed: {exc}") from exc

    def read(self, key: str) -> bytes:
        from botocore.exceptions import ClientError
        try:
            response = self._client.get_object(
                Bucket=self._bucket, Key=self._full_key(key)
            )
            return response["Body"].read()
        except ClientError as exc:
            if exc.response["Error"]["Code"] in ("NoSuchKey", "404"):
                raise FileNotFoundError(f"S3 key not found: {key!r}")
            raise StorageError(f"S3Storage.read({key!r}) failed: {exc}") from exc
        except Exception as exc:
            raise StorageError(f"S3Storage.read({key!r}) failed: {exc}") from exc

    def exists(self, key: str) -> bool:
        from botocore.exceptions import ClientError
        try:
            self._client.head_object(
                Bucket=self._bucket, Key=self._full_key(key)
            )
            return True
        except ClientError:
            return False
        except Exception:
            return False

    def delete(self, key: str) -> None:
        try:
            self._client.delete_object(
                Bucket=self._bucket, Key=self._full_key(key)
            )
        except Exception as exc:
            raise StorageError(f"S3Storage.delete({key!r}) failed: {exc}") from exc

    def list(self, prefix: str = "") -> list[str]:
        full_prefix = self._full_key(prefix) if prefix else self._prefix
        try:
            paginator = self._client.get_paginator("list_objects_v2")
            keys = []
            for page in paginator.paginate(Bucket=self._bucket, Prefix=full_prefix):
                for obj in page.get("Contents", []):
                    keys.append(self._strip_prefix(obj["Key"]))
            return sorted(keys)
        except Exception as exc:
            raise StorageError(f"S3Storage.list({prefix!r}) failed: {exc}") from exc

    # ── File-path operations ──────────────────────────────────────────────────

    def upload_file(self, local_path: str, key: str) -> str:
        """
        Streams a local file to S3 using boto3's multipart upload.

        For files > 8MB, boto3 automatically uses multipart upload which
        is more efficient and reliable than a single PUT request.
        """
        if not Path(local_path).exists():
            raise FileNotFoundError(f"Source file not found: {local_path!r}")
        try:
            self._client.upload_file(
                Filename=local_path,
                Bucket=self._bucket,
                Key=self._full_key(key),
            )
        except Exception as exc:
            raise StorageError(f"S3Storage.upload_file({key!r}) failed: {exc}") from exc
        return key

    def download_file(self, key: str, local_path: str) -> None:
        """Downloads an S3 object to a local file using streaming."""
        from botocore.exceptions import ClientError
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        try:
            self._client.download_file(
                Bucket=self._bucket,
                Key=self._full_key(key),
                Filename=local_path,
            )
        except ClientError as exc:
            if exc.response["Error"]["Code"] in ("NoSuchKey", "404"):
                raise FileNotFoundError(f"S3 key not found: {key!r}")
            raise StorageError(f"S3Storage.download_file({key!r}) failed: {exc}") from exc

    def get_url(self, key: str, expires_in: int = 3600) -> str:
        """
        Returns a presigned HTTPS URL for temporary access to a private object.

        The URL embeds a temporary AWS signature and expires after expires_in
        seconds. No AWS credentials are needed to access the URL.
        """
        try:
            return self._client.generate_presigned_url(
                "get_object",
                Params={"Bucket": self._bucket, "Key": self._full_key(key)},
                ExpiresIn=expires_in,
            )
        except Exception as exc:
            raise StorageError(f"S3Storage.get_url({key!r}) failed: {exc}") from exc

    def copy(self, src_key: str, dst_key: str) -> None:
        """Server-side S3 copy — does not transfer bytes through the client."""
        try:
            self._client.copy(
                CopySource={"Bucket": self._bucket, "Key": self._full_key(src_key)},
                Bucket=self._bucket,
                Key=self._full_key(dst_key),
            )
        except Exception as exc:
            raise StorageError(f"S3Storage.copy({src_key!r} → {dst_key!r}) failed: {exc}") from exc

    def __repr__(self) -> str:
        return f"S3Storage(bucket={self._bucket!r}, prefix={self._prefix!r})"
