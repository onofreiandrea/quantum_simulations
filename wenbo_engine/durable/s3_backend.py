"""Optional S3 durable backend (behind a lazy boto3 import).

This backend is OPTIONAL and not exercised by the test suite.  ``boto3`` is
imported lazily inside ``__init__`` so importing :mod:`wenbo_engine.durable`
never fails when boto3 is absent.

S3 PUTs are atomic at the object level (a GET either returns the whole object
or 404s), so the :class:`~wenbo_engine.durable.backend.DurableBackend`
all-or-nothing contract is satisfied without temp-key + rename.

Keys map directly to S3 object keys under an optional ``prefix`` inside the
bucket.  JuiceFS users should prefer the LocalPathBackend pointed at the mount;
this class is for talking to S3 directly.
"""
from __future__ import annotations

from wenbo_engine.durable.backend import DurableBackend, PutResult, sha256_bytes


class S3Backend(DurableBackend):
    """Durable store backed by an S3 bucket (requires boto3)."""

    def __init__(self, bucket: str | None = None, prefix: str = "",
                 *, root: str | None = None, client=None):
        """Create an S3 backend.

        ``bucket`` (or ``root`` as ``"bucket"`` / ``"bucket/prefix"``) names the
        target bucket; ``prefix`` is an optional key prefix.  ``client`` lets a
        caller (or a test) inject a preconfigured boto3 S3 client.
        """
        if bucket is None and root:
            # Allow root="bucket" or root="bucket/prefix" for CLI parity.
            part = root.strip("/")
            if "/" in part:
                bucket, rest = part.split("/", 1)
                prefix = f"{rest}/{prefix}".strip("/") if prefix else rest
            else:
                bucket = part
        if not bucket:
            raise ValueError("S3Backend requires a bucket (via bucket= or root=)")
        self.bucket = bucket
        self.prefix = prefix.strip("/")
        if client is not None:
            self._s3 = client
        else:
            try:
                import boto3
            except ImportError as e:  # pragma: no cover - optional dependency
                raise ImportError(
                    "S3Backend requires boto3 (pip install boto3) or an "
                    "injected client=") from e
            self._s3 = boto3.client("s3")

    def _full(self, key: str) -> str:
        key = key.strip("/")
        return f"{self.prefix}/{key}" if self.prefix else key

    def put(self, key: str, data: bytes) -> PutResult:
        self._s3.put_object(Bucket=self.bucket, Key=self._full(key), Body=data)
        return PutResult(key=key, size_bytes=len(data),
                         checksum=sha256_bytes(data))

    def get(self, key: str) -> bytes:
        obj = self._s3.get_object(Bucket=self.bucket, Key=self._full(key))
        return obj["Body"].read()

    def exists(self, key: str) -> bool:
        try:
            self._s3.head_object(Bucket=self.bucket, Key=self._full(key))
            return True
        except Exception:
            return False

    def list(self, prefix: str) -> list[str]:
        full = self._full(prefix)
        paginator = self._s3.get_paginator("list_objects_v2")
        strip = len(self.prefix) + 1 if self.prefix else 0
        keys: list[str] = []
        for page in paginator.paginate(Bucket=self.bucket, Prefix=full):
            for obj in page.get("Contents", []):
                k = obj["Key"]
                keys.append(k[strip:] if strip else k)
        return sorted(keys)

    def delete(self, key: str) -> None:
        self._s3.delete_object(Bucket=self.bucket, Key=self._full(key))
