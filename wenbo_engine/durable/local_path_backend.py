"""Filesystem-backed durable store (the REQUIRED backend).

``LocalPathBackend`` treats its ``root`` as a plain directory tree and maps a
key to ``root/<key>``.  Writes are atomic (temp file in the same directory +
fsync + ``os.replace``), matching the durability discipline used throughout
``wenbo_engine.storage`` / ``wenbo_engine.recovery``.

Because the root is just a filesystem path, a **JuiceFS mount** (or NFS, or a
second local disk) works through this backend unchanged: point ``root`` at the
mountpoint and every put/get goes through the mounted filesystem.

Pure stdlib — no numpy, no MPI, no kernel imports.
"""
from __future__ import annotations

import os
from pathlib import Path

from wenbo_engine.durable.backend import DurableBackend, PutResult, sha256_bytes


def _fsync_dir(d: Path) -> None:
    """Best-effort directory fsync so a rename is durable (no-op if disallowed)."""
    try:
        fd = os.open(str(d), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except (OSError, PermissionError):
        pass


class LocalPathBackend(DurableBackend):
    """Durable store rooted at a filesystem path (local / NFS / JuiceFS mount)."""

    def __init__(self, root: str | Path):
        if not root:
            raise ValueError("LocalPathBackend requires a non-empty root path")
        self.root = Path(root)

    # ── key ↔ path ──────────────────────────────────────────────────────

    def _path(self, key: str) -> Path:
        key = key.strip("/")
        if not key:
            raise ValueError("empty durable key")
        # Reject traversal so a malformed key cannot escape the root.
        p = (self.root / key).resolve()
        root = self.root.resolve()
        if root != p and root not in p.parents:
            raise ValueError(f"durable key escapes root: {key!r}")
        return self.root / key

    # ── primitives ──────────────────────────────────────────────────────

    def put(self, key: str, data: bytes) -> PutResult:
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(str(tmp), str(p))     # atomic publish
        _fsync_dir(p.parent)             # make the rename durable
        return PutResult(key=key, size_bytes=len(data),
                         checksum=sha256_bytes(data))

    def get(self, key: str) -> bytes:
        with open(self._path(key), "rb") as f:
            return f.read()

    def exists(self, key: str) -> bool:
        return self._path(key).is_file()

    def list(self, prefix: str) -> list[str]:
        base = self.root / prefix.strip("/")
        if not base.exists():
            return []
        keys: list[str] = []
        root = self.root
        if base.is_file():
            return [str(base.relative_to(root)).replace(os.sep, "/")]
        for p in base.rglob("*"):
            if p.is_file() and not p.name.endswith(".tmp"):
                keys.append(str(p.relative_to(root)).replace(os.sep, "/"))
        return sorted(keys)

    def delete(self, key: str) -> None:
        p = self._path(key)
        try:
            p.unlink()
        except FileNotFoundError:
            pass

    # ── file fast-paths (avoid loading whole chunks into RAM twice) ──────

    def put_file(self, key: str, path) -> PutResult:
        """Stream a local file to ``key`` (chunked copy + checksum, atomic)."""
        import hashlib
        src = Path(path)
        dst = self._path(key)
        dst.parent.mkdir(parents=True, exist_ok=True)
        tmp = dst.with_suffix(dst.suffix + ".tmp")
        h = hashlib.sha256()
        size = 0
        with open(src, "rb") as fin, open(tmp, "wb") as fout:
            while True:
                block = fin.read(1 << 20)
                if not block:
                    break
                fout.write(block)
                h.update(block)
                size += len(block)
            fout.flush()
            os.fsync(fout.fileno())
        os.replace(str(tmp), str(dst))
        _fsync_dir(dst.parent)
        return PutResult(key=key, size_bytes=size, checksum=h.hexdigest())

    def size(self, key: str) -> int:
        return self._path(key).stat().st_size

    def checksum(self, key: str) -> str:
        import hashlib
        h = hashlib.sha256()
        with open(self._path(key), "rb") as f:
            while True:
                block = f.read(1 << 20)
                if not block:
                    break
                h.update(block)
        return h.hexdigest()
