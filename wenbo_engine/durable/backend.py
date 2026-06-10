"""Abstract durable-backend interface.

A :class:`DurableBackend` is a flat key→bytes object store with atomic puts.
Keys are ``/``-separated POSIX-style strings (e.g.
``"<run_id>/generations/gen_000010/rank_0000/chunks/chunk_000000.bin"``).  The
durable managers build keys; backends only move bytes.

Atomicity contract
-------------------
``put`` must be *all-or-nothing*: a reader either sees the complete object or
does not see it at all (never a torn/partial object).  Filesystem backends
achieve this with temp-file + rename; object stores get it for free (a PUT is
atomic).  This is what lets the durable commit record be the single durability
point — every object it names is guaranteed whole once visible.

Every backend method is local-filesystem / network I/O only.  No numpy, no
MPI, no kernel imports — and nothing here runs on the hot gate-execution path.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass


@dataclass
class PutResult:
    """Outcome of a single :meth:`DurableBackend.put`."""
    key: str
    size_bytes: int
    checksum: str            # sha256 hex of the stored bytes


def sha256_bytes(data: bytes) -> str:
    """sha256 hex digest of ``data`` (matches recovery's file-based scheme)."""
    return hashlib.sha256(data).hexdigest()


class DurableBackend:
    """Atomic key→bytes durable store.

    Subclasses implement the five primitives below.  The default helpers
    (:meth:`put_file`, :meth:`get_to_file`) are written in terms of them so a
    backend only has to provide raw byte movement.
    """

    # ── primitives (override) ───────────────────────────────────────────

    def put(self, key: str, data: bytes) -> PutResult:
        """Atomically store ``data`` at ``key``; return its size + checksum."""
        raise NotImplementedError

    def get(self, key: str) -> bytes:
        """Return the bytes stored at ``key`` (raises if absent)."""
        raise NotImplementedError

    def exists(self, key: str) -> bool:
        """True iff an object is stored at ``key``."""
        raise NotImplementedError

    def list(self, prefix: str) -> list[str]:
        """Return every key under ``prefix`` (sorted)."""
        raise NotImplementedError

    def delete(self, key: str) -> None:
        """Remove ``key`` if present (idempotent)."""
        raise NotImplementedError

    # ── file helpers (built on the primitives) ──────────────────────────

    def put_file(self, key: str, path) -> PutResult:
        """Upload a local file's bytes to ``key`` atomically."""
        with open(path, "rb") as f:
            return self.put(key, f.read())

    def get_to_file(self, key: str, path) -> int:
        """Download ``key`` into a local file written atomically.

        Returns the number of bytes written.  The destination is written via a
        temp file + rename so a partial download is never visible at ``path``.
        """
        import os
        from pathlib import Path
        data = self.get(key)
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        with open(tmp, "wb") as f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(str(tmp), str(p))
        return len(data)

    def size(self, key: str) -> int:
        """Size in bytes of the object at ``key`` (raises if absent)."""
        return len(self.get(key))

    def checksum(self, key: str) -> str:
        """sha256 hex of the object at ``key`` (raises if absent)."""
        return sha256_bytes(self.get(key))
