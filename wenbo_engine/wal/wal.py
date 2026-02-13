"""Write-ahead log — atomic JSON file, cluster-safe.

Tracks double-buffer simulation progress:
  - Which buffer (a/b) holds the latest committed state
  - How many steps have been completed

Uses tmp + fsync + rename.  Works on local FS, BeeGFS, NFS, Lustre.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def _circuit_hash(circuit_dict: dict) -> str:
    """Deterministic hash of the circuit for identity verification."""
    from wenbo_engine.circuit.io import validate_circuit_dict
    normalized = validate_circuit_dict(circuit_dict)
    raw = json.dumps(normalized, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


class WAL:
    """Double-buffer WAL.

    File layout (wal.json):
    {
      "circuit_hash": "abc123...",
      "committed_buf": "a",
      "done_steps": 0
    }

    Recovery: read done_steps → resume from that step.
    committed_buf → use as source; the other buffer is the destination.
    """

    def __init__(self, path: str | Path, circuit_dict: dict | None = None):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._chash = _circuit_hash(circuit_dict) if circuit_dict else None

        if self.path.exists():
            self._data = self._read()
            if self._chash and self._data.get("circuit_hash"):
                if self._data["circuit_hash"] != self._chash:
                    raise ValueError(
                        "WAL circuit hash mismatch — different circuit? "
                        f"WAL={self._data['circuit_hash']} vs new={self._chash}"
                    )
        else:
            self._data = {
                "circuit_hash": self._chash or "",
                "committed_buf": "a",
                "done_steps": 0,
            }
            self._flush()

    # ── atomic read / write ──────────────────────────────────────────

    def _read(self) -> dict:
        with open(self.path) as f:
            return json.loads(f.read())

    def _flush(self) -> None:
        tmp = self.path.with_suffix(".tmp")
        raw = json.dumps(self._data, indent=2)
        with open(tmp, "w") as f:
            f.write(raw)
            f.flush()
            os.fsync(f.fileno())
        os.replace(str(tmp), str(self.path))

    # ── public API ───────────────────────────────────────────────────

    @property
    def committed_buf(self) -> str:
        return self._data.get("committed_buf", "a")

    @property
    def done_steps(self) -> int:
        return self._data.get("done_steps", 0)

    def commit_step(self, step_idx: int, new_buf: str) -> None:
        """Mark a step as fully committed."""
        self._data["committed_buf"] = new_buf
        self._data["done_steps"] = step_idx + 1
        self._flush()

    def close(self) -> None:
        """No-op (no file handles to close).  Kept for API compat."""
        pass
