"""Manifest: metadata for a committed state-vector version.

A manifest.json lives inside each version directory and records:
  - n_qubits, chunk_size, n_chunks, dtype
  - list of chunk file names
  - creation timestamp
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List


@dataclass
class Manifest:
    n_qubits: int
    chunk_size: int          # number of amplitudes per chunk
    n_chunks: int
    dtype: str = "complex64"
    chunks: List[str] = field(default_factory=list)
    created: float = field(default_factory=time.time)

    def validate(self):
        total = self.chunk_size * self.n_chunks
        expected = 1 << self.n_qubits
        if total != expected:
            raise ValueError(
                f"chunk_size*n_chunks={total} != 2^n_qubits={expected}"
            )
        if len(self.chunks) != self.n_chunks:
            raise ValueError(
                f"chunk list length {len(self.chunks)} != n_chunks {self.n_chunks}"
            )
        if self.dtype != "complex64":
            raise ValueError(f"unsupported dtype {self.dtype}")


def write_manifest_atomic(directory: str | Path, manifest: Manifest) -> Path:
    """Write manifest.json atomically (tmp → fsync → rename)."""
    manifest.validate()
    d = Path(directory)
    d.mkdir(parents=True, exist_ok=True)
    tmp = d / "manifest.json.tmp"
    final = d / "manifest.json"
    data = json.dumps(asdict(manifest), indent=2)
    with open(tmp, "w") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(str(tmp), str(final))
    return final


def read_manifest(directory: str | Path) -> Manifest:
    p = Path(directory) / "manifest.json"
    with open(p) as f:
        d = json.loads(f.read())
    m = Manifest(**d)
    m.validate()
    return m
