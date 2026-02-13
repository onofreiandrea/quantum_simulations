"""Block store: read / write dense complex64 chunk files on disk."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from wenbo_engine.storage.manifest import Manifest, write_manifest_atomic

DTYPE = np.complex64


def chunk_filename(idx: int) -> str:
    return f"chunk_{idx:06d}.bin"


def write_chunk_atomic(path: str | Path, data: np.ndarray) -> None:
    """Write a single chunk file atomically."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".tmp")
    buf = np.ascontiguousarray(data, dtype=DTYPE).tobytes()
    with open(tmp, "wb") as f:
        f.write(buf)
        f.flush()
        os.fsync(f.fileno())
    os.replace(str(tmp), str(p))


def read_chunk(path: str | Path) -> np.ndarray:
    return np.fromfile(str(path), dtype=DTYPE)


def init_zero_state(
    directory: str | Path,
    n_qubits: int,
    chunk_size: int = 1 << 20,
) -> Manifest:
    """Create |0…0⟩ on disk and commit manifest."""
    N = 1 << n_qubits
    if N % chunk_size != 0:
        raise ValueError("2^n_qubits must be divisible by chunk_size")
    n_chunks = N // chunk_size
    d = Path(directory)
    chunks_dir = d / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)

    chunk_names = []
    for c in range(n_chunks):
        arr = np.zeros(chunk_size, dtype=DTYPE)
        if c == 0:
            arr[0] = 1.0  # |0…0⟩
        name = chunk_filename(c)
        write_chunk_atomic(chunks_dir / name, arr)
        chunk_names.append(name)

    m = Manifest(
        n_qubits=n_qubits,
        chunk_size=chunk_size,
        n_chunks=n_chunks,
        chunks=chunk_names,
    )
    write_manifest_atomic(d, m)
    return m
