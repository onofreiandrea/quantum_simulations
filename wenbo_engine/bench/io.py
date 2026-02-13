"""Benchmark: sequential I/O bandwidth (read + write chunks)."""
from __future__ import annotations

import tempfile
import time
from pathlib import Path

import numpy as np

from wenbo_engine.storage.block_store import write_chunk_atomic, read_chunk, DTYPE


def bench_io(chunk_size: int = 1 << 20, n_chunks: int = 16):
    """Measure sequential read/write throughput in MB/s."""
    bytes_per_chunk = chunk_size * np.dtype(DTYPE).itemsize
    total_bytes = bytes_per_chunk * n_chunks

    with tempfile.TemporaryDirectory() as td:
        p = Path(td)
        data = np.random.randn(chunk_size).astype(np.float32).view(DTYPE)

        # write
        t0 = time.perf_counter()
        for i in range(n_chunks):
            write_chunk_atomic(p / f"chunk_{i:06d}.bin", data)
        t_write = time.perf_counter() - t0

        # read
        t0 = time.perf_counter()
        for i in range(n_chunks):
            _ = read_chunk(p / f"chunk_{i:06d}.bin")
        t_read = time.perf_counter() - t0

    mb = total_bytes / 1e6
    print(f"chunk_size={chunk_size}  n_chunks={n_chunks}  total={mb:.1f} MB")
    print(f"  write: {t_write:.3f}s  → {mb/t_write:.1f} MB/s")
    print(f"  read:  {t_read:.3f}s  → {mb/t_read:.1f} MB/s")
    return {"write_MBs": mb / t_write, "read_MBs": mb / t_read}


if __name__ == "__main__":
    for cs_exp in [18, 20, 22]:
        bench_io(chunk_size=1 << cs_exp, n_chunks=16)
        print()
