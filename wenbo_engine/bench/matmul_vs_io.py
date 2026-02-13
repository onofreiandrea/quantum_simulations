"""Benchmark: matmul kernel speed vs I/O bandwidth (side-by-side).

Key insight: matmul is usually faster than I/O by magnitudes on cluster
storage.  This benchmark quantifies the ratio so we can tune chunk_size
and buffer_depth to keep the CPU busy while waiting for I/O.
"""
from __future__ import annotations

import tempfile
import time
from pathlib import Path

import numpy as np

from wenbo_engine.kernel import gates as gmod
from wenbo_engine.kernel.cpu_batched import apply_1q, apply_2q
from wenbo_engine.storage.block_store import (
    write_chunk_atomic, read_chunk, DTYPE,
)


def _bench_io_rw(chunk_size: int, n_chunks: int = 8) -> dict:
    """Measure read+write throughput in MB/s."""
    bytes_per_chunk = chunk_size * np.dtype(DTYPE).itemsize
    data = np.random.randn(chunk_size).astype(np.float32).view(DTYPE)

    with tempfile.TemporaryDirectory() as td:
        p = Path(td)
        # write
        t0 = time.perf_counter()
        for i in range(n_chunks):
            write_chunk_atomic(p / f"c{i:06d}.bin", data)
        t_write = time.perf_counter() - t0

        # read
        t0 = time.perf_counter()
        for i in range(n_chunks):
            _ = read_chunk(p / f"c{i:06d}.bin")
        t_read = time.perf_counter() - t0

    total_mb = bytes_per_chunk * n_chunks / 1e6
    return {
        "write_MBs": total_mb / t_write,
        "read_MBs": total_mb / t_read,
        "rw_MBs": total_mb / (t_read + t_write) * 2,  # full cycle
        "io_time_per_chunk_ms": (t_read + t_write) / n_chunks * 1e3,
    }


def _bench_kernel(chunk_size: int, n_gates: int = 10, reps: int = 5) -> dict:
    """Measure kernel throughput: apply n_gates to one chunk, reps times."""
    U_h = gmod.gate_matrix("H", {})
    U_cx = gmod.gate_matrix("CNOT", {})
    chunk = np.zeros(chunk_size, dtype=np.complex128)
    chunk[0] = 1.0

    # warm up
    apply_1q(chunk.copy(), 0, U_h)

    # 1Q gates
    t0 = time.perf_counter()
    for _ in range(reps):
        for _ in range(n_gates):
            apply_1q(chunk, 0, U_h)
    t_1q = time.perf_counter() - t0

    # 2Q gates
    t0 = time.perf_counter()
    for _ in range(reps):
        for _ in range(n_gates):
            apply_2q(chunk, 0, 1, U_cx)
    t_2q = time.perf_counter() - t0

    bytes_per_gate = chunk_size * 16  # complex128
    total_1q = bytes_per_gate * n_gates * reps
    total_2q = bytes_per_gate * n_gates * reps

    return {
        "1q_GBs": total_1q / t_1q / 1e9,
        "2q_GBs": total_2q / t_2q / 1e9,
        "1q_time_per_gate_ms": t_1q / (n_gates * reps) * 1e3,
        "2q_time_per_gate_ms": t_2q / (n_gates * reps) * 1e3,
        "compute_time_per_chunk_ms": (t_1q + t_2q) / (2 * n_gates * reps) * 1e3,
    }


def bench_compare(chunk_sizes=None):
    """Run side-by-side comparison and print a table."""
    if chunk_sizes is None:
        chunk_sizes = [1 << e for e in [16, 18, 20, 22]]

    print("=" * 78)
    print("MATMUL vs I/O BENCHMARK  (hyperparameter exploration)")
    print("=" * 78)
    print()
    print(f"{'chunk_size':>12} {'chunk_MB':>9} {'I/O MB/s':>10} "
          f"{'1Q GB/s':>9} {'2Q GB/s':>9} {'ratio':>8} {'verdict':>20}")
    print("-" * 78)

    for cs in chunk_sizes:
        io = _bench_io_rw(cs)
        kern = _bench_kernel(cs)

        chunk_mb = cs * np.dtype(DTYPE).itemsize / 1e6
        io_speed = io["rw_MBs"]
        kern_speed_mb = (kern["1q_GBs"] + kern["2q_GBs"]) / 2 * 1e3  # GB→MB
        ratio = kern_speed_mb / max(io_speed, 1e-9)

        if ratio > 10:
            verdict = "I/O bound (fuse!)"
        elif ratio > 2:
            verdict = "I/O leaning"
        else:
            verdict = "balanced"

        print(f"{cs:>12,} {chunk_mb:>9.2f} {io_speed:>10.1f} "
              f"{kern['1q_GBs']:>9.2f} {kern['2q_GBs']:>9.2f} "
              f"{ratio:>7.1f}x {verdict:>20}")

    print()
    print("ratio = kernel_throughput / io_throughput")
    print("If ratio >> 1, we are I/O bound: increase buffer_depth,")
    print("  batch more levels per I/O pass (gate fusion), use larger chunks.")
    print("If ratio ~ 1, we are balanced: current settings are near optimal.")
    print()

    # Also print per-chunk timing
    print(f"{'chunk_size':>12} {'IO per chunk':>14} {'compute/gate':>14} "
          f"{'gates to match IO':>18}")
    print("-" * 62)
    for cs in chunk_sizes:
        io = _bench_io_rw(cs)
        kern = _bench_kernel(cs)
        io_ms = io["io_time_per_chunk_ms"]
        comp_ms = kern["compute_time_per_chunk_ms"]
        n_gates = int(io_ms / max(comp_ms, 1e-9)) if comp_ms > 0 else 999
        print(f"{cs:>12,} {io_ms:>12.3f}ms {comp_ms:>12.4f}ms {n_gates:>18}")

    print()
    print("'gates to match IO' = how many gates to apply per chunk to keep CPU")
    print("  busy for one full I/O cycle. Gate fusion batches levels to hit this.")


if __name__ == "__main__":
    bench_compare()
