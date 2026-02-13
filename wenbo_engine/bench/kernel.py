"""Benchmark: scalar vs batched kernel throughput."""
from __future__ import annotations

import time
import numpy as np

from wenbo_engine.kernel import gates as gmod
from wenbo_engine.kernel import cpu_scalar, cpu_batched


def _bench_1q(mod, chunk_size, gate_name="H", reps=20):
    U = gmod.gate_matrix(gate_name, {})
    chunk = np.zeros(chunk_size, dtype=np.complex128)
    chunk[0] = 1.0
    # warm up
    mod.apply_1q(chunk.copy(), 0, U)
    t0 = time.perf_counter()
    for _ in range(reps):
        mod.apply_1q(chunk, 0, U)
    dt = time.perf_counter() - t0
    bytes_touched = chunk_size * 16 * reps  # complex128 = 16 bytes
    return bytes_touched / dt / 1e9  # GB/s


def _bench_2q(mod, chunk_size, gate_name="CNOT", reps=20):
    U = gmod.gate_matrix(gate_name, {})
    chunk = np.zeros(chunk_size, dtype=np.complex128)
    chunk[0] = 1.0
    mod.apply_2q(chunk.copy(), 0, 1, U)
    t0 = time.perf_counter()
    for _ in range(reps):
        mod.apply_2q(chunk, 0, 1, U)
    dt = time.perf_counter() - t0
    bytes_touched = chunk_size * 16 * reps
    return bytes_touched / dt / 1e9


def bench_kernel(chunk_size: int = 1 << 20):
    print(f"chunk_size = {chunk_size}  ({chunk_size * 16 / 1e6:.1f} MB)")
    print(f"{'kernel':<12} {'gate':<8} {'GB/s':>8}")
    print("-" * 32)
    for name, mod in [("scalar", cpu_scalar), ("batched", cpu_batched)]:
        for g in ["H", "X", "T"]:
            tp = _bench_1q(mod, chunk_size, g)
            print(f"{name:<12} {g:<8} {tp:>8.2f}")
        for g in ["CNOT", "CZ"]:
            tp = _bench_2q(mod, chunk_size, g)
            print(f"{name:<12} {g:<8} {tp:>8.2f}")
    print()


if __name__ == "__main__":
    for e in [16, 18, 20]:
        bench_kernel(1 << e)
