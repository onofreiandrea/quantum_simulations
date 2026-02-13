"""Benchmark: end-to-end circuit simulation throughput."""
from __future__ import annotations

import tempfile
import time

import numpy as np

from wenbo_engine.runner.single_node import run as sn_run
from wenbo_engine.runner.pipeline import run as pl_run
from wenbo_engine.tests.fixtures.circuits import ghz, qft


def bench_e2e(circ_fn, circ_name: str, chunk_size: int = 0):
    cd = circ_fn()
    n = cd["number_of_qubits"]
    N = 1 << n
    if chunk_size == 0:
        chunk_size = N
    total_bytes = N * 8  # complex64 on disk

    results = {}
    for runner_name, runner in [("single_node", sn_run), ("pipeline", pl_run)]:
        with tempfile.TemporaryDirectory() as td:
            t0 = time.perf_counter()
            runner(cd, td, chunk_size=chunk_size)
            dt = time.perf_counter() - t0
        mb_s = total_bytes * len(cd["gates"]) / dt / 1e6
        results[runner_name] = {"time": dt, "MBs": mb_s}
        print(f"  {runner_name:<14} {dt:.4f}s  {mb_s:.1f} MB/s")
    return results


if __name__ == "__main__":
    for name, fn, nq in [
        ("GHZ-6", lambda: ghz(6), 6),
        ("GHZ-10", lambda: ghz(10), 10),
        ("QFT-6", lambda: qft(6), 6),
        ("QFT-8", lambda: qft(8), 8),
    ]:
        print(f"\n{name}  (n={nq}, gates={len(fn()['gates'])})")
        bench_e2e(fn, name)
