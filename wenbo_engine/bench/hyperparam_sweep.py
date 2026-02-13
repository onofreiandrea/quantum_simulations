"""Benchmark: hyperparameter sweep over chunk_size, buffer_depth, fusion.

Chunk size, matmul workload, and buffer depth are hyperparameters that
vary according to I/O bandwidth and CPU computing power.  This script
explores the best configuration for a given machine.

This script sweeps over:
  - chunk_size: [2^16, 2^18, 2^20]
  - buffer_depth: [1, 2, 4, 8]
  - use_fusion: [False, True]
For a reference circuit, measuring end-to-end wall time.
"""
from __future__ import annotations

import tempfile
import time
from pathlib import Path

from wenbo_engine.runner.single_node import run as sn_run
from wenbo_engine.runner.pipeline import run as pl_run
from wenbo_engine.tests.fixtures.circuits import ghz, qft
from wenbo_engine.circuit.io import validate_circuit_dict, levelize
from wenbo_engine.circuit.fusion import fusion_stats


def _timed_run(runner, cd, chunk_size, **kwargs):
    with tempfile.TemporaryDirectory() as td:
        t0 = time.perf_counter()
        runner(cd, td, chunk_size=chunk_size, use_wal=False, **kwargs)
        return time.perf_counter() - t0


def sweep(
    circuit_fn,
    circuit_name: str,
    chunk_exponents: list[int] | None = None,
    buffer_depths: list[int] | None = None,
    reps: int = 1,
):
    """Run hyperparameter sweep and print results table."""
    if chunk_exponents is None:
        chunk_exponents = [16, 18, 20]
    if buffer_depths is None:
        buffer_depths = [1, 2, 4, 8]

    cd = validate_circuit_dict(circuit_fn())
    n = cd["number_of_qubits"]
    N = 1 << n
    n_gates = len(cd["gates"])
    levels = levelize(cd)

    print(f"\n{'=' * 80}")
    print(f"HYPERPARAMETER SWEEP: {circuit_name}")
    print(f"  n_qubits={n}, state_size={N}, gates={n_gates}, levels={len(levels)}")
    print(f"{'=' * 80}\n")

    # Show fusion stats for each chunk_size
    print("Fusion analysis:")
    for exp in chunk_exponents:
        cs = 1 << exp
        if cs > N:
            cs = N
        import math
        k = int(math.log2(cs))
        stats = fusion_stats(levels, k)
        print(f"  chunk_size=2^{exp}: {stats['io_reduction']}, "
              f"ops {stats['ops_before']}→{stats['ops_after']}")
    print()

    # Header
    print(f"{'runner':<12} {'chunk':>8} {'buf':>5} {'fusion':>7} "
          f"{'time(s)':>9} {'speedup':>8}")
    print("-" * 55)

    baseline = None
    results = []

    for exp in chunk_exponents:
        cs = 1 << exp
        if cs > N:
            cs = N

        for use_fusion in [False, True]:
            # Single-node runner
            times = []
            for _ in range(reps):
                t = _timed_run(sn_run, cd, cs, use_fusion=use_fusion)
                times.append(t)
            avg = sum(times) / len(times)
            if baseline is None:
                baseline = avg
            speedup = baseline / avg if avg > 0 else 0
            tag = "yes" if use_fusion else "no"
            print(f"{'single_node':<12} {f'2^{exp}':>8} {'--':>5} {tag:>7} "
                  f"{avg:>9.4f} {speedup:>7.2f}x")
            results.append(("single_node", exp, 0, use_fusion, avg))

            # Pipeline runner (sweep buffer_depth)
            for bd in buffer_depths:
                times = []
                for _ in range(reps):
                    t = _timed_run(pl_run, cd, cs,
                                   buffer_depth=bd, use_fusion=use_fusion)
                    times.append(t)
                avg = sum(times) / len(times)
                speedup = baseline / avg if avg > 0 else 0
                print(f"{'pipeline':<12} {f'2^{exp}':>8} {bd:>5} {tag:>7} "
                      f"{avg:>9.4f} {speedup:>7.2f}x")
                results.append(("pipeline", exp, bd, use_fusion, avg))

    # Best config
    if results:
        best = min(results, key=lambda r: r[4])
        print(f"\nBest config: runner={best[0]}, chunk_size=2^{best[1]}, "
              f"buffer_depth={best[2]}, fusion={best[3]}, time={best[4]:.4f}s")


if __name__ == "__main__":
    # Use moderately sized circuits for meaningful sweeps
    sweep(lambda: qft(8), "QFT-8", chunk_exponents=[4, 6, 8])
    sweep(lambda: ghz(10), "GHZ-10", chunk_exponents=[6, 8, 10])
    sweep(lambda: qft(10), "QFT-10", chunk_exponents=[6, 8, 10])
