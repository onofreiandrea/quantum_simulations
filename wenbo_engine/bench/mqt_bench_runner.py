"""MQT Bench benchmark runner.

Generates circuits via MQT Bench, verifies correctness against Qiskit
Statevector for small n, measures performance for larger n.
"""
from __future__ import annotations

import tempfile
import time

import numpy as np

from wenbo_engine.kernel.ref_dense import simulate
from wenbo_engine.runner.single_node import run as sn_run, collect_state

# Qiskit / MQT imports (optional)
try:
    from qiskit import transpile
    from qiskit.quantum_info import Statevector
    from mqt.bench import get_benchmark
    from mqt.bench.benchmark_generation import BenchmarkLevel
    from wenbo_engine.circuit.import_qiskit import qiskit_to_dict, SUPPORTED_BASIS
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False


BENCHMARKS = [
    # Basic / structural  —  lightweight gates, tests qubit-scaling up to 20
    ("ghz",                      [3, 4, 5, 6, 8, 10, 14, 18, 20]),
    ("graphstate",               [3, 4, 5, 6, 8, 10, 14, 18, 20]),
    ("wstate",                   [3, 4, 5, 6, 8, 10, 14, 18, 20]),
    ("bv",                       [3, 4, 5, 6, 8, 10, 14, 18, 20]),
    ("dj",                       [3, 4, 5, 6, 8, 10, 14, 18, 20]),
    # QFT family  —  O(n^2) gates, up to 16 qubits (~3500 gates)
    ("qft",                      [3, 4, 5, 6, 8, 10, 12, 14, 16]),
    ("qftentangled",             [3, 4, 5, 6, 8, 10, 12, 14, 16]),
    ("qpeexact",                 [3, 4, 5, 6, 8, 10, 12, 14]),
    ("qpeinexact",               [3, 4, 5, 6, 8, 10, 12, 14]),
    # Algorithms  —  heavy gate counts, keep n moderate
    ("grover",                   [3, 4, 5, 6, 8, 10]),
    ("ae",                       [3, 4, 5, 6, 8, 10]),
    ("hhl",                      [3, 4]),
    ("qwalk",                    [3, 4]),
    # Arithmetic
    ("half_adder",               [3, 4]),
    ("full_adder",               [4, 5]),
    ("cdkm_ripple_carry_adder",  [4, 5]),
    ("vbe_ripple_carry_adder",   [4, 5]),
    ("draper_qft_adder",         [4, 5]),
    ("modular_adder",            [4, 5]),
    ("multiplier",               [4, 5]),
    ("rg_qft_multiplier",        [4, 5]),
    ("hrs_cumulative_multiplier",[5]),
    # Cryptography  —  395K gates at n=18
    ("shor",                     [18]),
    # Finance
    ("bmw_quark_cardinality",    [3, 4]),
    ("bmw_quark_copula",         [4, 5]),
    # Variational / ML  —  scales easily to 20 qubits
    ("qnn",                      [3, 4, 5, 6, 8, 10, 12, 14]),
    ("vqe_real_amp",             [3, 4, 5, 6, 8, 10, 14, 18, 20]),
    ("vqe_su2",                  [3, 4, 5]),
    ("vqe_two_local",            [3, 4, 5, 6, 8, 10, 14, 16]),
    ("qaoa",                     [3, 4, 5]),
    # Random  —  dense circuits, stress-test gate variety
    ("randomcircuit",            [3, 4, 5, 6, 8, 10, 12, 14]),
]
CORRECTNESS_MAX_N = 20  # ref_dense handles n≤20 fine (~4 MB state at n=18)


def _convert(bench_name: str, n: int):
    qc = get_benchmark(bench_name, circuit_size=n, level=BenchmarkLevel.INDEP)
    qc.remove_final_measurements(inplace=True)
    qc_t = transpile(qc, basis_gates=SUPPORTED_BASIS, optimization_level=0)
    cd = qiskit_to_dict(qc_t)
    return qc, cd


def _check_correctness(qc, cd, atol=1e-6):
    ours = simulate(cd)
    ref = np.array(Statevector(qc).data)
    overlap = float(np.abs(np.vdot(ref, ours)))
    return overlap > 1.0 - atol, overlap


def _bench_perf(cd, chunk_size=0):
    n = cd["number_of_qubits"]
    N = 1 << n
    if chunk_size == 0:
        chunk_size = N
    with tempfile.TemporaryDirectory() as td:
        t0 = time.perf_counter()
        sn_run(cd, td, chunk_size=chunk_size)
        dt = time.perf_counter() - t0
    total_bytes = N * 8  # complex64 on disk
    mb_s = total_bytes * len(cd["gates"]) / dt / 1e6
    return dt, mb_s


def main():
    if not HAS_DEPS:
        print("ERROR: qiskit and/or mqt.bench not installed.")
        print("  pip install qiskit mqt.bench")
        return

    header = f"{'benchmark':<14} {'n':>3} {'#gates':>7} {'correct':>8} {'time(s)':>8} {'MB/s':>8} {'chunk':>10}"
    print(header)
    print("-" * len(header))

    for bench_name, sizes in BENCHMARKS:
        for n in sizes:
            try:
                qc, cd = _convert(bench_name, n)
            except Exception as e:
                print(f"{bench_name:<14} {n:>3}  SKIP ({e})")
                continue

            n_gates = len(cd["gates"])
            cs = 1 << n

            correct_str = ""
            if n <= CORRECTNESS_MAX_N:
                ok, overlap = _check_correctness(qc, cd)
                correct_str = "PASS" if ok else f"FAIL({overlap:.4f})"

            dt, mb_s = _bench_perf(cd, chunk_size=cs)
            print(
                f"{bench_name:<14} {n:>3} {n_gates:>7} {correct_str:>8} "
                f"{dt:>8.4f} {mb_s:>8.1f} {cs:>10}"
            )


if __name__ == "__main__":
    main()
