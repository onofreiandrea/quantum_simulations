"""Comprehensive tests for non-local (butterfly exchange) gates.

Setup: 4 qubits, chunk_size=4 (= 2^2).
  → qubits 0,1 are local  (bit in local index)
  → qubits 2,3 are non-local (bit in chunk index)
  → 4 chunks of 4 amplitudes each

Tests every case:
  A) 1-qubit non-local
  B) 2-qubit: qa local, qb non-local
  C) 2-qubit: qa non-local, qb local
  D) 2-qubit: both non-local
  E) mixed levels (local + non-local in same level)
  F) full circuits with small chunk_size vs ref_dense
"""
import tempfile
import numpy as np
import pytest

from wenbo_engine.kernel.ref_dense import simulate
from wenbo_engine.runner.single_node import run, collect_state
from wenbo_engine.tests.fixtures.circuits import bell_2q, ghz, qft

NQ = 4
CS = 4   # chunk_size = 2^2 → qubits 0,1 local


def _run_and_compare(cd, chunk_size=CS, atol=1e-6):
    ref = simulate(cd)
    with tempfile.TemporaryDirectory() as td:
        final = run(cd, td, chunk_size=chunk_size)
        got = collect_state(final)
    np.testing.assert_allclose(got, ref, atol=atol)


# ── A) 1-qubit non-local ────────────────────────────────────────────

class Test1QNonLocal:
    def test_h_on_q2(self):
        _run_and_compare({
            "number_of_qubits": NQ,
            "gates": [{"qubits": [2], "gate": "H"}],
        })

    def test_x_on_q3(self):
        cd = {"number_of_qubits": NQ, "gates": [{"qubits": [3], "gate": "X"}]}
        ref = simulate(cd)
        # X on qubit 3 → |0000⟩ → |1000⟩ → index 8
        assert abs(ref[8] - 1.0) < 1e-12
        _run_and_compare(cd)

    def test_t_on_q2(self):
        _run_and_compare({
            "number_of_qubits": NQ,
            "gates": [
                {"qubits": [0], "gate": "H"},
                {"qubits": [2], "gate": "H"},
                {"qubits": [2], "gate": "T"},
            ],
        })


# ── B) 2-qubit: qa local, qb non-local ─────────────────────────────

class Test2QLocalNonLocal:
    def test_cnot_q0_q2(self):
        _run_and_compare({
            "number_of_qubits": NQ,
            "gates": [
                {"qubits": [0], "gate": "H"},
                {"qubits": [0, 2], "gate": "CNOT"},
            ],
        })

    def test_cnot_q1_q3(self):
        _run_and_compare({
            "number_of_qubits": NQ,
            "gates": [
                {"qubits": [1], "gate": "H"},
                {"qubits": [1, 3], "gate": "CNOT"},
            ],
        })

    def test_cz_q0_q3(self):
        _run_and_compare({
            "number_of_qubits": NQ,
            "gates": [
                {"qubits": [0], "gate": "H"},
                {"qubits": [3], "gate": "H"},
                {"qubits": [0, 3], "gate": "CZ"},
            ],
        })

    def test_swap_q1_q2(self):
        _run_and_compare({
            "number_of_qubits": NQ,
            "gates": [
                {"qubits": [1], "gate": "X"},
                {"qubits": [1, 2], "gate": "SWAP"},
            ],
        })


# ── C) 2-qubit: qa non-local, qb local ─────────────────────────────

class Test2QNonLocalLocal:
    def test_cnot_q2_q0(self):
        _run_and_compare({
            "number_of_qubits": NQ,
            "gates": [
                {"qubits": [2], "gate": "H"},
                {"qubits": [2, 0], "gate": "CNOT"},
            ],
        })

    def test_cnot_q3_q1(self):
        _run_and_compare({
            "number_of_qubits": NQ,
            "gates": [
                {"qubits": [3], "gate": "H"},
                {"qubits": [3, 1], "gate": "CNOT"},
            ],
        })

    def test_cy_q2_q1(self):
        _run_and_compare({
            "number_of_qubits": NQ,
            "gates": [
                {"qubits": [2], "gate": "H"},
                {"qubits": [2, 1], "gate": "CY"},
            ],
        })


# ── D) 2-qubit: both non-local ──────────────────────────────────────

class Test2QBothNonLocal:
    def test_cnot_q2_q3(self):
        _run_and_compare({
            "number_of_qubits": NQ,
            "gates": [
                {"qubits": [2], "gate": "H"},
                {"qubits": [2, 3], "gate": "CNOT"},
            ],
        })

    def test_swap_q2_q3(self):
        _run_and_compare({
            "number_of_qubits": NQ,
            "gates": [
                {"qubits": [2], "gate": "X"},
                {"qubits": [2, 3], "gate": "SWAP"},
            ],
        })

    def test_cz_q3_q2(self):
        _run_and_compare({
            "number_of_qubits": NQ,
            "gates": [
                {"qubits": [2], "gate": "H"},
                {"qubits": [3], "gate": "H"},
                {"qubits": [3, 2], "gate": "CZ"},
            ],
        })


# ── E) mixed local + non-local in the same level ────────────────────

class TestMixedLevel:
    def test_h_all_qubits(self):
        """H on all 4 qubits → same level, qubits 0,1 local, 2,3 non-local."""
        _run_and_compare({
            "number_of_qubits": NQ,
            "gates": [{"qubits": [i], "gate": "H"} for i in range(NQ)],
        })

    def test_parallel_cnots_local_and_nonlocal(self):
        """Two CNOTs in one level: one local, one non-local."""
        _run_and_compare({
            "number_of_qubits": NQ,
            "gates": [
                {"qubits": [0], "gate": "H"},
                {"qubits": [2], "gate": "H"},
                # level 2: CNOT(0,1) local + CNOT(2,3) both-nonlocal
                {"qubits": [0, 1], "gate": "CNOT"},
                {"qubits": [2, 3], "gate": "CNOT"},
            ],
        })


# ── F) full circuits with small chunk_size ───────────────────────────

class TestFullCircuitsSmallChunk:
    def test_ghz4_cs4(self):
        _run_and_compare(ghz(4), chunk_size=4)

    def test_ghz4_cs2(self):
        """chunk_size=2 → only qubit 0 is local."""
        _run_and_compare(ghz(4), chunk_size=2)

    def test_ghz6_cs4(self):
        _run_and_compare(ghz(6), chunk_size=4)

    def test_qft4_cs4(self):
        _run_and_compare(qft(4), chunk_size=4)

    def test_qft4_cs2(self):
        _run_and_compare(qft(4), chunk_size=2)

    def test_bell_cs2(self):
        """Bell on 2 qubits, chunk_size=2 → qubit 1 is non-local."""
        cd = bell_2q()
        _run_and_compare(cd, chunk_size=2)


# ── G) pipeline runner with non-local gates ──────────────────────────

class TestPipelineNonLocal:
    def test_ghz4_pipeline(self):
        from wenbo_engine.runner.pipeline import run as prun
        cd = ghz(4)
        ref = simulate(cd)
        with tempfile.TemporaryDirectory() as td:
            final = prun(cd, td, chunk_size=4)
            got = collect_state(final)
        np.testing.assert_allclose(got, ref, atol=1e-6)

    def test_qft4_pipeline(self):
        from wenbo_engine.runner.pipeline import run as prun
        cd = qft(4)
        ref = simulate(cd)
        with tempfile.TemporaryDirectory() as td:
            final = prun(cd, td, chunk_size=4)
            got = collect_state(final)
        np.testing.assert_allclose(got, ref, atol=1e-6)
