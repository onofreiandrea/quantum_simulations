"""Tests for circuit staging (Atlas ILP, heuristic, greedy).

Verifies:
  1. Staging produces correct final state (vs ref_dense oracle).
  2. permute_state correctly reorders amplitudes.
  3. Staging reduces step count compared to batch_levels.
  4. All-local circuits are unaffected by staging.
  5. End-to-end with the runner (use_staging=True).
  6. Insular qubit detection is correct.
  7. ILP produces optimal (fewest) stages when available.
"""
import tempfile

import numpy as np
import pytest

from wenbo_engine.circuit.staging import (
    atlas_stages, permute_state, staging_stats, QubitMap,
    non_insular_qubits, HAS_PULP,
    _compute_local_qubits_heuristic,
)
from wenbo_engine.circuit.fusion import batch_levels
from wenbo_engine.circuit.io import validate_circuit_dict, levelize
from wenbo_engine.kernel.ref_dense import simulate
from wenbo_engine.runner.single_node import run, collect_state

# Methods to test (ILP only if PuLP is available).
_METHODS = ["heuristic", "greedy"]
if HAS_PULP:
    _METHODS.append("ilp")


# ── helpers ──────────────────────────────────────────────────────────

def _run_staged(cd, chunk_size, method="heuristic"):
    """Run with staging, collect state with permutation applied."""
    with tempfile.TemporaryDirectory() as td:
        final = run(cd, td, chunk_size=chunk_size, use_wal=False,
                     use_staging=True, staging_method=method)
        return collect_state(final, apply_permutation=True, work_dir=td)


def _run_baseline(cd, chunk_size):
    """Run without staging."""
    with tempfile.TemporaryDirectory() as td:
        final = run(cd, td, chunk_size=chunk_size, use_wal=False,
                     use_staging=False)
        return collect_state(final)


# ── Test: insular qubit detection ────────────────────────────────────

def test_insular_z():
    """Z gate: target qubit is insular (diagonal), returns empty."""
    assert non_insular_qubits({"gate": "Z", "qubits": [3], "params": {}}) == []


def test_insular_s():
    assert non_insular_qubits({"gate": "S", "qubits": [0], "params": {}}) == []


def test_insular_t():
    assert non_insular_qubits({"gate": "T", "qubits": [2], "params": {}}) == []


def test_insular_cz():
    """CZ: fully diagonal (sparse), no qubit needs to be local."""
    assert non_insular_qubits({"gate": "CZ", "qubits": [0, 3], "params": {}}) == []


def test_insular_cr():
    """CR: fully diagonal (sparse), no qubit needs to be local."""
    assert non_insular_qubits({"gate": "CR", "qubits": [1, 4], "params": {"k": 2}}) == []


def test_non_insular_h():
    """H gate: target qubit is non-insular (not diagonal)."""
    assert non_insular_qubits({"gate": "H", "qubits": [0], "params": {}}) == [0]


def test_non_insular_cnot():
    """CNOT: both qubits are non-insular."""
    assert non_insular_qubits({"gate": "CNOT", "qubits": [0, 1], "params": {}}) == [0, 1]


def test_non_insular_swap():
    assert non_insular_qubits({"gate": "SWAP", "qubits": [2, 5], "params": {}}) == [2, 5]


# ── Test: permute_state ──────────────────────────────────────────────

def test_permute_identity():
    """Identity mapping -> no change."""
    state = np.array([1, 0, 0, 0], dtype=np.complex128)
    result = permute_state(state, [0, 1])
    np.testing.assert_array_equal(result, state)


def test_permute_swap_2q():
    """Swap qubits 0 and 1 in a 2-qubit state."""
    state = np.zeros(4, dtype=np.complex128)
    state[2] = 1.0  # |01> (q0=0, q1=1)
    phys_state = np.zeros(4, dtype=np.complex128)
    phys_state[1] = 1.0
    result = permute_state(phys_state, [1, 0])
    np.testing.assert_allclose(result, state, atol=1e-12)


def test_permute_3q():
    """Cyclic permutation of 3 qubits."""
    log_to_phys = [2, 0, 1]
    logical_state = np.zeros(8, dtype=np.complex128)
    logical_state[1] = 1.0
    phys_state = np.zeros(8, dtype=np.complex128)
    phys_state[4] = 1.0
    result = permute_state(phys_state, log_to_phys)
    np.testing.assert_allclose(result, logical_state, atol=1e-12)


# ── Test: QubitMap ───────────────────────────────────────────────────

def test_qubit_map_swap():
    qm = QubitMap(4)
    assert qm.phys(0) == 0
    qm.swap_phys(0, 3)
    assert qm.phys(0) == 3
    assert qm.phys(3) == 0
    assert qm.logical(0) == 3
    assert qm.logical(3) == 0


def test_qubit_map_local_set():
    qm = QubitMap(6)
    assert qm.local_set(3) == {0, 1, 2}
    qm.swap_phys(1, 4)
    assert qm.local_set(3) == {0, 4, 2}


# ── Test: heuristic produces valid stage sets ────────────────────────

def test_heuristic_basic():
    """Heuristic returns valid local-qubit sets for a simple circuit."""
    gates = [
        {"qubits": [0], "gate": "H", "params": {}},
        {"qubits": [3], "gate": "H", "params": {}},
        {"qubits": [0, 3], "gate": "CNOT", "params": {}},
    ]
    stages = _compute_local_qubits_heuristic(gates, n=4, k=2)
    assert len(stages) >= 1
    for s in stages:
        assert len(s) == 2
        assert all(0 <= q < 4 for q in s)


# ── Test: staging correctness (vs ref_dense), parametrized by method ─

@pytest.mark.parametrize("method", _METHODS)
def test_staging_correctness_4q(method):
    """4 qubits, chunk_size=4 (k=2), gates on qubits 0-3 forcing non-local."""
    cd = {"number_of_qubits": 4, "gates": [
        {"qubits": [0], "gate": "H"},
        {"qubits": [2], "gate": "H"},
        {"qubits": [0, 2], "gate": "CNOT"},
        {"qubits": [1, 3], "gate": "CNOT"},
    ]}
    ref = simulate(cd)
    got = _run_staged(cd, chunk_size=4, method=method)
    np.testing.assert_allclose(got, ref, atol=1e-6)


@pytest.mark.parametrize("method", _METHODS)
def test_staging_correctness_5q(method):
    """5 qubits, various gates, chunk_size=4 (k=2)."""
    cd = {"number_of_qubits": 5, "gates": [
        {"qubits": [0], "gate": "H"},
        {"qubits": [1], "gate": "H"},
        {"qubits": [2], "gate": "H"},
        {"qubits": [3], "gate": "X"},
        {"qubits": [4], "gate": "Y"},
        {"qubits": [0, 3], "gate": "CNOT"},
        {"qubits": [1, 4], "gate": "CZ"},
        {"qubits": [2, 3], "gate": "SWAP"},
    ]}
    ref = simulate(cd)
    got = _run_staged(cd, chunk_size=4, method=method)
    np.testing.assert_allclose(got, ref, atol=1e-6)


@pytest.mark.parametrize("method", _METHODS)
def test_staging_matches_baseline(method):
    """Staged and non-staged runs produce the same result."""
    cd = {"number_of_qubits": 4, "gates": [
        {"qubits": [0], "gate": "H"},
        {"qubits": [1], "gate": "H"},
        {"qubits": [2], "gate": "H"},
        {"qubits": [3], "gate": "H"},
        {"qubits": [0, 2], "gate": "CNOT"},
        {"qubits": [1, 3], "gate": "CNOT"},
        {"qubits": [0, 3], "gate": "CZ"},
    ]}
    ref = simulate(cd)
    staged = _run_staged(cd, chunk_size=4, method=method)
    baseline = _run_baseline(cd, chunk_size=4)
    np.testing.assert_allclose(staged, ref, atol=1e-6)
    np.testing.assert_allclose(baseline, ref, atol=1e-6)


# ── Test: all-local circuit unaffected ───────────────────────────────

@pytest.mark.parametrize("method", _METHODS)
def test_staging_all_local(method):
    """When all gates are local, staging = baseline (identity mapping)."""
    cd = {"number_of_qubits": 4, "gates": [
        {"qubits": [0], "gate": "H"},
        {"qubits": [1], "gate": "X"},
    ]}
    steps, mapping = atlas_stages(cd, k=2, method=method)
    assert mapping == [0, 1, 2, 3]
    ref = simulate(cd)
    got = _run_staged(cd, chunk_size=4, method=method)
    np.testing.assert_allclose(got, ref, atol=1e-6)


# ── Test: step count reduction ───────────────────────────────────────

@pytest.mark.parametrize("method", ["heuristic", "greedy"])
def test_staging_reduces_steps(method):
    """Staging reduces I/O passes when circuit has clustered qubit phases."""
    cd = {"number_of_qubits": 8, "gates": [
        # Phase 1: deep chain on qubits 4,5,6 (non-local by default)
        {"qubits": [4], "gate": "H"},
        {"qubits": [5], "gate": "H"},
        {"qubits": [6], "gate": "H"},
        {"qubits": [4, 5], "gate": "CNOT"},
        {"qubits": [5, 6], "gate": "CNOT"},
        {"qubits": [4, 6], "gate": "CZ"},
        {"qubits": [4], "gate": "T"},
        {"qubits": [5], "gate": "S"},
        {"qubits": [4, 5], "gate": "CNOT"},
        {"qubits": [5, 6], "gate": "CNOT"},
        # Phase 2: deep chain on qubits 0,1,2 (local by default)
        {"qubits": [0], "gate": "H"},
        {"qubits": [1], "gate": "H"},
        {"qubits": [2], "gate": "H"},
        {"qubits": [0, 1], "gate": "CNOT"},
        {"qubits": [1, 2], "gate": "CNOT"},
        {"qubits": [0, 2], "gate": "CZ"},
    ]}
    stats = staging_stats(cd, k=3, method=method)
    assert stats["staged_steps"] <= stats["baseline_steps"], (
        f"staged={stats['staged_steps']} > baseline={stats['baseline_steps']}"
    )


# ── Test: ILP optimal (when available) ───────────────────────────────

@pytest.mark.skipif(not HAS_PULP, reason="PuLP not installed")
def test_ilp_correctness_4q():
    """ILP method produces correct result for a 4-qubit circuit."""
    cd = {"number_of_qubits": 4, "gates": [
        {"qubits": [0], "gate": "H"},
        {"qubits": [2], "gate": "H"},
        {"qubits": [0, 2], "gate": "CNOT"},
        {"qubits": [1, 3], "gate": "CNOT"},
    ]}
    ref = simulate(cd)
    got = _run_staged(cd, chunk_size=4, method="ilp")
    np.testing.assert_allclose(got, ref, atol=1e-6)


@pytest.mark.skipif(not HAS_PULP, reason="PuLP not installed")
def test_ilp_leq_heuristic_steps():
    """ILP should produce fewer or equal stages than the heuristic."""
    cd = {"number_of_qubits": 8, "gates": [
        {"qubits": [4], "gate": "H"},
        {"qubits": [5], "gate": "H"},
        {"qubits": [6], "gate": "H"},
        {"qubits": [4, 5], "gate": "CNOT"},
        {"qubits": [5, 6], "gate": "CNOT"},
        {"qubits": [4, 6], "gate": "CZ"},
        {"qubits": [0], "gate": "H"},
        {"qubits": [1], "gate": "H"},
        {"qubits": [0, 1], "gate": "CNOT"},
        {"qubits": [1, 2], "gate": "CNOT"},
    ]}
    ilp_stats = staging_stats(cd, k=3, method="ilp")
    heur_stats = staging_stats(cd, k=3, method="heuristic")
    assert ilp_stats["staged_steps"] <= heur_stats["staged_steps"], (
        f"ILP={ilp_stats['staged_steps']} > heuristic={heur_stats['staged_steps']}"
    )


def test_staging_stats_output():
    """staging_stats returns expected keys."""
    cd = {"number_of_qubits": 4, "gates": [
        {"qubits": [0], "gate": "H"},
        {"qubits": [0, 2], "gate": "CNOT"},
    ]}
    stats = staging_stats(cd, k=2)
    assert "baseline_steps" in stats
    assert "staged_steps" in stats
    assert "reduction" in stats
