"""Tests proving qubit reordering preserves simulation correctness.

For each circuit:
  1. Simulate original circuit → state_orig
  2. Reorder qubits → reordered_circuit + permutation
  3. Simulate reordered circuit → state_reord
  4. Apply permutation to state_orig → state_orig_permuted
  5. Assert state_reord == state_orig_permuted

This proves the reordered circuit produces the exact same physics,
just with relabeled qubits.
"""
import math

import numpy as np
import pytest

from wenbo_engine.circuit.io import validate_circuit_dict, levelize
from wenbo_engine.circuit.reorder import reorder_qubits, permute_state_vector
from wenbo_engine.kernel.ref_dense import simulate
from wenbo_engine.tests.fixtures.circuits import (
    bell_2q, ghz, qft, quest_random, hardware_efficient_ansatz,
)


# ── Correctness: reordered simulation matches original ───────────────

@pytest.mark.parametrize("circuit_fn,label", [
    (bell_2q, "bell_2q"),
    (lambda: ghz(4), "ghz_4"),
    (lambda: ghz(8), "ghz_8"),
    (lambda: qft(4), "qft_4"),
    (lambda: qft(6), "qft_6"),
    (lambda: quest_random(8, n_gates=20, seed=42), "quest_random_8q_20g"),
    (lambda: quest_random(10, n_gates=50, seed=42), "quest_random_10q_50g"),
    (lambda: quest_random(12, n_gates=50, seed=99), "quest_random_12q_50g"),
    (lambda: hardware_efficient_ansatz(6, layers=3), "hea_6q_3l"),
])
def test_reorder_preserves_state(circuit_fn, label):
    cd = circuit_fn()
    cd = validate_circuit_dict(cd)
    n = cd["number_of_qubits"]

    # Simulate original
    state_orig = simulate(cd)

    # Reorder and simulate
    cd_reord, perm = reorder_qubits(cd)
    cd_reord = validate_circuit_dict(cd_reord)
    state_reord = simulate(cd_reord)

    # Apply permutation to original state
    state_orig_permuted = permute_state_vector(state_orig, perm, n)

    # They must match
    np.testing.assert_allclose(
        state_reord, state_orig_permuted, atol=1e-12,
        err_msg=f"Reordered state does not match permuted original for {label}"
    )


# ── Norms must be identical ──────────────────────────────────────────

@pytest.mark.parametrize("n_gates", [10, 30, 50])
def test_reorder_preserves_norm(n_gates):
    cd = validate_circuit_dict(quest_random(10, n_gates=n_gates, seed=42))
    cd_reord, _ = reorder_qubits(cd)
    cd_reord = validate_circuit_dict(cd_reord)

    norm_orig = np.linalg.norm(simulate(cd))
    norm_reord = np.linalg.norm(simulate(cd_reord))

    assert abs(norm_orig - 1.0) < 1e-10
    assert abs(norm_reord - 1.0) < 1e-10
    assert abs(norm_orig - norm_reord) < 1e-12


# ── Gate classification improvement ─────────────────────────────────

def _classify_gates(circuit_dict, k, n_local_bits):
    """Count local, rank-nonlocal, MPI-nonlocal gates."""
    local, rank_nl, mpi_nl = 0, 0, 0
    for g in circuit_dict["gates"]:
        qs = g["qubits"]
        if all(q < k for q in qs):
            local += 1
        elif any((q - k) >= n_local_bits for q in qs if q >= k):
            mpi_nl += 1
        else:
            rank_nl += 1
    return local, rank_nl, mpi_nl


@pytest.mark.parametrize("n,n_gates,k,n_ranks", [
    (10, 10, 4, 2),
    (10, 30, 4, 2),
    (12, 20, 4, 4),
    (38, 10, 24, 4),
    (38, 50, 24, 4),
])
def test_reorder_reduces_nonlocal(n, n_gates, k, n_ranks):
    """Reordering should never increase the number of non-local gates."""
    p = int(math.log2(n_ranks))
    n_local_bits = n - k - p

    cd = validate_circuit_dict(quest_random(n, n_gates=n_gates, seed=42))
    cd_reord, _ = reorder_qubits(cd)
    cd_reord = validate_circuit_dict(cd_reord)

    orig_local, orig_rnl, orig_mpi = _classify_gates(cd, k, n_local_bits)
    reord_local, reord_rnl, reord_mpi = _classify_gates(cd_reord, k, n_local_bits)

    # Reordering should not increase non-local gates
    assert reord_mpi <= orig_mpi, \
        f"MPI gates increased: {orig_mpi} → {reord_mpi}"
    assert (reord_rnl + reord_mpi) <= (orig_rnl + orig_mpi), \
        f"Total non-local increased: {orig_rnl + orig_mpi} → {reord_rnl + reord_mpi}"


# ── Specific check: 38q 50-gate circuit has 0 MPI gates after reorder ─

def test_38q_50g_zero_mpi_gates():
    """Our target benchmark circuit should have 0 MPI gates after reorder."""
    n, k, p = 38, 24, 2
    n_local_bits = n - k - p

    cd = validate_circuit_dict(quest_random(n, n_gates=50, seed=42))
    cd_reord, perm = reorder_qubits(cd)
    cd_reord = validate_circuit_dict(cd_reord)

    _, _, orig_mpi = _classify_gates(cd, k, n_local_bits)
    _, _, reord_mpi = _classify_gates(cd_reord, k, n_local_bits)

    # Original has MPI gates
    assert orig_mpi > 0, "Original should have MPI gates"

    # Reordered has ZERO MPI gates
    assert reord_mpi == 0, \
        f"Expected 0 MPI gates after reorder, got {reord_mpi}"


# ── Permutation is valid ─────────────────────────────────────────────

def test_permutation_is_valid():
    cd = validate_circuit_dict(quest_random(10, n_gates=20, seed=42))
    _, perm = reorder_qubits(cd)

    n = cd["number_of_qubits"]
    # perm must be a bijection on {0, ..., n-1}
    assert set(perm.keys()) == set(range(n))
    assert set(perm.values()) == set(range(n))


# ── Reorder is idempotent on already-optimal circuits ────────────────

def test_reorder_all_local_circuit():
    """A circuit where all gates are already local should not be harmed."""
    # 2-qubit circuit: everything is local regardless
    cd = validate_circuit_dict(bell_2q())
    state_orig = simulate(cd)

    cd_reord, perm = reorder_qubits(cd)
    cd_reord = validate_circuit_dict(cd_reord)
    state_reord = simulate(cd_reord)

    state_orig_permuted = permute_state_vector(state_orig, perm, 2)
    np.testing.assert_allclose(state_reord, state_orig_permuted, atol=1e-12)
