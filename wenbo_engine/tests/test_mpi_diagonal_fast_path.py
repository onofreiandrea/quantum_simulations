"""Diagonal MPI-nonlocal classification + local fast path."""
import numpy as np
import pytest

from wenbo_engine.kernel import gates as gmod
from wenbo_engine.kernel.ref_dense import simulate
from wenbo_engine.mpi.diagonal_nonlocal import (
    is_diagonal, is_permutation, classify_nonlocal_gate,
    apply_diagonal_nonlocal_chunk,
)
DTYPE = np.complex64


def _U(name, **params):
    return gmod.gate_matrix(name, params).astype(DTYPE)


# ── 4,5,6,7: classification ─────────────────────────────────────────────

def test_cr_k_classified_diagonal():
    for k in (1, 2, 3, 4, 5):
        kind, req = classify_nonlocal_gate(_U("CR", k=k))
        assert kind == "diagonal" and req is False, k


def test_cz_classified_diagonal():
    kind, req = classify_nonlocal_gate(_U("CZ"))
    assert kind == "diagonal" and req is False


@pytest.mark.parametrize("g,p", [("RZ", {"theta": 0.3}), ("T", {}), ("S", {}),
                                 ("Z", {})])
def test_phase_gates_classified_diagonal(g, p):
    kind, req = classify_nonlocal_gate(_U(g, **p))
    assert kind == "diagonal" and req is False


@pytest.mark.parametrize("g,p", [("RX", {"theta": 0.3}), ("RY", {"theta": 0.3}),
                                 ("H", {})])
def test_mixing_gates_classified_true_mixing(g, p):
    kind, req = classify_nonlocal_gate(_U(g, **p))
    assert kind == "true_mixing" and req is True


def test_cnot_classified_permutation():
    kind, req = classify_nonlocal_gate(_U("CNOT"))
    assert kind == "permutation" and req is True
    assert is_permutation(_U("CNOT")) and not is_diagonal(_U("CNOT"))


# ── 8: diagonal fast path matches the dense reference ───────────────────

def test_diagonal_fast_path_matches_reference():
    # n=4, chunk_size=4 (k=2), 2 ranks -> n_local_bits=1; q3 = rank bit, q2 = chunk bit
    n, k, ranks = 4, 2, 2
    p = 1; nlb = n - k - p; cs = 1 << k; ncr = (1 << (n - k)) // ranks
    circ = {"number_of_qubits": n, "gates": [
        {"qubits": [0], "gate": "H"}, {"qubits": [1], "gate": "H"},
        {"qubits": [2], "gate": "H"}, {"qubits": [3], "gate": "H"},
        {"qubits": [3], "gate": "RZ", "params": {"theta": 0.7}},   # diag rank bit
        {"qubits": [3, 0], "gate": "CZ"},                          # diag rank+local
        {"qubits": [3, 2], "gate": "CR", "params": {"k": 3}},      # diag rank+chunk
        {"qubits": [3], "gate": "T"}]}
    ref = simulate(circ).astype(DTYPE)
    state = simulate({"number_of_qubits": n, "gates": circ["gates"][:4]}).astype(DTYPE)
    for g in circ["gates"][4:]:
        qs = g["qubits"]; U = _U(g["gate"], **g.get("params", {}))
        assert classify_nonlocal_gate(U)[0] == "diagonal"
        for r in range(ranks):
            for ci in range(ncr):
                base = (r << (k + nlb)) | (ci << k)
                ch = state[base:base + cs].copy()
                apply_diagonal_nonlocal_chunk(ch, ci, qs, U, r, k, nlb)
                state[base:base + cs] = ch
    assert np.allclose(state, ref, atol=1e-5)


def test_diagonal_preserves_dtype():
    ch = np.ones(4, dtype=DTYPE)
    apply_diagonal_nonlocal_chunk(ch, 0, [3], _U("RZ", theta=0.5), 1, 2, 1)
    assert ch.dtype == np.complex64
