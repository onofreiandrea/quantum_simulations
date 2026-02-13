"""In-memory reference simulator (practical up to n ≈ 20, oracle for correctness).

Applies gates one-by-one directly to the state vector (no full unitary build).
Endianness: little-endian (qubit 0 = bit 0 = LSB).
"""
from __future__ import annotations

import numpy as np
from wenbo_engine.kernel import gates as gmod
from wenbo_engine.circuit.io import validate_circuit_dict


def _apply_1q(psi: np.ndarray, q: int, U: np.ndarray) -> None:
    N = len(psi)
    step = 1 << q
    block = step << 1
    base = np.arange(0, N, block)
    off = np.arange(step)
    idx0 = (base[:, None] + off[None, :]).ravel()
    idx1 = idx0 + step
    a, b = psi[idx0].copy(), psi[idx1].copy()
    psi[idx0] = U[0, 0] * a + U[0, 1] * b
    psi[idx1] = U[1, 0] * a + U[1, 1] * b


def _bases_2q(N: int, qa: int, qb: int) -> np.ndarray:
    """All indices with bits qa and qb both zero."""
    idx = np.arange(N)
    return idx[((idx >> qa) & 1 == 0) & ((idx >> qb) & 1 == 0)]


def _apply_2q(psi: np.ndarray, qa: int, qb: int, U: np.ndarray) -> None:
    """U in big-endian sub-space: qa=MSB, qb=LSB."""
    bases = _bases_2q(len(psi), qa, qb)
    i00 = bases
    i01 = bases | (1 << qb)
    i10 = bases | (1 << qa)
    i11 = bases | (1 << qa) | (1 << qb)
    v = np.stack([psi[i00], psi[i01], psi[i10], psi[i11]])  # (4, M)
    r = U @ v  # (4, M)
    psi[i00], psi[i01], psi[i10], psi[i11] = r[0], r[1], r[2], r[3]


def simulate(circuit_dict: dict) -> np.ndarray:
    """Run circuit, return final state vector (complex128)."""
    cd = validate_circuit_dict(circuit_dict)
    n = cd["number_of_qubits"]
    psi = np.zeros(1 << n, dtype=np.complex128)
    psi[0] = 1.0  # |0...0>
    for g in cd["gates"]:
        U = gmod.gate_matrix(g["gate"], g["params"])
        qubits = g["qubits"]
        if len(qubits) == 1:
            _apply_1q(psi, qubits[0], U)
        else:
            _apply_2q(psi, qubits[0], qubits[1], U)
    return psi
