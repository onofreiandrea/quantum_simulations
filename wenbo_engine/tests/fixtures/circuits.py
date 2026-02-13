"""Canonical test circuits (dict format)."""
from __future__ import annotations

import numpy as np


def bell_2q() -> dict:
    """H(0) → CNOT(0,1).  Expected: (|00>+|11>)/√2."""
    return {
        "number_of_qubits": 2,
        "gates": [
            {"qubits": [0], "gate": "H"},
            {"qubits": [0, 1], "gate": "CNOT"},
        ],
    }


def x_on_q0_3q() -> dict:
    """X on qubit 0, 3 qubits.  |000> → |001>.  Amplitude at index 1."""
    return {
        "number_of_qubits": 3,
        "gates": [
            {"qubits": [0], "gate": "X"},
        ],
    }


def ry_theta() -> dict:
    """RY(pi/3) on qubit 0, 2 qubits."""
    return {
        "number_of_qubits": 2,
        "gates": [
            {"qubits": [0], "gate": "RY", "params": {"theta": np.pi / 3}},
        ],
    }


def cr3_encoded() -> dict:
    """CR3 (name-encoded CR with k=3) on qubits 0,1."""
    return {
        "number_of_qubits": 2,
        "gates": [
            {"qubits": [0], "gate": "H"},
            {"qubits": [1], "gate": "H"},
            {"qubits": [0, 1], "gate": "CR3"},
        ],
    }


def ghz(n: int) -> dict:
    gates = [{"qubits": [0], "gate": "H"}]
    for q in range(1, n):
        gates.append({"qubits": [q - 1, q], "gate": "CNOT"})
    return {"number_of_qubits": n, "gates": gates}


def qft(n: int) -> dict:
    gates = []
    for j in range(n):
        gates.append({"qubits": [j], "gate": "H"})
        for k in range(j + 1, n):
            gates.append({"qubits": [k, j], "gate": "CR", "params": {"k": k - j + 1}})
    return {"number_of_qubits": n, "gates": gates}
