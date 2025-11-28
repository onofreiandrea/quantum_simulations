"""
Circuit generators reused from InfiniQuantumSim.
"""
from __future__ import annotations

import numpy as np

from .gates import GGate


def generate_ghz_circuit(num_qubits: int, reverse: bool = False):
    gates = [{"qubits": [0], "gate": "H"}]
    for q in range(1, num_qubits):
        gates.append({"qubits": [q - 1, q], "gate": "CNOT"})
    if reverse:
        gates.reverse()
    return {"number_of_qubits": num_qubits, "gates": gates}


def generate_qft_circuit(num_qubits: int, reverse: bool = False):
    gates = []
    for j in range(num_qubits):
        gates.append({"qubits": [j], "gate": "H"})
        for k in range(j + 1, num_qubits):
            exponent = k - j + 1
            gates.append({"qubits": [k, j], "gate": "CR", "params": {"k": exponent}})
    if reverse:
        gates.reverse()
    return {"number_of_qubits": num_qubits, "gates": gates}


def generate_qpe_circuit(num_qubits: int):
    gates = []
    U = np.array([[1, 0], [0, -1]], dtype=complex)
    for j in range(num_qubits):
        gates.append({"qubits": [j], "gate": "H"})
    for j in range(num_qubits):
        exponent = 2**j
        gates.append({"qubits": [j, num_qubits], "gate": "CU", "params": {"U": U, "exponent": exponent}})
    for j in range(num_qubits):
        for k in range(j):
            exponent = j - k + 1
            gates.append({"qubits": [k, j], "gate": "CR", "params": {"k": exponent}})
        gates.append({"qubits": [j], "gate": "H"})
    return {"number_of_qubits": num_qubits + 1, "gates": gates}


def generate_w_circuit(n_qubits: int, reverse: bool = False):
    gates = [
        {"qubits": [0], "gate": "X"},
        {"qubits": [1], "gate": "G", "params": {"p": n_qubits}},
        {"qubits": [1, 0], "gate": "CNOT"},
    ]
    for i in range(n_qubits - 2):
        U = GGate(0, n_qubits - 1 - i).tensor
        gates.append(
            {"qubits": [i + 1, i + 2], "gate": "CU", "params": {"U": U, "exponent": 1, "name": f"CG{n_qubits - 1 - i}"}}
        )
        gates.append({"qubits": [i + 2, i + 1], "gate": "CNOT"})
    if reverse:
        gates.reverse()
    return {"number_of_qubits": n_qubits, "gates": gates}


def generate_hadamard_wall(n_qubits: int):
    return {"number_of_qubits": n_qubits, "gates": [{"qubits": [i], "gate": "H"} for i in range(n_qubits)]}


def generate_w_qft(n_qubits: int):
    circuit = generate_w_circuit(n_qubits)
    circuit["gates"].extend(generate_qft_circuit(n_qubits)["gates"])
    return circuit


def generate_ghz_qft(n_qubits: int):
    circuit = generate_ghz_circuit(n_qubits)
    circuit["gates"].extend(generate_qft_circuit(n_qubits)["gates"])
    return circuit


def generate_ghz_proned(n_qubits: int, depth: int):
    gates = []
    reverse = False
    while len(gates) < depth:
        gates.extend(generate_ghz_circuit(n_qubits, reverse)["gates"])
        reverse = not reverse
    return {"number_of_qubits": n_qubits, "gates": gates[:depth]}

