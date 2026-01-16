"""
Gate definitions reused from the InfiniQuantumSim project.
"""
from __future__ import annotations

import numpy as np


_PERM_2Q = np.array([0, 2, 1, 3])


def _to_little_endian(matrix: np.ndarray) -> np.ndarray:
    """Permute a 4x4 matrix from big-endian to little-endian ordering."""
    return matrix[np.ix_(_PERM_2Q, _PERM_2Q)]


class Gate:
    """Base gate class storing qubit indices and tensor representation."""

    def __init__(self, qubits, tensor, name, two_qubit_gate=False):
        self.qubits = qubits
        self.tensor = tensor
        self.gate_name = name
        self.two_qubit_gate = two_qubit_gate


class HadamardGate(Gate):
    def __init__(self, qubit: int):
        matrix = (1 / np.sqrt(2)) * np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex)
        super().__init__([qubit], matrix, "H")


class XGate(Gate):
    def __init__(self, qubit: int):
        matrix = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
        super().__init__([qubit], matrix, "X")


class YGate(Gate):
    def __init__(self, qubit: int):
        matrix = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
        super().__init__([qubit], matrix, "Y")


class ZGate(Gate):
    def __init__(self, qubit: int):
        matrix = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
        super().__init__([qubit], matrix, "Z")


class SGate(Gate):
    def __init__(self, qubit: int):
        matrix = np.array([[1.0, 0.0], [0.0, 1.0j]], dtype=complex)
        super().__init__([qubit], matrix, "S")


class TGate(Gate):
    def __init__(self, qubit: int):
        matrix = np.array([[1.0, 0.0], [0.0, np.exp(1j * np.pi / 4)]], dtype=complex)
        super().__init__([qubit], matrix, "T")


class RYGate(Gate):
    def __init__(self, qubit: int, theta: float):
        matrix = np.array(
            [[np.cos(theta / 2), -np.sin(theta / 2)], [np.sin(theta / 2), np.cos(theta / 2)]],
            dtype=complex,
        )
        super().__init__([qubit], matrix, "RY")


class RGate(Gate):
    def __init__(self, qubit: int, k: int):
        matrix = np.array([[1.0, 0.0], [0.0, np.exp(2j * np.pi / 2**k)]], dtype=complex)
        super().__init__([qubit], matrix, f"R{k}")


class GGate(Gate):
    def __init__(self, qubit: int, p: int):
        matrix = np.array(
            [
                [np.sqrt(1 / p), -np.sqrt(1 - (1 / p))],
                [np.sqrt(1 - (1 / p)), np.sqrt(1 / p)],
            ],
            dtype=complex,
        )
        super().__init__([qubit], matrix, f"G{p}")


class CNOTGate(Gate):
    def __init__(self, control_qubit: int, target_qubit: int):
        matrix = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
                [0, 0, 1, 0],
            ],
            dtype=complex,
        )
        super().__init__(
            [control_qubit, target_qubit], _to_little_endian(matrix).reshape(2, 2, 2, 2), "CNOT", True
        )


class SWAPGate(Gate):
    def __init__(self, qubit_a: int, qubit_b: int):
        matrix = np.array(
            [
                [1, 0, 0, 0],
                [0, 0, 1, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 1],
            ],
            dtype=complex,
        )
        super().__init__(
            [qubit_a, qubit_b], _to_little_endian(matrix).reshape(2, 2, 2, 2), "SWAP", True
        )


class CZGate(Gate):
    def __init__(self, control_qubit: int, target_qubit: int):
        matrix = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, -1],
            ],
            dtype=complex,
        )
        super().__init__(
            [control_qubit, target_qubit], _to_little_endian(matrix).reshape(2, 2, 2, 2), "CZ", True
        )


class CYGate(Gate):
    def __init__(self, control_qubit: int, target_qubit: int):
        matrix = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, -1j],
                [0, 0, 1j, 0],
            ],
            dtype=complex,
        )
        super().__init__(
            [control_qubit, target_qubit], _to_little_endian(matrix).reshape(2, 2, 2, 2), "CY", True
        )


class CRGate(Gate):
    def __init__(self, control_qubit: int, target_qubit: int, k: int):
        matrix = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 0, np.exp(2j * np.pi / 2**k)],
            ],
            dtype=complex,
        )
        super().__init__(
            [control_qubit, target_qubit], _to_little_endian(matrix).reshape(2, 2, 2, 2), f"CR{k}", True
        )


class CUGate(Gate):
    def __init__(self, control_qubit: int, target_qubit: int, U: np.ndarray, exponent: int, name: str | None = None):
        matrix = np.linalg.matrix_power(U, exponent)
        gate_name = name or f"CU{exponent}"
        controlled = np.array(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, matrix[0, 0], matrix[0, 1]],
                [0, 0, matrix[1, 0], matrix[1, 1]],
            ],
            dtype=complex,
        )
        super().__init__(
            [control_qubit, target_qubit],
            _to_little_endian(controlled).reshape(2, 2, 2, 2),
            gate_name,
            True,
        )

