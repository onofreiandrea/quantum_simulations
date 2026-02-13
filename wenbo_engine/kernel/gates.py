"""Canonical gate matrices.

Convention:
  1-qubit gates: 2×2 complex128 ndarray.
  2-qubit gates: 4×4 complex128 ndarray in *big-endian* sub-space order:
      row/col 0 → (q_a=0, q_b=0)
      row/col 1 → (q_a=0, q_b=1)
      row/col 2 → (q_a=1, q_b=0)
      row/col 3 → (q_a=1, q_b=1)
  where q_a = qubits[0], q_b = qubits[1] from the gate entry.
"""
from __future__ import annotations

import numpy as np

_S2 = 1.0 / np.sqrt(2.0)


def _mat(*rows):
    return np.array(rows, dtype=np.complex128)


# ── 1-qubit fixed ───────────────────────────────────────────────────
def H():
    return _mat([_S2, _S2], [_S2, -_S2])

def X():
    return _mat([0, 1], [1, 0])

def Y():
    return _mat([0, -1j], [1j, 0])

def Z():
    return _mat([1, 0], [0, -1])

def S():
    return _mat([1, 0], [0, 1j])

def T():
    return _mat([1, 0], [0, np.exp(1j * np.pi / 4)])


# ── 1-qubit parameterised ──────────────────────────────────────────
def RY(theta: float):
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return _mat([c, -s], [s, c])

def R(k: int):
    return _mat([1, 0], [0, np.exp(2j * np.pi / 2**k)])

def G(p: int):
    a = np.sqrt(1.0 / p)
    b = np.sqrt(1.0 - 1.0 / p)
    return _mat([a, -b], [b, a])


# ── 2-qubit fixed (big-endian sub-space) ────────────────────────────
def CNOT():
    return _mat([1,0,0,0],[0,1,0,0],[0,0,0,1],[0,0,1,0])

def SWAP():
    return _mat([1,0,0,0],[0,0,1,0],[0,1,0,0],[0,0,0,1])

def CZ():
    return _mat([1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,-1])

def CY():
    return _mat([1,0,0,0],[0,1,0,0],[0,0,0,-1j],[0,0,1j,0])


# ── 2-qubit parameterised ──────────────────────────────────────────
def CR(k: int):
    return _mat([1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0, np.exp(2j*np.pi/2**k)])

def CU(U: np.ndarray, exponent: int):
    Up = np.linalg.matrix_power(np.asarray(U, dtype=np.complex128), exponent)
    return _mat(
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, Up[0,0], Up[0,1]],
        [0, 0, Up[1,0], Up[1,1]],
    )


# ── dispatcher ──────────────────────────────────────────────────────
_FIXED_1Q = {"H": H, "X": X, "Y": Y, "Z": Z, "S": S, "T": T}
_PARAM_1Q = {"RY": RY, "R": R, "G": G}
_FIXED_2Q = {"CNOT": CNOT, "SWAP": SWAP, "CZ": CZ, "CY": CY}
_PARAM_2Q = {"CR": CR, "CU": CU}


def gate_matrix(name: str, params: dict) -> np.ndarray:
    """Return the unitary matrix for a gate entry."""
    if name in _FIXED_1Q:
        return _FIXED_1Q[name]()
    if name in _FIXED_2Q:
        return _FIXED_2Q[name]()
    if name == "RY":
        return RY(params["theta"])
    if name == "R":
        return R(params["k"])
    if name == "G":
        return G(params["p"])
    if name == "CR":
        return CR(params["k"])
    if name == "CU":
        return CU(params["U"], params["exponent"])
    raise ValueError(f"unknown gate {name}")


def is_2q(name: str) -> bool:
    return name in _FIXED_2Q or name in _PARAM_2Q
