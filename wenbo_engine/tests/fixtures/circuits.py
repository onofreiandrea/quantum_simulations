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


# ── Non-stabilizer circuits ──────────────────────────────────────────
# These circuits contain gates outside the Clifford group (T, RY with
# irrational angles) and CANNOT be efficiently simulated classically.
# They are the actual use case for state vector simulation at scale.


def random_clifford_t(n: int, depth: int = 10, seed: int = 42) -> dict:
    """Random circuit with H, T, CNOT layers — similar to quantum supremacy circuits.

    Each layer: random 1Q gates (H or T) on all qubits, then random CNOT pairs.
    Non-stabilizer due to T gates (pi/8 phase, outside Clifford group).
    """
    rng = np.random.RandomState(seed)
    gates = []
    for _ in range(depth):
        # Single-qubit layer: random H or T on each qubit
        for q in range(n):
            gate = rng.choice(["H", "T", "S"])
            gates.append({"qubits": [q], "gate": gate})
        # CNOT layer: random non-overlapping pairs
        perm = rng.permutation(n).tolist()
        for i in range(0, n - 1, 2):
            gates.append({"qubits": [perm[i], perm[i + 1]], "gate": "CNOT"})
    return {"number_of_qubits": n, "gates": gates}


def hardware_efficient_ansatz(n: int, layers: int = 5, seed: int = 42) -> dict:
    """Hardware-efficient variational ansatz (RY + CNOT layers).

    Common in VQE/QAOA. Non-stabilizer due to arbitrary RY rotations.
    Each layer: RY(theta) on all qubits, then nearest-neighbor CNOTs.
    """
    rng = np.random.RandomState(seed)
    gates = []
    for _ in range(layers):
        # RY layer with random angles
        for q in range(n):
            theta = rng.uniform(0, 2 * np.pi)
            gates.append({"qubits": [q], "gate": "RY", "params": {"theta": theta}})
        # Entangling layer: nearest-neighbor CNOTs
        for q in range(0, n - 1, 2):
            gates.append({"qubits": [q, q + 1], "gate": "CNOT"})
        for q in range(1, n - 1, 2):
            gates.append({"qubits": [q, q + 1], "gate": "CNOT"})
    return {"number_of_qubits": n, "gates": gates}


def quest_random(n: int, n_gates: int = 1000, seed: int = 42) -> dict:
    """Random circuit matching the QuEST/AWS benchmark (Baruffa et al. 2022).

    Algorithm from the blog post "Simulating 44-Qubit quantum circuits using
    AWS ParallelCluster":
      - For each gate: flip a coin
      - Heads: CZ on two random qubits
      - Tails: pick random 1Q gate from {RX, RY, RZ, H} with random angle
    All non-stabilizer due to RX/RY/RZ with irrational angles.
    """
    rng = np.random.RandomState(seed)
    gates = []
    for _ in range(n_gates):
        if rng.randint(2) == 0:
            # Two-qubit CZ on random pair
            q1, q2 = rng.choice(n, size=2, replace=False).tolist()
            gates.append({"qubits": [q1, q2], "gate": "CZ"})
        else:
            # Single-qubit gate
            q = int(rng.randint(n))
            g = str(rng.choice(["RX", "RY", "RZ", "H"]))
            if g == "H":
                gates.append({"qubits": [q], "gate": "H"})
            else:
                theta = float(rng.uniform(0, np.pi))
                gates.append({"qubits": [q], "gate": g, "params": {"theta": theta}})
    return {"number_of_qubits": n, "gates": gates}


def supremacy_like(n: int, cycles: int = 8, seed: int = 42) -> dict:
    """Google-style supremacy circuit pattern.

    Alternates between random 1Q gates (from {sqrt(X), sqrt(Y), sqrt(W)},
    approximated here with H/T/RY) and a fixed 2Q gate pattern.
    Non-stabilizer due to T gates and irrational rotations.
    """
    rng = np.random.RandomState(seed)
    gates = []
    # Initial layer of H on all qubits
    for q in range(n):
        gates.append({"qubits": [q], "gate": "H"})

    for cycle in range(cycles):
        # Random 1Q gates (non-Clifford mix)
        for q in range(n):
            choice = rng.randint(3)
            if choice == 0:
                gates.append({"qubits": [q], "gate": "T"})
            elif choice == 1:
                theta = rng.uniform(0.1, np.pi)
                gates.append({"qubits": [q], "gate": "RY", "params": {"theta": theta}})
            else:
                gates.append({"qubits": [q], "gate": "H"})
                gates.append({"qubits": [q], "gate": "T"})

        # 2Q layer: alternating pairs (even/odd offset per cycle)
        offset = cycle % 2
        for q in range(offset, n - 1, 2):
            gates.append({"qubits": [q, q + 1], "gate": "CZ"})

    return {"number_of_qubits": n, "gates": gates}
