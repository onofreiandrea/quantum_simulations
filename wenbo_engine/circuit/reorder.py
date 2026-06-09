"""Qubit reordering to minimize non-local gate operations.

Places the most-frequently targeted qubits in the lowest bit positions
(which map to chunk-internal / local qubits), and the least-targeted
qubits in the highest positions (which become MPI-nonlocal).

This is a qubit index permutation — mathematically identical circuit,
but changes the physical data layout to minimize expensive I/O and
network transfers.
"""
from __future__ import annotations

from collections import Counter

import numpy as np


def reorder_qubits(circuit_dict: dict) -> tuple[dict, dict[int, int]]:
    """Reorder qubits so the most-targeted qubits get the lowest positions.

    Returns:
        (reordered_circuit_dict, perm)
        where perm maps old_qubit -> new_qubit.
    """
    n = circuit_dict["number_of_qubits"]

    freq: Counter = Counter()
    for g in circuit_dict["gates"]:
        for q in g["qubits"]:
            freq[q] += 1

    # Sort: most targeted → position 0, least targeted → position n-1
    # Untargeted qubits (freq=0) naturally go to the highest positions
    all_qubits = list(range(n))
    sorted_by_freq = sorted(all_qubits, key=lambda q: freq.get(q, 0), reverse=True)

    # perm: old_qubit → new_qubit
    perm = {old_q: new_q for new_q, old_q in enumerate(sorted_by_freq)}

    new_gates = []
    for g in circuit_dict["gates"]:
        new_g = {
            "qubits": [perm[q] for q in g["qubits"]],
            "gate": g["gate"],
        }
        if "params" in g:
            new_g["params"] = g["params"]
        new_gates.append(new_g)

    return {
        "number_of_qubits": n,
        "gates": new_gates,
    }, perm


def permute_state_vector(state: np.ndarray, perm: dict[int, int],
                         n: int) -> np.ndarray:
    """Apply qubit permutation to a state vector.

    If perm maps old_qubit → new_qubit, then for each basis state,
    bit old_q in the original index maps to bit new_q in the new index.

    Only practical for small n (≤ ~20).
    """
    new_state = np.zeros_like(state)
    for i in range(len(state)):
        j = 0
        for old_q, new_q in perm.items():
            if i & (1 << old_q):
                j |= (1 << new_q)
        new_state[j] = state[i]
    return new_state
