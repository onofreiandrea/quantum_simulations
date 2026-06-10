"""Activity-based qubit placement heuristic.

Given the circuit's qubit activity and the hardware layout
(``n`` qubits, ``k`` chunk-local bits, ``p`` rank bits), choose a
permutation ``perm`` mapping ``old_qubit -> new_qubit`` (new physical bit
position) that minimizes predicted data movement and MPI traffic.

Heuristic (matches the task spec):

  1. Count qubit activity per circuit.
  2. Count 2-qubit interaction frequency.
  3. Map the hottest qubits to low physical bits (the chunk-local
     positions ``0..k-1``), so most gates become chunk-local.
  4. Avoid placing active qubits on rank bits (the top ``p`` positions
     ``n-p..n-1``), so they do not generate MPI Sendrecv traffic.
  5. If unavoidable (more active qubits than non-rank positions), place
     the *least* active remaining qubits on rank bits — minimizing
     predicted MPI bytes.

Physical bit layout (matches :mod:`wenbo_engine.mpi.mpi_runner`):

    local         bits 0 .. k-1
    rank-nonlocal bits k .. n-p-1
    rank (MPI)    bits n-p .. n-1

The returned permutation has the SAME contract as
:func:`wenbo_engine.circuit.reorder.reorder_qubits`'s ``perm``
(``old_qubit -> new_qubit``), so it composes with the existing
state-permutation utilities.
"""
from __future__ import annotations

from wenbo_engine.planner.qubit_activity import QubitActivity, qubit_activity


def plan_placement(circuit_dict: dict, k: int, p: int,
                   activity: QubitActivity | None = None) -> dict[int, int]:
    """Return ``perm`` mapping ``old_qubit -> new physical bit`` position.

    Parameters
    ----------
    circuit_dict : dict
        Validated circuit dict.
    k : int
        Number of chunk-local bit positions (log2 chunk size).
    p : int
        Number of rank bits (log2 num_ranks); ``0`` for single rank.
    activity : QubitActivity, optional
        Pre-computed activity; computed from the circuit if omitted.
    """
    n = circuit_dict["number_of_qubits"]
    if activity is None:
        activity = qubit_activity(circuit_dict)

    n_rank_positions = p
    n_nonrank_positions = n - p  # local + rank-nonlocal bits

    # Hottest qubits first (deterministic tie-break inside QubitActivity).
    ordered = activity.hottest()

    # Step 3 + 4: fill the non-rank positions (0..n-p-1) with the hottest
    # qubits first.  The very lowest positions (0..k-1) get the hottest of
    # all — making those gates chunk-local.  The remaining (coldest) qubits
    # spill onto the rank bits (step 5: least active on rank bits, which
    # minimizes predicted MPI bytes since fewer gates touch them).
    nonrank_qubits = ordered[:n_nonrank_positions]
    rank_qubits = ordered[n_nonrank_positions:]

    perm: dict[int, int] = {}
    # New physical positions 0..n-p-1 go to the hottest (non-rank) qubits.
    for new_pos, old_q in enumerate(nonrank_qubits):
        perm[old_q] = new_pos
    # New physical positions n-p..n-1 (rank bits) go to coldest qubits.
    for offset, old_q in enumerate(rank_qubits):
        perm[old_q] = n_nonrank_positions + offset

    assert len(perm) == n, "placement must map every qubit"
    assert sorted(perm.values()) == list(range(n)), "perm must be a bijection"
    return perm


def apply_placement(circuit_dict: dict, perm: dict[int, int]) -> dict:
    """Apply a placement permutation, returning a relabelled circuit dict.

    ``perm`` maps ``old_qubit -> new_qubit``; gate qubit indices are
    rewritten accordingly.  Mathematically identical circuit, different
    physical data layout.  Mirrors
    :func:`wenbo_engine.circuit.reorder.reorder_qubits`'s relabelling.
    """
    n = circuit_dict["number_of_qubits"]
    new_gates = []
    for g in circuit_dict["gates"]:
        new_g = {
            "qubits": [perm[q] for q in g["qubits"]],
            "gate": g["gate"],
        }
        if "params" in g:
            new_g["params"] = g["params"]
        new_gates.append(new_g)
    return {"number_of_qubits": n, "gates": new_gates}
