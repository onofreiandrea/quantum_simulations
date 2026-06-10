"""Gate dependency DAG built from qubit dependencies.

A gate ``g`` depends on the most recent earlier gate that touched any of
its qubits (a read/write hazard on a shared qubit).  This is the same
ordering that :func:`wenbo_engine.circuit.io.levelize` respects, exposed
here as an explicit predecessor / successor structure so a planner can
check that no stage reorders a gate past a conflicting one.

Two gates with *disjoint* qubit sets commute and may be reordered freely;
two gates sharing a qubit must keep their original relative order.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CircuitDAG:
    """Dependency DAG over the gates of a (validated) circuit dict.

    ``predecessors[g]`` / ``successors[g]`` are the *immediate* dependency
    edges (transitive edges omitted).  ``order_index`` is the gate's
    original position, which defines a valid topological order.
    """

    n_qubits: int
    n_gates: int
    predecessors: list[list[int]] = field(default_factory=list)
    successors: list[list[int]] = field(default_factory=list)

    def is_topological(self, order: list[int]) -> bool:
        """True if ``order`` is a valid linear extension of the DAG.

        Every gate must appear exactly once and after all of its
        predecessors.
        """
        if sorted(order) != list(range(self.n_gates)):
            return False
        position = [0] * self.n_gates
        for pos, g in enumerate(order):
            position[g] = pos
        for g in range(self.n_gates):
            for p in self.predecessors[g]:
                if position[p] >= position[g]:
                    return False
        return True

    def conflicts(self, ga: int, gb: int, qubits: list[list[int]]) -> bool:
        """True if gates ``ga`` and ``gb`` share a qubit (do not commute)."""
        return bool(set(qubits[ga]) & set(qubits[gb]))


def build_dag(circuit_dict: dict) -> CircuitDAG:
    """Build a :class:`CircuitDAG` from a validated circuit dict.

    Edge rule: gate ``g`` gets an immediate predecessor edge from the last
    gate seen so far on each of its qubits (deduplicated).  This captures
    exactly the hazards levelization enforces.
    """
    gates = circuit_dict["gates"]
    n = circuit_dict["number_of_qubits"]
    n_gates = len(gates)

    predecessors: list[list[int]] = [[] for _ in range(n_gates)]
    successors: list[list[int]] = [[] for _ in range(n_gates)]

    last_on_qubit: dict[int, int] = {}
    for gi, g in enumerate(gates):
        preds: list[int] = []
        seen: set[int] = set()
        for q in g["qubits"]:
            if q in last_on_qubit:
                p = last_on_qubit[q]
                if p not in seen:
                    seen.add(p)
                    preds.append(p)
        predecessors[gi] = preds
        for p in preds:
            successors[p].append(gi)
        for q in g["qubits"]:
            last_on_qubit[q] = gi

    return CircuitDAG(
        n_qubits=n,
        n_gates=n_gates,
        predecessors=predecessors,
        successors=successors,
    )
