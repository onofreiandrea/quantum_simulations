"""Per-circuit qubit activity counts and 2-qubit interaction frequency.

Used by the placement heuristic (:mod:`.placement_planner`):

  * ``activity[q]``   — number of gates that touch qubit ``q``.
  * ``interaction[(a, b)]`` — number of 2-qubit gates on the *unordered*
    pair ``{a, b}`` (a, b stored sorted so the key is canonical).

"Hot" qubits are those with high activity; pairs with high interaction
frequency want to share locality so their 2-qubit gates stay chunk-local.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field


@dataclass
class QubitActivity:
    n_qubits: int
    activity: Counter = field(default_factory=Counter)
    interaction: Counter = field(default_factory=Counter)

    def hottest(self, count: int | None = None) -> list[int]:
        """Qubits ordered hottest-first.

        Tie-break by ascending qubit index for determinism.  Untouched
        qubits (activity 0) appear last, also in index order.
        """
        order = sorted(
            range(self.n_qubits),
            key=lambda q: (-self.activity.get(q, 0), q),
        )
        if count is None:
            return order
        return order[:count]

    def interaction_pairs(self) -> list[tuple[tuple[int, int], int]]:
        """``[((a, b), freq), ...]`` ordered by descending frequency.

        Deterministic tie-break: ascending ``(a, b)``.
        """
        return sorted(
            self.interaction.items(),
            key=lambda kv: (-kv[1], kv[0]),
        )


def qubit_activity(circuit_dict: dict) -> QubitActivity:
    """Count per-qubit activity and 2-qubit interaction frequency."""
    n = circuit_dict["number_of_qubits"]
    qa = QubitActivity(n_qubits=n)
    for g in circuit_dict["gates"]:
        qs = g["qubits"]
        for q in qs:
            qa.activity[q] += 1
        if len(qs) == 2:
            a, b = sorted(qs)
            qa.interaction[(a, b)] += 1
    return qa
