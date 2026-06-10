"""Gate-aware MPI exchange planning.

Pure, side-effect-free analysis of a stage's MPI-nonlocal gates: resolve each
gate's partner rank + the exact local/remote chunk pairs it needs, classify
whether it fits the batched fast path, and group gates by partner rank so the
runner can batch exchanges (and reuse received remote buffers) instead of doing
one blocking ``Sendrecv`` per chunk per gate.

No MPI, no I/O, no numpy state — just arithmetic over the bit layout, so it is
fully unit-testable.  The chunk index that a rank exchanges with its partner is
the SAME local index (an MPI-nonlocal qubit is a *rank* bit, so both ranks hold
the matching chunk at the same local offset).
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field


def resolve_partner(rank: int, mpi_q: int, k: int, n_local_bits: int):
    """Return (rank_bit, partner_rank, i_am_low) for an MPI-nonlocal qubit."""
    rank_bit = (mpi_q - k) - n_local_bits
    partner_rank = rank ^ (1 << rank_bit)
    i_am_low = (rank & (1 << rank_bit)) == 0
    return rank_bit, partner_rank, i_am_low


@dataclass
class GateExchange:
    """One MPI-nonlocal gate resolved to a concrete exchange.

    ``kind`` is one of:
      ``"1q"``            single-qubit MPI gate (batchable).
      ``"2q_one_local"``  2-qubit gate, one MPI qubit + one *chunk-local*
                          qubit (batchable).
      ``"fallback"``      anything else (2q with both qubits MPI, or the other
                          qubit rank-nonlocal) — the runner applies these with
                          the existing naive per-gate path; correctness first.
    """
    qs: list
    U: object
    kind: str
    rank_bit: int = -1
    partner_rank: int = -1
    i_am_low: bool = False
    other_q: int | None = None
    mpi_is_first: bool | None = None
    # (local_chunk_index, remote_chunk_index) pairs this gate exchanges.
    chunk_pairs: list = field(default_factory=list)

    @property
    def batchable(self) -> bool:
        return self.kind in ("1q", "2q_one_local")


def classify_gate(qs, U, rank: int, k: int, n_local_bits: int,
                  n_chunks_per_rank: int) -> GateExchange:
    qs_mpi = [q for q in qs if (q - k) >= n_local_bits]
    if len(qs) == 1:
        rb, p, low = resolve_partner(rank, qs[0], k, n_local_bits)
        pairs = [(ci, ci) for ci in range(n_chunks_per_rank)]
        return GateExchange(qs, U, "1q", rb, p, low, chunk_pairs=pairs)
    if len(qs) == 2 and len(qs_mpi) == 1:
        mpi_q = qs_mpi[0]
        other = qs[0] if qs[1] == mpi_q else qs[1]
        if other < k:                          # other qubit is chunk-local
            rb, p, low = resolve_partner(rank, mpi_q, k, n_local_bits)
            pairs = [(ci, ci) for ci in range(n_chunks_per_rank)]
            return GateExchange(qs, U, "2q_one_local", rb, p, low,
                                other_q=other, mpi_is_first=(qs.index(mpi_q) == 0),
                                chunk_pairs=pairs)
    return GateExchange(qs, U, "fallback")


def plan_stage(mpi_nonlocal_ops, rank: int, k: int, n_local_bits: int,
               n_chunks_per_rank: int) -> list[GateExchange]:
    """Resolve every MPI gate in a stage to a :class:`GateExchange`."""
    return [classify_gate(qs, U, rank, k, n_local_bits, n_chunks_per_rank)
            for qs, U in mpi_nonlocal_ops]


def group_by_partner(plan: list[GateExchange]) -> dict[int, list[GateExchange]]:
    """Group batchable gate exchanges by partner rank (for batched Sendrecv)."""
    groups: dict[int, list[GateExchange]] = defaultdict(list)
    for ge in plan:
        if ge.batchable:
            groups[ge.partner_rank].append(ge)
    return dict(groups)


def fallback_gates(plan: list[GateExchange]) -> list[GateExchange]:
    return [ge for ge in plan if not ge.batchable]
