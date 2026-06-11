"""Candidate execution strategies and their predicted cost.

The recovery-aware planner v1 evaluates a fixed set of *candidate strategies*
(layout × execution-mode × extent-I/O × MPI-exchange) and estimates, for each,
the data-movement quantities and the recovery-aware cost the run would incur.
Estimates are grounded in the real runner's compiled steps + compute-unit plan
(passed in via :class:`PlanContext`), so the prediction reflects the same
levelization and fusion the runner actually executes — no hand-waving.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from wenbo_engine.planner.stage_cost_model import recovery_aware_cost


def estimate_mpi_exchange(steps: list[dict], *, num_ranks: int, k: int,
                          n_local_bits: int, n_chunks_per_rank: int,
                          chunk_bytes: int) -> dict:
    """Deterministic, planner-derived MPI exchange estimate per exchange mode.

    Uses the REAL gate-aware exchange planner
    (:mod:`wenbo_engine.mpi.exchange_planner`) to resolve every MPI-nonlocal
    gate to its partner rank + chunk pairs, then counts exchanges exactly the
    way each runner path does:

      * naive: one ``Sendrecv`` per chunk per gate;
      * gate_aware: one batched ``Sendrecv`` per partner per step (moving the
        rank's ``n_chunks_per_rank`` chunks once, reused across that partner's
        gates), plus the naive path for any non-batchable fallback gate.

    Returns ``{"naive": {...}, "gate_aware": {...}}`` where each is
    ``{sendrecv_count, mpi_bytes_sent, partner_rank_pairs}`` aggregated across
    all ranks.  This replaces the previous fixed reuse heuristic.
    """
    from wenbo_engine.mpi.exchange_planner import (
        plan_stage, group_by_partner, fallback_gates,
    )
    naive_sr = naive_chunks = 0
    ga_sr = ga_chunks = 0
    pairs: set = set()
    for r in range(num_ranks):
        for s in steps:
            ops = s.get("mpi_nonlocal_ops") or []
            if not ops:
                continue
            plan = plan_stage(ops, r, k, n_local_bits, n_chunks_per_rank)
            # naive: per chunk per gate
            for ge in plan:
                cp = len(ge.chunk_pairs) or n_chunks_per_rank
                naive_sr += cp
                naive_chunks += cp
            # gate_aware: one batched exchange per partner (C chunks), reused
            groups = group_by_partner(plan)
            for partner in groups:
                ga_sr += 1
                ga_chunks += n_chunks_per_rank
                pairs.add(frozenset((r, partner)))
            for ge in fallback_gates(plan):
                cp = len(ge.chunk_pairs) or n_chunks_per_rank
                ga_sr += cp
                ga_chunks += cp
    return {
        "naive": {"sendrecv_count": naive_sr,
                  "mpi_bytes_sent": naive_chunks * chunk_bytes,
                  "partner_rank_pairs": len(pairs)},
        "gate_aware": {"sendrecv_count": ga_sr,
                       "mpi_bytes_sent": ga_chunks * chunk_bytes,
                       "partner_rank_pairs": len(pairs)},
        "method": "exchange_planner",
    }


@dataclass(frozen=True)
class StrategyCandidate:
    name: str
    storage_layout: str          # "chunks" | "extents"
    execution_mode: str          # "step" | "compute_unit"
    extent_io_mode: str          # "materialize" | "direct"
    mpi_exchange_mode: str       # "naive" | "gate_aware"
    compute_unit_min_gates: int = 4

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "storage_layout": self.storage_layout,
            "execution_mode": self.execution_mode,
            "extent_io_mode": self.extent_io_mode,
            "mpi_exchange_mode": self.mpi_exchange_mode,
            "compute_unit_min_gates": self.compute_unit_min_gates,
        }


def enumerate_candidates(compute_unit_min_gates: int = 4
                         ) -> list[StrategyCandidate]:
    """The planner-v1 candidate strategies (always in this order)."""
    return [
        StrategyCandidate(
            "chunks+step+naive", "chunks", "step", "materialize", "naive",
            compute_unit_min_gates),
        StrategyCandidate(
            "chunks+step+gate_aware", "chunks", "step", "materialize",
            "gate_aware", compute_unit_min_gates),
        StrategyCandidate(
            "extents+step+gate_aware", "extents", "step", "materialize",
            "gate_aware", compute_unit_min_gates),
        StrategyCandidate(
            "extents+compute_unit+materialize+gate_aware", "extents",
            "compute_unit", "materialize", "gate_aware", compute_unit_min_gates),
        StrategyCandidate(
            "extents+compute_unit+direct+gate_aware", "extents",
            "compute_unit", "direct", "gate_aware", compute_unit_min_gates),
    ]


@dataclass
class PlanContext:
    """Everything candidate estimation needs, computed once per circuit/hw."""
    steps: list[dict]            # per circuit step: gates/local/rank_nl/mpi_nl/active_qubits
    units: list[dict]            # compute-unit plan (min_gates applied): + kind/n_steps
    n_chunks_per_rank: int
    chunk_bytes: int
    num_ranks: int
    total_gates: int
    total_mpi_ops: int
    recovery: str
    compute_unit_min_gates: int
    cost_model: dict
    max_local_run_gates: int = 0
    has_fused_local_unit: bool = False
    n_compute_unit_fallbacks: int = 0
    # deterministic per-mode MPI estimate from the exchange planner:
    # {"naive": {...}, "gate_aware": {...}, "method": "exchange_planner"}
    mpi_estimates: dict = field(default_factory=dict)


def _passes_for(candidate: StrategyCandidate, ctx: PlanContext) -> list[dict]:
    """The list of execution passes (one read+write+commit each) for a strategy."""
    if candidate.execution_mode == "compute_unit":
        return ctx.units
    return ctx.steps


def estimate_candidate(candidate: StrategyCandidate, ctx: PlanContext) -> dict:
    """Predict data-movement quantities + recovery-aware cost for a candidate.

    Returns a flat dict containing every required estimate term:
    bytes_read/written, read_ops, write_ops, temporary_chunk_files_created,
    mpi_bytes_sent, sendrecv_count, kernel_time, commit_count, commit_cost,
    durable_checkpoint_cost, expected_recomputation_cost,
    layout_materialization_cost, and estimated_total_cost (plus the full cost
    breakdown under ``cost_terms``).
    """
    C = ctx.n_chunks_per_rank
    B = ctx.chunk_bytes
    R = ctx.num_ranks
    layout, exec_mode, io_mode = (candidate.storage_layout,
                                  candidate.execution_mode,
                                  candidate.extent_io_mode)

    bytes_read = bytes_written = 0
    read_ops = write_ops = 0
    extent_materialize_bytes = 0
    direct_extent_chunks = 0

    for pas in _passes_for(candidate, ctx):
        is_local = (pas.get("rank_nl", 0) == 0 and pas.get("mpi_nl", 0) == 0)
        # logical state read+write for this pass (aggregate across ranks)
        bytes_read += C * B * R
        bytes_written += C * B * R

        # physical chunk-file op counts + temp round-trip overhead
        direct_local = (layout == "extents" and io_mode == "direct" and is_local)
        if direct_local:
            # raw extent-slice I/O: no chunk-file ops, no temp round trip
            direct_extent_chunks += C * R
        else:
            read_ops += C * R
            write_ops += C * R
            if layout == "extents":
                # materialize round trip: unpack extents→chunks + pack chunks→extents
                extent_materialize_bytes += 2 * C * B * R

    # MPI exchange: deterministic, planner-derived (NOT a heuristic) — depends
    # only on the circuit's MPI gates + rank count + exchange mode, not on the
    # storage/execution mode.
    mpi_est = (ctx.mpi_estimates or {}).get(candidate.mpi_exchange_mode,
                                            {"sendrecv_count": 0,
                                             "mpi_bytes_sent": 0,
                                             "partner_rank_pairs": 0})
    mpi_bytes = mpi_est["mpi_bytes_sent"]
    sendrecv_count = mpi_est["sendrecv_count"]

    passes = _passes_for(candidate, ctx)
    commit_count = len(passes) + 1            # +1 for the gen-0 init commit
    kernel_bytes = ctx.total_gates * C * B * R
    temporary_chunk_files_created = write_ops
    # one generation's worth of work is at risk between commits
    recompute_bytes = C * B * R
    # fresh run: gen-0 is written directly in-layout, no pre-run conversion
    layout_materialize_bytes = 0
    durable_bytes = 0                         # no durable policy in v1 default

    cost = recovery_aware_cost(
        bytes_read=bytes_read, bytes_written=bytes_written,
        mpi_bytes=mpi_bytes, sendrecv_count=sendrecv_count,
        kernel_bytes=kernel_bytes, commits=commit_count,
        extent_materialize_bytes=extent_materialize_bytes,
        direct_extent_chunks=direct_extent_chunks,
        layout_materialize_bytes=layout_materialize_bytes,
        durable_bytes=durable_bytes, recompute_bytes=recompute_bytes,
        model=ctx.cost_model)

    return {
        "candidate": candidate.name,
        "strategy": candidate.to_dict(),
        # raw predicted data-movement quantities (cluster-aggregate)
        "bytes_read": bytes_read,
        "bytes_written": bytes_written,
        # same quantities at both aggregation levels, explicitly labelled so a
        # cluster prediction is never compared to a rank-0 actual unlabelled.
        "predicted_cluster_bytes_read": bytes_read,
        "predicted_cluster_bytes_written": bytes_written,
        "predicted_per_rank_bytes_read": bytes_read // R,
        "predicted_per_rank_bytes_written": bytes_written // R,
        "read_ops": read_ops,
        "write_ops": write_ops,
        "temporary_chunk_files_created": temporary_chunk_files_created,
        "mpi_bytes_sent": mpi_bytes,
        "sendrecv_count": sendrecv_count,
        "estimated_partner_rank_pairs": mpi_est.get("partner_rank_pairs", 0),
        "mpi_estimate_method": (ctx.mpi_estimates or {}).get(
            "method", "exchange_planner"),
        "kernel_time": cost["kernel_cost"],
        "commit_count": commit_count,
        # cost terms (seconds)
        "commit_cost": cost["commit_cost"],
        "durable_checkpoint_cost": cost["durable_checkpoint_cost"],
        "expected_recomputation_cost": cost["expected_recomputation_cost"],
        "layout_materialization_cost": cost["layout_materialization_cost"],
        "extent_materialization_cost": cost["extent_materialization_cost"],
        "direct_extent_io_cost": cost["direct_extent_io_cost"],
        "estimated_total_cost": cost["estimated_total_cost"],
        "cost_terms": cost,
    }
