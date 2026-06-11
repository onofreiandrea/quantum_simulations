"""Recovery-aware hierarchical planner v1.

Before a run, this planner:

  1. compiles the circuit into the SAME levelized steps the MPI runner executes
     (so predictions are grounded, not hand-waved),
  2. predicts the compute-unit fusion plan (with the adaptive min-gates rule),
  3. enumerates the candidate execution strategies and estimates each one's
     data-movement quantities + recovery-aware cost,
  4. selects a safe strategy with deterministic, explained rules, and
  5. emits a per-stage plan, the candidate table, and the decision.

After the run, :mod:`wenbo_engine.planner.cost_report` compares the selected
candidate's predicted metrics against the measured metrics.

It changes no kernels, no recovery semantics, and no MPI exchange — it only
chooses among already-supported, already-safe execution modes and prices them.
"""
from __future__ import annotations

import math

import numpy as np

from wenbo_engine.storage.block_store import DTYPE
from wenbo_engine.planner.stage_cost_model import (
    load_cost_model, recovery_aware_cost,
)
from wenbo_engine.planner.stage_plan import StagePlan
from wenbo_engine.planner.strategy_candidate import (
    StrategyCandidate, PlanContext, enumerate_candidates, estimate_candidate,
    estimate_mpi_exchange,
)
from wenbo_engine.planner.strategy_selector import select_strategy

PLANNER_NAME = "recovery_aware_v1"


def _qubits_of_ops(ops) -> set[int]:
    out: set[int] = set()
    for qs, _U in ops:
        out.update(int(q) for q in qs)
    return out


def _build_step_descriptors(steps: list[dict]) -> list[dict]:
    descs = []
    for i, s in enumerate(steps):
        lo, rn, mn = (s["local_ops"], s["rank_nonlocal_ops"],
                      s["mpi_nonlocal_ops"])
        aq = _qubits_of_ops(lo) | _qubits_of_ops(rn) | _qubits_of_ops(mn)
        descs.append({
            "stage_id": i, "gates": len(lo) + len(rn) + len(mn),
            "local": len(lo), "rank_nl": len(rn), "mpi_nl": len(mn),
            "n_steps": 1, "kind": "step", "active_qubits": sorted(aq),
        })
    return descs


def _build_unit_descriptors(steps: list[dict], *, n_chunks_per_rank: int,
                            min_gates: int) -> tuple[list[dict], int]:
    """Predict the compute-unit plan; return (descriptors, n_fallbacks)."""
    from wenbo_engine.runtime.overlay_scheduler import build_compute_units
    units = build_compute_units(steps, rank=0,
                                n_chunks_per_rank=n_chunks_per_rank,
                                start_gen=0, min_gates=min_gates)
    descs = []
    fallbacks = 0
    for u in units:
        if u.kind == "local":
            local, rn, mn = u.gate_count, 0, 0
            aq = set(int(q) for qs, _ in u.local_ops for q in qs)
        else:
            s = u.step or {}
            local = len(s.get("local_ops", []))
            rn = len(s.get("rank_nonlocal_ops", []))
            mn = len(s.get("mpi_nonlocal_ops", []))
            aq = (_qubits_of_ops(s.get("local_ops", []))
                  | _qubits_of_ops(s.get("rank_nonlocal_ops", []))
                  | _qubits_of_ops(s.get("mpi_nonlocal_ops", [])))
            if getattr(u, "fallback", False):
                fallbacks += 1
        descs.append({
            "stage_id": u.compute_unit_id, "gates": u.gate_count,
            "local": local, "rank_nl": rn, "mpi_nl": mn,
            "n_steps": u.n_steps, "kind": u.kind, "active_qubits": sorted(aq),
        })
    return descs, fallbacks


def _max_local_run_gates(step_descs: list[dict]) -> int:
    best = run = 0
    for d in step_descs:
        if d["rank_nl"] == 0 and d["mpi_nl"] == 0 and d["gates"] > 0:
            run += d["gates"]
            best = max(best, run)
        else:
            run = 0
    return best


def build_plan_context(circuit_dict: dict, *, n: int, chunk_bits: int,
                       num_ranks: int, recovery: str,
                       compute_unit_min_gates: int = 4,
                       cost_model: dict | None = None) -> PlanContext:
    """Compile the circuit into the runner's steps + compute-unit plan."""
    from wenbo_engine.circuit.io import levelize, validate_circuit_dict
    from wenbo_engine.mpi.mpi_runner import _compile_steps

    circuit_dict = validate_circuit_dict(circuit_dict)
    k = chunk_bits
    p = int(round(math.log2(num_ranks))) if num_ranks >= 1 else 0
    n_local_bits = n - k - p
    n_chunks_total = 1 << (n - k)
    n_chunks_per_rank = max(1, n_chunks_total // max(1, num_ranks))
    chunk_bytes = (1 << k) * np.dtype(DTYPE).itemsize

    steps = _compile_steps(levelize(circuit_dict), k, n_local_bits)
    # deterministic per-mode MPI estimate from the real exchange planner
    mpi_estimates = estimate_mpi_exchange(
        steps, num_ranks=num_ranks, k=k, n_local_bits=n_local_bits,
        n_chunks_per_rank=n_chunks_per_rank, chunk_bytes=chunk_bytes)
    step_descs = _build_step_descriptors(steps)
    unit_descs, fallbacks = _build_unit_descriptors(
        steps, n_chunks_per_rank=n_chunks_per_rank,
        min_gates=compute_unit_min_gates)

    total_gates = sum(d["gates"] for d in step_descs)
    total_mpi_ops = sum(d["mpi_nl"] for d in step_descs)
    max_run = _max_local_run_gates(step_descs)

    return PlanContext(
        steps=step_descs, units=unit_descs,
        n_chunks_per_rank=n_chunks_per_rank, chunk_bytes=chunk_bytes,
        num_ranks=num_ranks, total_gates=total_gates,
        total_mpi_ops=total_mpi_ops, recovery=recovery,
        compute_unit_min_gates=compute_unit_min_gates,
        cost_model=cost_model or load_cost_model(),
        max_local_run_gates=max_run,
        has_fused_local_unit=(max_run >= compute_unit_min_gates),
        n_compute_unit_fallbacks=fallbacks,
        mpi_estimates=mpi_estimates,
    )


def _stage_plans_for(selected: StrategyCandidate, ctx: PlanContext
                     ) -> list[StagePlan]:
    """Per-stage plan for the SELECTED strategy."""
    passes = (ctx.units if selected.execution_mode == "compute_unit"
              else ctx.steps)
    C, B, R = ctx.n_chunks_per_rank, ctx.chunk_bytes, ctx.num_ranks
    # deterministic MPI total for the selected mode, attributed per stage in
    # proportion to each stage's share of MPI-nonlocal gates.
    mpi_est = (ctx.mpi_estimates or {}).get(selected.mpi_exchange_mode,
                                            {"sendrecv_count": 0,
                                             "mpi_bytes_sent": 0})
    total_mnl = sum(d["mpi_nl"] for d in passes) or 1
    plans: list[StagePlan] = []
    for d in passes:
        is_local = d["rank_nl"] == 0 and d["mpi_nl"] == 0
        direct_local = (selected.storage_layout == "extents"
                        and selected.extent_io_mode == "direct" and is_local)
        extent_mat = (2 * C * B * R if (selected.storage_layout == "extents"
                      and not direct_local) else 0)
        direct_chunks = C * R if direct_local else 0
        frac = d["mpi_nl"] / total_mnl if d["mpi_nl"] else 0.0
        sc = round(mpi_est["sendrecv_count"] * frac)
        mpi_bytes_stage = round(mpi_est["mpi_bytes_sent"] * frac)
        cost = recovery_aware_cost(
            bytes_read=C * B * R, bytes_written=C * B * R,
            mpi_bytes=mpi_bytes_stage, sendrecv_count=sc,
            kernel_bytes=d["gates"] * C * B * R, commits=1,
            extent_materialize_bytes=extent_mat,
            direct_extent_chunks=direct_chunks,
            layout_materialize_bytes=0, durable_bytes=0,
            recompute_bytes=C * B * R, model=ctx.cost_model)
        plans.append(StagePlan(
            stage_id=d["stage_id"], gates=d["gates"],
            active_qubits=d["active_qubits"],
            locality_summary={"local": d["local"], "rank_nonlocal": d["rank_nl"],
                              "mpi_nonlocal": d["mpi_nl"]},
            n_steps=d["n_steps"], kind=d["kind"],
            storage_layout=selected.storage_layout,
            execution_mode=selected.execution_mode,
            extent_io_mode=selected.extent_io_mode,
            mpi_exchange_mode=selected.mpi_exchange_mode,
            compute_unit_min_gates=selected.compute_unit_min_gates,
            commit_policy="per_stage_global_commit_record",
            durable_policy="none",
            estimated_nvme_read_bytes=C * B * R,
            estimated_nvme_write_bytes=C * B * R,
            estimated_mpi_bytes=mpi_bytes_stage,
            estimated_sendrecv_count=sc,
            estimated_kernel_time=cost["kernel_cost"],
            estimated_layout_materialization_cost=cost["layout_materialization_cost"],
            estimated_commit_cost=cost["commit_cost"],
            estimated_durable_checkpoint_cost=cost["durable_checkpoint_cost"],
            expected_recomputation_cost=cost["expected_recomputation_cost"],
        ))
    return plans


def _ram_block(*, n: int, chunk_bits: int, num_ranks: int, selected: dict,
               has_mpi: bool, ram_budget_gib: float | None,
               max_overlay_chunks: int | None,
               max_remote_buffer_gib: float | None,
               auto_chunk_bits: bool) -> dict:
    """RAM working-set model for the SELECTED strategy (capacity_planner)."""
    from wenbo_engine.planner import capacity_planner as cap
    exec_mode = selected["execution_mode"]
    mpi_mode = selected["mpi_exchange_mode"]
    budget_bytes = (ram_budget_gib * cap.GIB) if ram_budget_gib else 0.0
    max_remote_bytes = (max_remote_buffer_gib * cap.GIB
                        if max_remote_buffer_gib is not None else None)
    recommended = None
    if budget_bytes > 0:
        recommended = cap.recommend_chunk_bits(
            num_qubits=n, num_ranks=num_ranks, ram_budget_bytes=budget_bytes,
            execution_mode=exec_mode, mpi_exchange_mode=mpi_mode,
            max_overlay_chunks=max_overlay_chunks,
            max_remote_buffer_bytes=max_remote_bytes, has_mpi=has_mpi)
    eff_cb = recommended if (auto_chunk_bits and recommended) else chunk_bits
    est = cap.estimate_peak_ram(
        num_qubits=n, num_ranks=num_ranks, chunk_bits=eff_cb,
        execution_mode=exec_mode, mpi_exchange_mode=mpi_mode,
        bounded_overlay=auto_chunk_bits, max_overlay_chunks=max_overlay_chunks,
        max_remote_buffer_bytes=max_remote_bytes, has_mpi=has_mpi)
    peak = est["estimated_peak_ram_bytes"]
    return {
        "chunk_bits": eff_cb,
        "recommended_chunk_bits": recommended,
        "estimated_peak_ram_gib": round(peak / cap.GIB, 4),
        "ram_budget_gib": ram_budget_gib,
        "ram_feasible": (budget_bytes <= 0) or (peak <= budget_bytes),
        "auto_chunk_bits_enabled": bool(auto_chunk_bits),
        "has_mpi": has_mpi,
        "components_gib": {k: round(v / cap.GIB, 4) for k, v in est.items()
                           if k.endswith("_bytes")},
    }


def plan_recovery_aware(circuit_dict: dict, *, n: int, chunk_bits: int,
                        num_ranks: int, recovery: str,
                        compute_unit_min_gates: int = 4,
                        cost_model: dict | None = None,
                        ram_budget_gib: float | None = None,
                        max_overlay_chunks: int | None = None,
                        max_remote_buffer_gib: float | None = None,
                        auto_chunk_bits: bool = False,
                        kernel_backend: str = "auto") -> dict:
    """Run the full recovery-aware planner; return the plan + decision dict.

    ``kernel_backend`` is orthogonal to strategy selection (numpy/numba/auto is
    a per-process numerical-backend choice, not a layout/execution decision); it
    is recorded in the plan for traceability but does not affect the candidates.
    """
    ctx = build_plan_context(
        circuit_dict, n=n, chunk_bits=chunk_bits, num_ranks=num_ranks,
        recovery=recovery, compute_unit_min_gates=compute_unit_min_gates,
        cost_model=cost_model)

    candidates = enumerate_candidates(compute_unit_min_gates)
    estimates = [estimate_candidate(c, ctx) for c in candidates]
    estimates_by_name = {e["candidate"]: e for e in estimates}

    decision = select_strategy(candidates, estimates_by_name, ctx)
    sel_name = decision["selected_strategy"]["name"]
    selected = next(c for c in candidates if c.name == sel_name)
    stage_plans = _stage_plans_for(selected, ctx)

    ram = _ram_block(
        n=n, chunk_bits=chunk_bits, num_ranks=num_ranks,
        selected=decision["selected_strategy"],
        has_mpi=ctx.total_mpi_ops > 0, ram_budget_gib=ram_budget_gib,
        max_overlay_chunks=max_overlay_chunks,
        max_remote_buffer_gib=max_remote_buffer_gib,
        auto_chunk_bits=auto_chunk_bits)

    return {
        "planner": PLANNER_NAME,
        "hardware": {
            "n_qubits": n, "chunk_bits": chunk_bits, "num_ranks": num_ranks,
            "recovery": recovery,
            "n_chunks_per_rank": ctx.n_chunks_per_rank,
            "chunk_bytes": ctx.chunk_bytes,
        },
        "context_summary": {
            "n_steps": len(ctx.steps),
            "n_compute_units": len(ctx.units),
            "total_gates": ctx.total_gates,
            "total_mpi_nonlocal_ops": ctx.total_mpi_ops,
            "max_local_run_gates": ctx.max_local_run_gates,
            "has_fused_local_unit": ctx.has_fused_local_unit,
            "predicted_compute_unit_fallbacks": ctx.n_compute_unit_fallbacks,
        },
        "decision": decision,
        "candidates": estimates,
        "stage_plans": [sp.to_dict() for sp in stage_plans],
        "ram": ram,
        "kernel_backend": kernel_backend,
    }


def selected_run_params(plan: dict) -> dict:
    """Extract the runner kwargs implied by the selected strategy."""
    s = plan["decision"]["selected_strategy"]
    return {
        "storage_layout": s["storage_layout"],
        "execution_mode": s["execution_mode"],
        "extent_io_mode": s["extent_io_mode"],
        "mpi_exchange_mode": s["mpi_exchange_mode"],
        "compute_unit_min_gates": s["compute_unit_min_gates"],
    }


def attach_window_feasibility(plan: dict, circuit_dict: dict, *,
                              chunk_bits: int, num_ranks: int,
                              ram_budget_gib: float | None = None) -> dict:
    """Annotate a recovery-aware plan with an MPI-window feasibility summary.

    Analysis-only and **opt-in**: this is never called by
    :func:`plan_recovery_aware` and never changes strategy selection or runtime
    behavior.  It returns ``plan`` with an added ``"mpi_window_feasibility"``
    key (the report summary, candidates omitted) so callers that already hold a
    plan can surface the window prediction without a separate entry point.
    """
    from wenbo_engine.planner.mpi_window_report import (
        build_window_report, report_to_summary_json,
    )
    rep = build_window_report(circuit_dict, chunk_bits, num_ranks,
                              ram_budget_gib=ram_budget_gib)
    plan["mpi_window_feasibility"] = report_to_summary_json(rep)
    return plan
