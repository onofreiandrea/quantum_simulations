"""Recovery-aware adaptive planner v2.

v2 is the decision layer that chooses among the **existing** execution
mechanisms (layout, execution mode, extent I/O, MPI exchange, MPI-window
execution, kernel backend, chunk_bits, RAM budgets, commit policy) by predicting
**wall time and recovery risk** — not bytes — using the calibrated telemetry the
previous branch measures.

It introduces no new executor, no new storage format, and changes no recovery,
MPI, or kernel code.  It reuses v1's grounded :class:`PlanContext` (real
compiled steps + compute-unit plan + deterministic MPI estimate) and v1's
:func:`estimate_candidate` byte quantities, then ranks candidates with
:mod:`wenbo_engine.planner.cost_model_v2`.

Crucially: an ``mpi_window_execution=safe`` candidate is selected only when its
*predicted wall time* is lower — the cluster showed a window can cut bytes ~11×
yet be slower because collective scatter + leader compute dominate.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from wenbo_engine.planner.recovery_aware_planner import (
    build_plan_context, _ram_block,
)
from wenbo_engine.planner.strategy_candidate import (
    StrategyCandidate, estimate_candidate,
)
from wenbo_engine.planner import cost_model_v2 as cm2

PLANNER_NAME = "recovery_aware_v2"


@dataclass(frozen=True)
class StrategyCandidateV2:
    name: str
    storage_layout: str
    execution_mode: str
    extent_io_mode: str
    mpi_exchange_mode: str
    mpi_window_execution: str            # "off" | "safe"
    compute_unit_min_gates: int = 4

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "storage_layout": self.storage_layout,
            "execution_mode": self.execution_mode,
            "extent_io_mode": self.extent_io_mode,
            "mpi_exchange_mode": self.mpi_exchange_mode,
            "mpi_window_execution": self.mpi_window_execution,
            "compute_unit_min_gates": self.compute_unit_min_gates,
        }

    def as_v1(self) -> StrategyCandidate:
        """The v1 candidate (no window field) for byte-quantity estimation."""
        return StrategyCandidate(
            self.name, self.storage_layout, self.execution_mode,
            self.extent_io_mode, self.mpi_exchange_mode,
            self.compute_unit_min_gates)


def enumerate_candidates_v2(compute_unit_min_gates: int = 4
                            ) -> list[StrategyCandidateV2]:
    """The v2 candidate set (deterministic order)."""
    mg = compute_unit_min_gates
    return [
        StrategyCandidateV2("chunks+step+naive+window_off", "chunks", "step",
                            "materialize", "naive", "off", mg),
        StrategyCandidateV2("chunks+step+gate_aware+window_off", "chunks",
                            "step", "materialize", "gate_aware", "off", mg),
        StrategyCandidateV2("chunks+step+gate_aware+window_safe", "chunks",
                            "step", "materialize", "gate_aware", "safe", mg),
        StrategyCandidateV2("extents+step+gate_aware+window_off", "extents",
                            "step", "materialize", "gate_aware", "off", mg),
        StrategyCandidateV2(
            "extents+compute_unit+materialize+gate_aware+window_off", "extents",
            "compute_unit", "materialize", "gate_aware", "off", mg),
        StrategyCandidateV2(
            "extents+compute_unit+direct+gate_aware+window_off", "extents",
            "compute_unit", "direct", "gate_aware", "off", mg),
    ]


def _choose_backend(kernel_backend: str, cal: dict) -> tuple[str, bool, str]:
    """Decide the numerical backend; return (used, numba_available, reason)."""
    from wenbo_engine.kernel.backend import numba_available
    avail = numba_available()
    speedup = cal.get("numba_speedup_factor") or 1.0
    if kernel_backend == "numpy":
        return "numpy", avail, "explicitly requested numpy"
    if kernel_backend == "numba":
        if avail:
            return "numba", True, "explicitly requested numba (available)"
        return "numpy", False, "numba requested but unavailable → numpy fallback"
    # auto
    if avail:
        return ("numba", True,
                f"auto: numba available (calibrated speedup x{speedup:g})")
    return "numpy", False, "auto: numba unavailable → numpy"


def _tiny_compute_units(ctx) -> bool:
    """True when compute_unit fusion would produce only tiny local fragments.

    If the longest local run is below the min-gates threshold there is no real
    fusion benefit (each 'unit' is ~one step); v2 then prefers step execution.
    """
    return ctx.max_local_run_gates < ctx.compute_unit_min_gates


def plan_recovery_aware_v2(circuit_dict: dict, *, n: int, chunk_bits: int,
                           num_ranks: int, recovery: str,
                           compute_unit_min_gates: int = 4,
                           ram_budget_gib: float | None = None,
                           max_overlay_chunks: int | None = None,
                           max_remote_buffer_gib: float | None = None,
                           auto_chunk_bits: bool = False,
                           kernel_backend: str = "auto",
                           calibration: dict | None = None,
                           measured_cost_model: dict | None = None) -> dict:
    """Build the v2 plan + decision; return a dict with the four artifact blocks.

    ``measured_cost_model`` is an optional prior-run ``cost_model.json``
    ``constants`` block; when present its non-null values calibrate the model.
    ``calibration`` (tests) overrides everything.
    """
    ctx = build_plan_context(
        circuit_dict, n=n, chunk_bits=chunk_bits, num_ranks=num_ranks,
        recovery=recovery, compute_unit_min_gates=compute_unit_min_gates)

    cal = dict(cm2.load_v2_calibration(measured_cost_model))
    if calibration:
        cal.update(calibration)

    backend_used, numba_avail, backend_reason = _choose_backend(kernel_backend, cal)

    # Window analytics (shared by every candidate's prediction).
    p = int(round(math.log2(num_ranks))) if num_ranks >= 1 else 0
    n_local_bits = n - chunk_bits - p
    window = cm2.analyze_window_info(
        _ctx_steps_as_compiled(circuit_dict, chunk_bits, n_local_bits),
        k=chunk_bits, n_local_bits=n_local_bits, num_ranks=num_ranks,
        chunk_size=(1 << chunk_bits), n_chunks_per_rank=ctx.n_chunks_per_rank,
        ram_budget_gib=ram_budget_gib)

    candidates = enumerate_candidates_v2(compute_unit_min_gates)
    tiny_units = _tiny_compute_units(ctx)

    estimates = []
    for c in candidates:
        base = estimate_candidate(c.as_v1(), ctx)
        ram = _ram_block(
            n=n, chunk_bits=chunk_bits, num_ranks=num_ranks,
            selected=c.to_dict(), has_mpi=ctx.total_mpi_ops > 0,
            ram_budget_gib=ram_budget_gib, max_overlay_chunks=max_overlay_chunks,
            max_remote_buffer_gib=max_remote_buffer_gib,
            auto_chunk_bits=auto_chunk_bits)
        # window RAM must also fit when the window is on
        ram_feasible = bool(ram["ram_feasible"])
        if c.mpi_window_execution == "safe" and window.executable:
            if ram_budget_gib is not None and window.estimated_ram_gib > ram_budget_gib:
                ram_feasible = False
        pred = cm2.predict_wall_time(
            base=base, ctx=ctx, candidate=c, cal=cal, window=window,
            peak_ram_gib=ram["estimated_peak_ram_gib"],
            ram_feasible=ram_feasible, kernel_backend=backend_used,
            numba_available=numba_avail)
        # storage feasibility: extents/direct always supported here
        storage_feasible = True
        est = {
            "candidate": c.name,
            "strategy": c.to_dict(),
            "byte_quantities": {
                "bytes_read": base["bytes_read"],
                "bytes_written": base["bytes_written"],
                "mpi_bytes_sent": base["mpi_bytes_sent"],
                "sendrecv_count": base["sendrecv_count"],
                "commit_count": base["commit_count"],
            },
            "window_executable": window.executable,
            "window_on": pred["window_on"],
            "storage_feasible": storage_feasible,
            "tiny_compute_units": (tiny_units
                                   if c.execution_mode == "compute_unit" else False),
            "ram": ram,
            **pred,
        }
        estimates.append(est)

    decision = _select(estimates, candidates, window, tiny_units, backend_used,
                       backend_reason, cal, ctx, chunk_bits, ram_budget_gib,
                       auto_chunk_bits)
    return {
        "planner": PLANNER_NAME,
        "hardware": {
            "n_qubits": n, "chunk_bits": chunk_bits, "num_ranks": num_ranks,
            "recovery": recovery, "n_chunks_per_rank": ctx.n_chunks_per_rank,
            "chunk_bytes": ctx.chunk_bytes,
        },
        "calibration": cal,
        "window_info": window.__dict__,
        "candidates": estimates,
        "decision": decision,
    }


def _ctx_steps_as_compiled(circuit_dict, chunk_bits, n_local_bits):
    from wenbo_engine.circuit.io import levelize, validate_circuit_dict
    from wenbo_engine.mpi.mpi_runner import _compile_steps
    return _compile_steps(levelize(validate_circuit_dict(circuit_dict)),
                          chunk_bits, n_local_bits)


def _select(estimates, candidates, window, tiny_units, backend_used,
            backend_reason, cal, ctx, chunk_bits, ram_budget_gib,
            auto_chunk_bits) -> dict:
    """Pick the min predicted-wall-time feasible candidate; explain everything."""
    by_name = {c.name: c for c in candidates}
    feasible = [e for e in estimates if e["ram_feasible"] and e["storage_feasible"]]
    pool = feasible or estimates  # if nothing feasible, still report (penalised)

    # Penalise tiny-compute-unit candidates so they don't win by a hair over step.
    def sort_key(e):
        penalty = 0.0
        if e["tiny_compute_units"]:
            penalty += 1e6   # avoid inefficient tiny fragments (still below infeasible)
        return (e["predicted_wall_time"] + penalty, e["candidate"])

    ranked = sorted(pool, key=sort_key)
    winner = ranked[0]
    sel = by_name[winner["candidate"]]

    rejected = []
    for e in ranked[1:]:
        rejected.append({
            "candidate": e["candidate"],
            "predicted_wall_time": e["predicted_wall_time"],
            "reason": _rejection_reason(e, winner),
        })

    reason = _selection_reason(winner, window, tiny_units)
    commit_policy = ("per_mpi_window_global_commit_record"
                     if winner["window_on"] else
                     ("per_compute_unit_global_commit_record"
                      if sel.execution_mode == "compute_unit"
                      else "per_step_global_commit_record"))

    return {
        "selected_strategy": sel.to_dict(),
        "selected_kernel_backend": backend_used,
        "selected_chunk_bits": chunk_bits,
        "selected_execution_mode": sel.execution_mode,
        "selected_storage_layout": sel.storage_layout,
        "selected_extent_io_mode": sel.extent_io_mode,
        "selected_mpi_exchange_mode": sel.mpi_exchange_mode,
        "selected_mpi_window_execution": sel.mpi_window_execution,
        "selected_ram_budget_gib": ram_budget_gib,
        "selected_commit_policy": commit_policy,
        "recommended_chunk_bits": winner.get("ram", {}).get("recommended_chunk_bits"),
        "backend_decision_reason": backend_reason,
        "reason_for_selection": reason,
        "rejected_candidates": [r["candidate"] for r in rejected],
        "reason_for_each_rejection": {r["candidate"]: r["reason"]
                                      for r in rejected},
        "predicted_wall_time_by_candidate": {
            e["candidate"]: e["predicted_wall_time"] for e in estimates},
        "predicted_cost_breakdown_by_candidate": {
            e["candidate"]: {
                k: e[k] for k in (
                    "predicted_io_time", "predicted_kernel_time",
                    "predicted_pairwise_mpi_time", "predicted_collective_mpi_time",
                    "predicted_window_leader_time", "predicted_window_segment_time",
                    "predicted_commit_time", "predicted_norm_time",
                    "predicted_recomputation_cost", "predicted_peak_ram_gib",
                    "ram_feasible")}
            for e in estimates},
        # selected-candidate scalar predictions (convenience for the report)
        "predicted_io_time": winner["predicted_io_time"],
        "predicted_kernel_time": winner["predicted_kernel_time"],
        "predicted_pairwise_mpi_time": winner["predicted_pairwise_mpi_time"],
        "predicted_collective_mpi_time": winner["predicted_collective_mpi_time"],
        "predicted_window_leader_time": winner["predicted_window_leader_time"],
        "predicted_commit_time": winner["predicted_commit_time"],
        "predicted_recomputation_cost": winner["predicted_recomputation_cost"],
        "predicted_peak_ram_gib": winner["predicted_peak_ram_gib"],
        "ram_feasible": winner["ram_feasible"],
        "storage_feasible": winner["storage_feasible"],
        "predicted_wall_time": winner["predicted_wall_time"],
    }


def _rejection_reason(e, winner) -> str:
    if not e["ram_feasible"]:
        return (f"RAM-infeasible (predicted peak {e['predicted_peak_ram_gib']} GiB "
                f"exceeds budget)")
    if e["tiny_compute_units"]:
        return ("compute_unit fusion yields only tiny local fragments "
                "(no real fusion benefit) — step preferred")
    if e["window_on"] and not winner["window_on"]:
        return (f"MPI window predicted slower: wall {e['predicted_wall_time']}s "
                f"(collective {e['predicted_collective_mpi_time']}s + leader "
                f"{e['predicted_window_leader_time']}s + segment "
                f"{e['predicted_window_segment_time']}s) vs winner "
                f"{winner['predicted_wall_time']}s — bytes saved ≠ wall-time win")
    dt = round(e["predicted_wall_time"] - winner["predicted_wall_time"], 6)
    return f"higher predicted wall time (+{dt}s vs selected)"


def _selection_reason(winner, window, tiny_units) -> str:
    parts = [f"lowest predicted wall time ({winner['predicted_wall_time']}s)"]
    if winner["window_on"]:
        parts.append("MPI window selected: predicted collective+leader+segment "
                     "cost is lower than per-step pairwise here")
    elif window.executable:
        parts.append("MPI window available but predicted SLOWER than per-step "
                     "(bytes saved would not reduce wall time) → window off")
    else:
        parts.append(f"no executable true-mixing window ({window.reason}) "
                     "→ window off")
    if winner["strategy"]["execution_mode"] == "compute_unit":
        parts.append("compute_unit fusion has real local runs to fuse")
    elif tiny_units:
        parts.append("avoided compute_unit (only tiny local fragments)")
    return "; ".join(parts)


def selected_run_params_v2(plan: dict) -> dict:
    """Runner kwargs implied by the v2-selected strategy (incl. window)."""
    d = plan["decision"]
    return {
        "storage_layout": d["selected_storage_layout"],
        "execution_mode": d["selected_execution_mode"],
        "extent_io_mode": d["selected_extent_io_mode"],
        "mpi_exchange_mode": d["selected_mpi_exchange_mode"],
        "mpi_window_execution": d["selected_mpi_window_execution"],
        "compute_unit_min_gates": d["selected_strategy"]["compute_unit_min_gates"],
        "kernel_backend": d["selected_kernel_backend"],
    }
