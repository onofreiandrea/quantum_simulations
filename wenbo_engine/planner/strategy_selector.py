"""Deterministic recovery-aware strategy selection.

Applies the planner-v1 selection rules to the candidate strategies and returns
a fully-explained decision.  Selection is deterministic: the same circuit +
hardware + recovery mode always yields the same choice (ties broken by the
fixed candidate order).

Rules:
  1. Use compute_unit only when the predicted local run length is above the
     compute-unit threshold (otherwise tiny fused units waste a commit each).
  2. Use direct extent I/O only when storage_layout=extents AND
     execution_mode=compute_unit.
  3. Use gate_aware MPI when MPI-nonlocal gates exist.
  4. Never use physical layout materialization unless its cost is explicitly
     counted (the cost model always counts a layout_materialization_cost term,
     so this is structurally guaranteed).
  5. Among the strategies that satisfy the safety rules, prefer the one that
     moves the least data / has the lowest predicted cost.
  6. If a cheaper candidate was rejected for a safety/recovery reason, the
     decision explains why the safer (possibly slower) strategy was chosen.
"""
from __future__ import annotations

from wenbo_engine.planner.strategy_candidate import StrategyCandidate, PlanContext

# The always-available safe baseline if no candidate satisfies every rule
# (e.g. MPI present but the only gate_aware candidates need generation recovery
# that isn't enabled).  chunks+step is supported under every recovery mode.
_SAFE_BASELINE = "chunks+step+naive"


def select_strategy(candidates: list[StrategyCandidate],
                    estimates_by_name: dict[str, dict],
                    ctx: PlanContext) -> dict:
    """Return the selection decision dict (selected / rejected / reason / costs)."""
    has_mpi = ctx.total_mpi_ops > 0
    long_local = ctx.has_fused_local_unit

    viable: list[StrategyCandidate] = []
    rejected: list[dict] = []

    for c in candidates:
        against: list[str] = []
        if c.storage_layout == "extents" and ctx.recovery != "generation":
            against.append("extents layout requires generation recovery "
                           f"(recovery={ctx.recovery!r})")
        if c.execution_mode == "compute_unit" and not long_local:
            against.append(
                "rule 1: compute_unit needs a predicted local run >= "
                f"{ctx.compute_unit_min_gates} gates "
                f"(max predicted local run = {ctx.max_local_run_gates})")
        if has_mpi and c.mpi_exchange_mode != "gate_aware":
            against.append("rule 3: MPI-nonlocal gates present → gate_aware "
                           "exchange required")
        if c.extent_io_mode == "direct" and not (
                c.storage_layout == "extents"
                and c.execution_mode == "compute_unit"):
            against.append("rule 2: direct extent I/O requires "
                           "extents + compute_unit")
        if against:
            rejected.append({"name": c.name, "strategy": c.to_dict(),
                             "reasons": against,
                             "estimated_total_cost":
                                 estimates_by_name[c.name]["estimated_total_cost"]})
        else:
            viable.append(c)

    # Rule 5: lowest predicted cost among viable; deterministic tie-break by
    # the fixed candidate order.
    order = {c.name: i for i, c in enumerate(candidates)}
    relaxed = False
    if not viable:
        # No candidate satisfies every rule under this recovery mode — fall back
        # to the safe baseline (supported everywhere) and record the relaxation.
        relaxed = True
        viable = [c for c in candidates if c.name == _SAFE_BASELINE] or list(candidates)
        rejected = [r for r in rejected if r["name"] not in
                    {c.name for c in viable}]

    viable.sort(key=lambda c: (estimates_by_name[c.name]["estimated_total_cost"],
                               order[c.name]))
    selected = viable[0]
    sel_cost = estimates_by_name[selected.name]["estimated_total_cost"]

    # Rule 6: was any rejected candidate predicted cheaper than the selection?
    cheaper_rejected = [r for r in rejected
                        if r["estimated_total_cost"] < sel_cost - 1e-12]

    reason = _explain(selected, ctx, has_mpi, long_local,
                      cheaper_rejected, relaxed)

    return {
        "selected_strategy": {"name": selected.name, **selected.to_dict()},
        "rejected_candidates": rejected,
        "reason_for_selection": reason,
        "predicted_cost_by_candidate": {
            name: est["estimated_total_cost"]
            for name, est in estimates_by_name.items()
        },
        "selection_inputs": {
            "has_mpi_nonlocal": has_mpi,
            "has_long_local_run": long_local,
            "max_predicted_local_run_gates": ctx.max_local_run_gates,
            "compute_unit_min_gates": ctx.compute_unit_min_gates,
            "recovery": ctx.recovery,
            "relaxed_to_baseline": relaxed,
        },
    }


def _explain(selected: StrategyCandidate, ctx: PlanContext, has_mpi: bool,
             long_local: bool, cheaper_rejected: list[dict],
             relaxed: bool) -> str:
    bits: list[str] = [f"Selected {selected.name}."]
    if selected.execution_mode == "compute_unit":
        bits.append(f"A local run of {ctx.max_local_run_gates} gates "
                    f"(>= threshold {ctx.compute_unit_min_gates}) makes "
                    "compute-unit fusion worthwhile (rule 1); short local "
                    "fragments still fall back to the step path at runtime.")
        if selected.extent_io_mode == "direct":
            bits.append("With extents + compute_unit, direct extent I/O avoids "
                        "the materialize temp-chunk round trip (rule 2, rule 5).")
    else:
        bits.append("No local run reaches the compute-unit threshold, so the "
                    "step path is used to avoid tiny, commit-heavy compute "
                    "units (rule 1).")
    if has_mpi:
        bits.append("MPI-nonlocal gates are present, so gate_aware exchange is "
                    "required and MPI stress is preserved (rule 3)." if
                    selected.mpi_exchange_mode == "gate_aware" else
                    "MPI-nonlocal gates are present.")
    else:
        bits.append("No MPI-nonlocal gates, so the workload stays MPI-light.")
    if relaxed:
        bits.append("No candidate satisfied every rule under this recovery "
                    "mode; relaxed to the safe baseline.")
    if cheaper_rejected:
        names = ", ".join(f"{r['name']} ({r['reasons'][0]})"
                          for r in cheaper_rejected)
        bits.append("Cheaper candidates were rejected for safety/recovery "
                    f"reasons and NOT selected: {names} (rule 6).")
    return " ".join(bits)
