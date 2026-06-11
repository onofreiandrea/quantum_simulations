"""Planning layer for the out-of-core simulator.

This package hosts two independent, complementary planners:

* **Optimizer v2** — builds an execution *plan* for a circuit under a chosen
  *ablation mode* and computes deterministic, in-process *plan metrics*
  (predicted bytes read / written, MPI bytes, Sendrecv count, stages,
  commits, full-state passes, estimated runtime) from a stage cost model, so
  ablation modes can be compared offline without a cluster run.  Modes:
  ``current``, ``current_static_reorder``, ``stage_v2``, ``stage_v2_fusion``,
  ``stage_v2_placement_fusion``.  Every plan is semantically equivalent to the
  original circuit (verified against ``ref_dense.simulate``).

* **Capacity planner** — a pure, side-effect-free advisory layer that answers
  "what is the largest *exact* state-vector simulation feasible on this
  hardware, under this precision, rank count, local storage, RAM, and recovery
  policy?"  It does arithmetic over the same storage/memory model the runners
  use (double buffer, retained generations, optional durable checkpoint), so a
  run can be sized before any byte is written.  45q is just one scenario it can
  score — nothing is hardcoded to it.

The two share no symbols; both APIs are re-exported here.
"""
from __future__ import annotations

# ── Optimizer v2 ────────────────────────────────────────────────────────
from wenbo_engine.planner.optimizer_v2 import (
    ABLATION_MODES,
    Plan,
    Stage,
    HardwareConfig,
    build_plan,
    plan_metrics,
    ablation_report,
    plan_to_gates,
)
from wenbo_engine.planner.stage_cost_model import (
    DEFAULT_COST_MODEL,
    load_cost_model,
    stage_cost,
    recovery_aware_cost,
)
from wenbo_engine.planner.plan_serializer import (
    serialize_plan,
    deserialize_plan,
    plan_to_json,
    serialize_recovery_aware_plan,
    serialize_candidate_strategies,
    recovery_aware_plan_to_json,
)

# ── Recovery-aware planner v1 ───────────────────────────────────────────
from wenbo_engine.planner.stage_plan import StagePlan
from wenbo_engine.planner.strategy_candidate import (
    StrategyCandidate,
    PlanContext,
    enumerate_candidates,
    estimate_candidate,
)
from wenbo_engine.planner.strategy_selector import select_strategy
from wenbo_engine.planner.recovery_aware_planner import (
    PLANNER_NAME as RECOVERY_AWARE_V1,
    plan_recovery_aware,
    build_plan_context,
    selected_run_params,
)
from wenbo_engine.planner.cost_report import build_cost_report
from wenbo_engine.planner.mpi_window_report import (
    build_window_report,
    report_to_candidates_json,
    report_to_summary_json,
)

# ── Capacity planner ────────────────────────────────────────────────────
from wenbo_engine.planner.capacity_planner import (
    BYTES_PER_AMP,
    RECOVERY_MODES,
    RECOVERY_DEFAULTS,
    PlannerConfig,
    QubitFeasibility,
    state_size_bytes,
    per_rank_state_bytes,
    evaluate_qubits,
    max_feasible_qubits,
    recommend_recovery_mode,
    plan,
)

__all__ = [
    # optimizer v2
    "ABLATION_MODES",
    "Plan",
    "Stage",
    "HardwareConfig",
    "build_plan",
    "plan_metrics",
    "ablation_report",
    "plan_to_gates",
    "DEFAULT_COST_MODEL",
    "load_cost_model",
    "stage_cost",
    "recovery_aware_cost",
    "serialize_plan",
    "deserialize_plan",
    "plan_to_json",
    # recovery-aware planner v1
    "StagePlan",
    "StrategyCandidate",
    "PlanContext",
    "enumerate_candidates",
    "estimate_candidate",
    "select_strategy",
    "RECOVERY_AWARE_V1",
    "plan_recovery_aware",
    "build_plan_context",
    "selected_run_params",
    "build_cost_report",
    # MPI-window feasibility (analysis-only)
    "build_window_report",
    "report_to_candidates_json",
    "report_to_summary_json",
    "serialize_recovery_aware_plan",
    "serialize_candidate_strategies",
    "recovery_aware_plan_to_json",
    # capacity planner
    "BYTES_PER_AMP",
    "RECOVERY_MODES",
    "RECOVERY_DEFAULTS",
    "PlannerConfig",
    "QubitFeasibility",
    "state_size_bytes",
    "per_rank_state_bytes",
    "evaluate_qubits",
    "max_feasible_qubits",
    "recommend_recovery_mode",
    "plan",
]
