"""Optimizer v2 — a measurable data-movement / communication planner.

This package builds an execution *plan* for a circuit under a chosen
*ablation mode* and computes deterministic, in-process *plan metrics*
(predicted bytes read / written, MPI bytes, Sendrecv count, number of
stages, number of commits, full-state passes, estimated runtime) from a
stage cost model.  The metrics are derived purely from the plan's
quantities and a hardware ``cost_model`` — no cluster run is required —
so an ablation between modes can be compared and asserted offline.

The five selectable ablation modes are:

  ``current``                     today's behavior (levelize + classify,
                                  no static reorder).
  ``current_static_reorder``      + :func:`reorder_qubits`.
  ``stage_v2``                    new staging (Atlas-style local sets).
  ``stage_v2_fusion``             new staging + 1Q fusion / level batching.
  ``stage_v2_placement_fusion``   new staging + activity-based placement
                                  + fusion.

Public surface:

  * :func:`build_plan` — build a :class:`~.optimizer_v2.Plan` for one mode.
  * :func:`plan_metrics` — compute the metric dict for a plan.
  * :func:`ablation_report` — compare ALL modes for one circuit + hw config.
  * :data:`ABLATION_MODES` — the ordered list of mode names.
  * :func:`serialize_plan` / :func:`deserialize_plan` — deterministic JSON.

Correctness over cleverness: every plan is semantically equivalent to the
original circuit (verified in the test-suite against ``ref_dense.simulate``).
"""
from __future__ import annotations

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
)
from wenbo_engine.planner.plan_serializer import (
    serialize_plan,
    deserialize_plan,
    plan_to_json,
)

__all__ = [
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
    "serialize_plan",
    "deserialize_plan",
    "plan_to_json",
]
