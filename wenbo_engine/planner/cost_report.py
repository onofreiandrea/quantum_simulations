"""Predicted-vs-actual cost report for the recovery-aware planner v1.

After a run, compares the SELECTED candidate's predicted metrics against the
measured metrics and computes a per-metric prediction error.  The estimates are
a deliberately-simple analytical model (this is v1), so errors are expected and
reported honestly — never hidden.
"""
from __future__ import annotations

# Metrics compared predicted-vs-actual.  Each must be present in the selected
# candidate's estimate AND obtainable from the measured run.
_COMPARED = (
    "bytes_read", "bytes_written", "read_ops", "write_ops",
    "temporary_chunk_files_created", "mpi_bytes_sent", "sendrecv_count",
    "kernel_time", "commit_count",
)


def _error_pct(predicted, actual):
    if predicted is None or actual is None:
        return None
    denom = abs(predicted) if abs(predicted) > 1e-12 else 1.0
    return round((actual - predicted) / denom * 100.0, 2)


def build_cost_report(plan: dict, actual: dict) -> dict:
    """Build cost_report.json content.

    ``plan`` is the dict from :func:`plan_recovery_aware`; ``actual`` is the
    measured-metrics dict (keys in :data:`_COMPARED` plus optional ``wall_time``
    / ``work_time``).  Returns predicted, actual and per-metric error.
    """
    sel_name = plan["decision"]["selected_strategy"]["name"]
    by_name = {c["candidate"]: c for c in plan["candidates"]}
    predicted = by_name.get(sel_name, {})

    metrics = {}
    for key in _COMPARED:
        p = predicted.get(key)
        a = actual.get(key)
        metrics[key] = {
            "predicted": p,
            "actual": a,
            "prediction_error_pct": _error_pct(p, a),
        }

    return {
        "planner": plan.get("planner", "recovery_aware_v1"),
        "selected_strategy": sel_name,
        "predicted_total_cost": predicted.get("estimated_total_cost"),
        "actual_wall_time": actual.get("wall_time"),
        "actual_work_time": actual.get("work_time"),
        "metrics": metrics,
        "predicted_cost_by_candidate":
            plan["decision"].get("predicted_cost_by_candidate", {}),
        "note": ("Predictions come from the analytical v1 cost model; per-metric "
                 "errors are reported, not minimized. Direct-extent local I/O is "
                 "not chunk-file-instrumented, so its read/write ops are 0 by "
                 "construction (predicted and actual)."),
    }
