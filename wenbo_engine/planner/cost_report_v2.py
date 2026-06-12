"""Predicted-vs-actual wall-time cost report for Planner v2.

Compares the v2-selected candidate's predicted wall-time breakdown against the
measured timing of the run.  Honest by construction: when an actual term was not
measured this run it is ``null`` (with the prediction still shown), and
``prediction_error_pct`` is ``null`` rather than fabricated.
"""
from __future__ import annotations


def _err_pct(predicted, actual):
    if predicted is None or actual is None:
        return None
    denom = abs(predicted) if abs(predicted) > 1e-9 else 1.0
    return round((actual - predicted) / denom * 100.0, 2)


# (predicted decision key, actual key)
_PAIRS = [
    ("predicted_wall_time", "wall_time"),
    ("predicted_io_time", "io_time"),
    ("predicted_kernel_time", "kernel_time"),
    ("predicted_pairwise_mpi_time", "pairwise_mpi_time"),
    ("predicted_collective_mpi_time", "collective_mpi_time"),
    ("predicted_window_leader_time", "window_leader_time"),
    ("predicted_commit_time", "commit_time"),
    ("predicted_peak_ram_gib", "peak_ram_gib"),
]


def build_cost_report_v2(plan: dict, actual: dict) -> dict:
    """Build cost_report_v2.json content (predicted vs measured wall time)."""
    dec = plan["decision"]
    available = bool(actual)
    metrics = {}
    for pkey, akey in _PAIRS:
        p = dec.get(pkey)
        a = actual.get(akey) if actual else None
        metrics[akey] = {
            "predicted": p,
            "actual": a,
            "actual_reason": (None if a is not None
                              else f"{akey} not measured this run"),
            "prediction_error_pct": _err_pct(p, a),
        }
    return {
        "planner": "recovery_aware_v2",
        "selected_strategy": dec["selected_strategy"]["name"],
        "selected_mpi_window_execution": dec["selected_mpi_window_execution"],
        "actual_metrics_available": available,
        "predicted_wall_time": dec.get("predicted_wall_time"),
        "actual_wall_time": actual.get("wall_time") if actual else None,
        "wall_time_prediction_error_pct": _err_pct(
            dec.get("predicted_wall_time"),
            actual.get("wall_time") if actual else None),
        "metrics": metrics,
        "predicted_wall_time_by_candidate":
            dec.get("predicted_wall_time_by_candidate", {}),
        "note": ("v2 ranks candidates by predicted WALL TIME (calibrated), not "
                 "bytes. A window candidate is selected only when its predicted "
                 "wall time is lowest — byte reduction alone never wins. Actual "
                 "terms come from this run's measured timing; unmeasured terms "
                 "are null, not fabricated."),
    }
