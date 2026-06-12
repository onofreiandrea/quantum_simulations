"""Predicted-vs-actual cost report for the recovery-aware planner v1.

After a run, compares the SELECTED candidate's predicted metrics against the
measured metrics and computes a per-metric prediction error.

Aggregation is explicit: byte metrics are reported at BOTH the cluster and the
per-rank level, and the error is always computed cluster-to-cluster (never a
cluster prediction against a rank-0 actual).  MPI estimates are deterministic
(exchange-planner-derived); the method is recorded so consumers can see they
are not heuristic.  When the caller has no measured metrics (e.g.
``run_experiment``), ``actual_metrics_available`` is ``false`` and the report is
predicted-only rather than silently incomplete.
"""
from __future__ import annotations

# Scalar metrics compared directly (already cluster-aggregated on both sides).
_SCALAR = (
    "read_ops", "write_ops", "temporary_chunk_files_created",
    "mpi_bytes_sent", "sendrecv_count", "kernel_time", "commit_count",
)


def _error_pct(predicted, actual):
    if predicted is None or actual is None:
        return None
    denom = abs(predicted) if abs(predicted) > 1e-12 else 1.0
    return round((actual - predicted) / denom * 100.0, 2)


# Measured timing fields surfaced in the cost report's timing block.  Planner
# v2's predicted timing is not implemented on this branch, so each predicted
# value is null-with-reason while the ACTUAL measured value is reported.
_TIMING_FIELDS = (
    "local_kernel_time", "nonlocal_kernel_time",
    "mpi_pairwise_sendrecv_time", "mpi_collective_gather_time",
    "mpi_collective_scatter_time", "mpi_window_leader_compute_time",
    "mpi_window_segment_time", "direct_extent_read_time",
    "direct_extent_write_time", "extent_materialize_time", "extent_pack_time",
    "commit_time", "norm_time", "numba_compile_time",
)

_NO_PREDICTED_TIMING = ("Planner v2 predicted timing is not implemented on this "
                        "branch (calibration-only); actual is measured.")


def build_timing_report(measured: dict) -> dict:
    """Predicted-vs-actual timing block for cost_report.json.

    Actual values come from the measured run; predicted is null-with-reason
    until Planner v2 lands.  A measured field that is absent is reported as
    null with its own reason (never silently omitted).
    """
    block = {}
    for f in _TIMING_FIELDS:
        actual = measured.get(f)
        block[f] = {
            "predicted": None,
            "predicted_reason": _NO_PREDICTED_TIMING,
            "actual": actual,
            "actual_reason": (None if actual is not None
                              else f"{f} not measured this run"),
            "prediction_error_pct": None,
        }
    return block


def build_cost_report(plan: dict, actual: dict) -> dict:
    """Build cost_report.json content.

    ``plan`` is the dict from :func:`plan_recovery_aware`; ``actual`` is the
    measured-metrics dict (empty ⇒ predicted-only).  Expected ``actual`` keys:
    the scalars in :data:`_SCALAR`, the labelled byte fields
    ``bytes_read_cluster`` / ``bytes_written_cluster`` /
    ``bytes_read_per_rank`` / ``bytes_written_per_rank``, plus optional
    ``num_ranks`` / ``wall_time`` / ``work_time``.
    """
    sel_name = plan["decision"]["selected_strategy"]["name"]
    by_name = {c["candidate"]: c for c in plan["candidates"]}
    predicted = by_name.get(sel_name, {})
    available = bool(actual)

    # ── byte metrics: labelled cluster + per-rank, error at cluster level ──
    bytes_block = {}
    for which in ("read", "written"):
        p_cluster = predicted.get(f"predicted_cluster_bytes_{which}")
        p_per_rank = predicted.get(f"predicted_per_rank_bytes_{which}")
        a_cluster = actual.get(f"bytes_{which}_cluster")
        a_per_rank = actual.get(f"bytes_{which}_per_rank")
        bytes_block[f"bytes_{which}"] = {
            "predicted_cluster": p_cluster,
            "actual_cluster": a_cluster,
            "predicted_per_rank": p_per_rank,
            "actual_per_rank": a_per_rank,
            "aggregation_compared": "cluster",
            "prediction_error_pct": _error_pct(p_cluster, a_cluster),
        }

    # ── scalar metrics (cluster-aggregated on both sides) ─────────────────
    metrics = dict(bytes_block)
    for key in _SCALAR:
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
        "actual_metrics_available": available,
        "mpi_estimate_method": predicted.get("mpi_estimate_method",
                                             "exchange_planner"),
        "mpi_estimate_is_heuristic": False,
        "predicted_total_cost": predicted.get("estimated_total_cost"),
        "actual_wall_time": actual.get("wall_time"),
        "actual_work_time": actual.get("work_time"),
        "num_ranks": actual.get("num_ranks"),
        "metrics": metrics,
        "timing": build_timing_report(actual),
        "predicted_cost_by_candidate":
            plan["decision"].get("predicted_cost_by_candidate", {}),
        "note": ("Predictions come from the analytical v1 cost model; MPI "
                 "exchange counts/bytes are deterministic (exchange-planner-"
                 "derived, not heuristic). Byte errors are computed "
                 "cluster-to-cluster. Direct-extent local I/O is not "
                 "chunk-file-instrumented, so its read/write ops are 0 by "
                 "construction (predicted and actual)."),
    }
