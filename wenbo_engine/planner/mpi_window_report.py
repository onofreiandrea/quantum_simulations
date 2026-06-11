"""Build the MPI-window feasibility *report* (analysis only).

Wraps :func:`wenbo_engine.mpi.window_planner.analyze_windows` into a report that
keeps the five tradeoffs **separate** — bytes saved, Sendrecv calls saved,
commits saved, extra RAM required, extra recomputation risk after a crash — and
makes an honest recommendation on whether a future gather/apply/scatter
executor is worth implementing.

No execution, no remote-cache reuse across steps, no commit, no recovery
changes.  Reads the *measured* run telemetry (when supplied) only to report the
true baseline alongside the analytic prediction.
"""
from __future__ import annotations

from wenbo_engine.mpi import window_planner
from wenbo_engine.mpi import window_cost_model as cm


def build_window_report(circuit_dict: dict, chunk_bits: int, num_ranks: int,
                        ram_budget_gib: float | None = None,
                        measured: dict | None = None,
                        mpi_telemetry: dict | None = None,
                        weights: cm.CostWeights | None = None) -> dict:
    """Return the full window-feasibility report dict.

    ``measured`` (optional) carries the real run's authoritative
    ``sendrecv_count`` and ``mpi_bytes_sent`` (cluster-summed) so the report's
    ``baseline_*`` fields equal what actually happened; ``mpi_telemetry`` adds
    the measured repeated/adjacent remote-fetch counts for cross-checking.
    """
    analysis = window_planner.analyze_windows(
        circuit_dict, chunk_bits, num_ranks, ram_budget_gib=ram_budget_gib,
        weights=weights)
    candidates = analysis["candidates"]
    summary = analysis["summary"]

    feasible = [c for c in candidates if c["safe_to_execute_future"]]
    best = None
    if summary["best_window_id"] is not None:
        best = next(c for c in candidates
                    if c["window_id"] == summary["best_window_id"])

    measured = measured or {}
    measured_baseline = {
        "sendrecv_count": measured.get("sendrecv_count"),
        "mpi_bytes_sent": measured.get("mpi_bytes_sent"),
    }

    # Analytic baseline over EVERY remote-requiring MPI step (modelled on one
    # rank), plus its cluster projection (× num_ranks).  By construction this
    # equals the real run's measured Sendrecv/byte telemetry — the cross-check
    # the report exposes.  ``windowed_*`` reports the portion inside candidate
    # windows (what a window executor would actually touch).
    fb = analysis["full_baseline"]
    win_sr_rank = sum(c["estimated_baseline_sendrecv"] for c in candidates)
    win_by_rank = sum(c["estimated_baseline_mpi_bytes"] for c in candidates)
    analytic_baseline = {
        "modelled_rank": analysis["layout"]["modelled_rank"],
        "sendrecv_per_rank": fb["sendrecv_per_rank"],
        "mpi_bytes_per_rank": fb["mpi_bytes_per_rank"],
        "sendrecv_cluster": fb["sendrecv_cluster"],
        "mpi_bytes_cluster": fb["mpi_bytes_cluster"],
        "windowed_sendrecv_per_rank": win_sr_rank,
        "windowed_mpi_bytes_per_rank": win_by_rank,
        "covers_all_mpi_steps": _covers_all_mpi_steps(analysis, mpi_telemetry),
        "note": ("cluster = per-rank × num_ranks over all remote-requiring MPI "
                 "steps; equals measured telemetry"),
    }
    mt = mpi_telemetry or {}
    measured_telemetry = {
        "repeated_remote_chunk_fetches":
            mt.get("repeated_remote_chunk_fetches"),
        "repeated_remote_chunk_fetches_adjacent_steps":
            mt.get("repeated_remote_chunk_fetches_adjacent_steps"),
        "distinct_remote_chunks_per_rank":
            mt.get("distinct_remote_chunks_per_rank"),
        "mpi_steps": mt.get("mpi_steps"),
    }

    # The five SEPARATED tradeoffs (additive across feasible candidates).
    tradeoffs = {
        "bytes_saved": summary["total_estimated_mpi_byte_reduction"],
        "sendrecv_calls_saved": summary["total_estimated_sendrecv_reduction"],
        "commits_saved": summary["total_commit_reduction"],
        "extra_ram_gib_required": summary["max_extra_ram_gib"],
        "extra_recomputation_cost_after_crash":
            summary["total_expected_recomputation_cost_increase"],
        "repeated_remote_fetches_avoided":
            summary["total_repeated_remote_fetches_avoided"],
    }

    worth, reason = _recommend(summary, tradeoffs, ram_budget_gib)

    return {
        "analysis_mode": "report",
        "executor_implemented": False,           # this branch analyses only
        "runtime_execution_changed": False,
        "remote_cache_scope": "step",            # unchanged; no cross-step reuse
        "layout": analysis["layout"],
        "num_candidate_windows": summary["num_candidate_windows"],
        "num_feasible_windows": summary["num_feasible_windows"],
        "diagonal_gates_in_mpi_steps": summary["diagonal_gates_in_mpi_steps"],
        "best_candidate": best,
        "tradeoffs": tradeoffs,
        "baseline_measured": measured_baseline,
        "baseline_analytic": analytic_baseline,
        "measured_telemetry": measured_telemetry,
        "executor_worth_implementing": worth,
        "recommendation": ("implement_executor" if worth
                           else "do_not_implement_executor_yet"),
        "recommendation_reason": reason,
        "summary": summary,
        "candidates": candidates,
    }


def _covers_all_mpi_steps(analysis: dict,
                          mpi_telemetry: dict | None) -> bool | None:
    """Whether candidate windows cover every remote-requiring MPI step.

    ``None`` when telemetry is unavailable.  Compares total windowed MPI steps
    against the measured ``mpi_steps`` (the runner counts a step as an MPI step
    only when it has remote-requiring gates after the diagonal fast path).
    """
    if not mpi_telemetry or "mpi_steps" not in mpi_telemetry:
        return None
    windowed = sum(c["mpi_steps_in_window"] for c in analysis["candidates"])
    return windowed == mpi_telemetry["mpi_steps"]


def _recommend(summary: dict, tradeoffs: dict,
               ram_budget_gib: float | None) -> tuple[bool, str]:
    """Honest implement / do-not-implement decision with a reason string."""
    if summary["num_candidate_windows"] == 0:
        return False, ("no candidate windows exist (no consecutive multi-step "
                       "runs of remote-requiring MPI gates) — a window "
                       "executor would have nothing to fuse")
    if summary["num_feasible_windows"] == 0:
        return False, ("candidate windows exist but none are feasible "
                       "(RAM budget too small or non-batchable gates) — "
                       "executor not worth implementing for this workload/budget")
    if (tradeoffs["sendrecv_calls_saved"] <= 0
            and tradeoffs["bytes_saved"] <= 0):
        return False, ("feasible windows save no Sendrecv calls and no bytes — "
                       "no communication benefit to justify an executor")
    return True, (
        f"{summary['num_feasible_windows']} feasible window(s) would save "
        f"{tradeoffs['sendrecv_calls_saved']} Sendrecv call(s) and "
        f"{tradeoffs['bytes_saved']} MPI byte(s), avoiding "
        f"{tradeoffs['repeated_remote_fetches_avoided']} repeated remote "
        f"fetch(es); cost is up to {tradeoffs['extra_ram_gib_required']:.4f} "
        f"GiB extra RAM and increased post-crash recomputation "
        f"(expected +{tradeoffs['extra_recomputation_cost_after_crash']:.2f} "
        "gate-units). Worth implementing IF the recovery tradeoff is acceptable")


def report_to_candidates_json(report: dict) -> list[dict]:
    """The candidate list for ``mpi_window_candidates.json``."""
    return report["candidates"]


def report_to_summary_json(report: dict) -> dict:
    """The summary view for ``mpi_window_report.json`` (candidates omitted)."""
    return {k: v for k, v in report.items() if k != "candidates"}
