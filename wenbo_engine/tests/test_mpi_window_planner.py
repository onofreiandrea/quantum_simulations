"""MPI-window feasibility planner + report (deterministic) + real-MPI smokes.

Analysis-only: these tests assert the planner predicts windows correctly and
that enabling the analysis NEVER changes runtime state or recovery invariants.
"""
import glob
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from wenbo_engine.bench.communication_workloads import (
    build_circuit, _default_chunk_bits,
)
from wenbo_engine.mpi import window_planner as wp
from wenbo_engine.planner.mpi_window_report import build_window_report

REPO = str(Path(__file__).resolve().parent.parent.parent)
N, DEPTH, RANKS = 24, 20, 4
CB = _default_chunk_bits(N, RANKS)


def _analyze(kind, ram=21.0):
    cd = build_circuit(kind, N, DEPTH, CB, RANKS, 42)
    return wp.analyze_windows(cd, CB, RANKS, ram_budget_gib=ram)


# ── 1: candidates generated for mixing_heavy ────────────────────────────

def test_candidates_for_mixing_heavy():
    r = _analyze("mpi_nonlocal_mixing_heavy")
    assert r["summary"]["num_candidate_windows"] >= 1
    c = r["candidates"][0]
    assert c["true_mixing_gates"] > 0
    assert c["gates_in_window"] == c["true_mixing_gates"] + c["permutation_gates"]


# ── 2: no candidates needed for phase_heavy (all diagonal, fast-pathed) ──

def test_no_candidates_for_phase_heavy():
    r = _analyze("mpi_nonlocal_phase_heavy")
    assert r["summary"]["num_candidate_windows"] == 0
    rep = build_window_report(
        build_circuit("mpi_nonlocal_phase_heavy", N, DEPTH, CB, RANKS, 42),
        CB, RANKS, ram_budget_gib=21.0)
    assert rep["executor_worth_implementing"] is False
    assert rep["recommendation"] == "do_not_implement_executor_yet"


# ── 3: repeated remote chunks across adjacent steps are counted ──────────

def test_repeated_remote_fetches_counted():
    r = _analyze("mpi_nonlocal_mixing_heavy")
    c = r["candidates"][0]
    assert c["mpi_steps_in_window"] >= 2
    assert c["remote_chunks"] > c["distinct_remote_chunks"]
    assert c["repeated_remote_fetches_avoided"] == (
        c["remote_chunks"] - c["distinct_remote_chunks"])
    assert c["repeated_remote_fetches_avoided"] > 0


# ── 6: estimated RAM is reported ─────────────────────────────────────────

def test_estimated_ram_reported():
    r = _analyze("mpi_nonlocal_mixing_heavy")
    c = r["candidates"][0]
    assert isinstance(c["estimated_ram_gib"], float)
    assert c["estimated_ram_gib"] > 0.0
    assert c["ram_budget_gib"] == 21.0


# ── 7: window rejected when RAM budget too small ─────────────────────────

def test_window_rejected_when_ram_too_small():
    r = _analyze("mpi_nonlocal_mixing_heavy", ram=1e-9)
    c = r["candidates"][0]
    assert c["ram_feasible"] is False
    assert c["safe_to_execute_future"] is False
    assert c["rejection_reason"] is not None
    assert "ram_budget_gib" in c["rejection_reason"]
    assert r["summary"]["num_feasible_windows"] == 0


# ── 8: window accepted as future-executable when RAM sufficient ──────────

def test_window_accepted_when_ram_sufficient():
    r = _analyze("mpi_nonlocal_mixing_heavy", ram=21.0)
    c = r["candidates"][0]
    assert c["ram_feasible"] is True
    assert c["safe_to_execute_future"] is True
    assert c["rejection_reason"] is None
    assert r["summary"]["num_feasible_windows"] >= 1


# ── 9: recomputation cost increases when commit count decreases ──────────

def test_recomputation_increases_when_commits_drop():
    r = _analyze("mpi_nonlocal_mixing_heavy")
    c = r["candidates"][0]
    assert c["commit_count_window"] < c["commit_count_baseline"]
    assert c["expected_recomputation_cost_increase"] > 0


# ── 10: report distinguishes bytes / sendrecv / commit / RAM / recovery ──

def test_report_separates_tradeoffs():
    rep = build_window_report(
        build_circuit("mpi_nonlocal_mixing_heavy", N, DEPTH, CB, RANKS, 42),
        CB, RANKS, ram_budget_gib=21.0)
    t = rep["tradeoffs"]
    for key in ("bytes_saved", "sendrecv_calls_saved", "commits_saved",
                "extra_ram_gib_required", "extra_recomputation_cost_after_crash",
                "repeated_remote_fetches_avoided"):
        assert key in t
    assert rep["recommendation"] in (
        "implement_executor", "do_not_implement_executor_yet")
    # recovery tradeoff is surfaced per-candidate, distinct from bytes/sendrecv
    assert "recovery_risk_note" in rep["best_candidate"]


def test_candidate_generation_deterministic():
    a = _analyze("mpi_nonlocal_mixing_heavy")
    b = _analyze("mpi_nonlocal_mixing_heavy")
    assert json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


# ── real-MPI smokes (4,5,11,12,13,14) ───────────────────────────────────

pytest.importorskip("mpi4py")
_mark = pytest.mark.skipif(shutil.which("mpirun") is None, reason="no mpirun")


def _bench(tmp, kind, analysis="report", n=12, depth=20):
    out = Path(tmp) / f"o_{kind}_{analysis}"
    wd = Path(tmp) / f"w_{kind}_{analysis}"
    for p in (out, wd):
        shutil.rmtree(p, ignore_errors=True)
    cmd = ["mpirun", "-np", "2", sys.executable, "-m",
           "wenbo_engine.bench.communication_workloads", "--kind", kind,
           "--n", str(n), "--depth", str(depth), "--recovery", "generation",
           "--planner", "recovery_aware_v1", "--mpi-exchange-mode", "gate_aware",
           "--mpi-window-analysis", analysis, "--ram-budget-gib", "21",
           "--verify", "--output-dir", str(out), "--work-dir", str(wd)]
    r = subprocess.run(cmd, env=dict(os.environ, PYTHONPATH=REPO),
                       capture_output=True, timeout=400)
    assert r.returncode == 0, r.stderr.decode()[-2000:]
    d = Path(glob.glob(str(out / "**" / "final_summary.json"),
                       recursive=True)[0]).parent
    return d


@pytest.fixture(scope="module")
def mixing_dir(tmp_path_factory):
    if shutil.which("mpirun") is None:
        pytest.skip("no mpirun")
    return _bench(tmp_path_factory.mktemp("win"), "mpi_nonlocal_mixing_heavy")


def _fs(d):
    return json.load(open(Path(d) / "final_summary.json"))


# 4 & 5: analytic baseline matches measured telemetry (cluster)
@_mark
def test_analytic_baseline_matches_telemetry(mixing_dir):
    fs = _fs(mixing_dir)
    rep = json.load(open(Path(mixing_dir) / "mpi_window_report.json"))
    ba = rep["baseline_analytic"]
    # full-run analytic baseline (all remote-requiring MPI steps) == telemetry
    assert ba["sendrecv_cluster"] == fs["sendrecv_count"]      # 4
    assert ba["mpi_bytes_cluster"] == fs["mpi_bytes_sent"]     # 5
    assert rep["baseline_measured"]["sendrecv_count"] == fs["sendrecv_count"]
    assert rep["baseline_measured"]["mpi_bytes_sent"] == fs["mpi_bytes_sent"]


# 11: runtime final state unchanged when analysis is enabled
@_mark
def test_final_state_unchanged_with_analysis(tmp_path):
    on = _fs(_bench(tmp_path, "mpi_nonlocal_mixing_heavy", "report"))
    off = _fs(_bench(tmp_path, "mpi_nonlocal_mixing_heavy", "off"))
    assert on["correct"] is True and off["correct"] is True
    assert abs(on["final_norm"] - off["final_norm"]) < 1e-9
    assert abs(on["final_norm"] - 1.0) < 1e-5
    # runtime communication identical with analysis on vs off
    assert on["sendrecv_count"] == off["sendrecv_count"]
    assert on["mpi_bytes_sent"] == off["mpi_bytes_sent"]


# 12: recovery invariants remain unchanged
@_mark
def test_recovery_invariants_unchanged(mixing_dir):
    fs = _fs(mixing_dir)
    assert fs["recovery_mode"] == "generation"
    rev = json.load(open(Path(mixing_dir) / "recovery_events.json"))
    assert rev["source_of_truth"] == "global_commit_record"
    assert rev["wal_json_present"] is False


# 13: final_summary includes mpi_window_analysis fields
@_mark
def test_final_summary_has_window_fields(mixing_dir):
    fs = _fs(mixing_dir)
    assert fs["mpi_window_analysis"] == "report"
    for k in ("mpi_window_num_candidates", "mpi_window_num_feasible",
              "mpi_window_bytes_saved", "mpi_window_sendrecv_saved",
              "mpi_window_commits_saved", "mpi_window_extra_ram_gib",
              "mpi_window_extra_recomputation_cost",
              "mpi_window_executor_worth_implementing",
              "mpi_window_recommendation"):
        assert k in fs, k


# 14: mpi_window_report.json + candidates.json are produced
@_mark
def test_window_report_json_produced(mixing_dir):
    assert (Path(mixing_dir) / "mpi_window_report.json").exists()
    assert (Path(mixing_dir) / "mpi_window_candidates.json").exists()
    rep = json.load(open(Path(mixing_dir) / "mpi_window_report.json"))
    assert rep["analysis_mode"] == "report"
    assert rep["runtime_execution_changed"] is False
    assert rep["remote_cache_scope"] == "step"
    cands = json.load(open(Path(mixing_dir) / "mpi_window_candidates.json"))
    assert isinstance(cands, list) and len(cands) >= 1
