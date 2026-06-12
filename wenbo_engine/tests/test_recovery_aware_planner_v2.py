"""Recovery-aware adaptive planner v2: decisions (pure) + real-MPI invariants.

v2 ranks candidate strategies by predicted WALL TIME (calibrated), not bytes.
Pure tests assert the decisions; MPI smokes assert artifacts, invariants, and
honest prediction error.  Every MPI run uses a unique work_dir.
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
from wenbo_engine.planner.recovery_aware_planner_v2 import (
    plan_recovery_aware_v2, enumerate_candidates_v2,
)

REPO = str(Path(__file__).resolve().parent.parent.parent)
N, RANKS = 24, 4
CB = _default_chunk_bits(N, RANKS)


def _plan(kind, *, ram=21.0, calibration=None, n=N, depth=20, cb=CB):
    cd = build_circuit(kind, n, depth, cb, RANKS, 42)
    return plan_recovery_aware_v2(
        cd, n=n, chunk_bits=cb, num_ranks=RANKS, recovery="generation",
        ram_budget_gib=ram, auto_chunk_bits=True, calibration=calibration)


# ── 1: candidate generation ─────────────────────────────────────────────

def test_generates_candidates():
    cands = enumerate_candidates_v2()
    assert len(cands) >= 6
    names = [c.name for c in cands]
    assert any("window_safe" in n for n in names)
    assert any("window_off" in n for n in names)
    assert any(c.execution_mode == "compute_unit" and c.extent_io_mode == "direct"
               for c in cands)


# ── 2: RAM-infeasible candidates rejected ───────────────────────────────

def test_rejects_ram_infeasible():
    plan = _plan("mpi_nonlocal_mixing_heavy", ram=1e-6)
    # nothing fits a 1e-6 GiB budget → every candidate flagged infeasible
    assert all(not e["ram_feasible"] for e in plan["candidates"])
    # the decision still returns (penalised) and records infeasibility
    assert plan["decision"]["ram_feasible"] is False


# ── 3 & 4: communication_light → compute_unit+direct; tiny frags → step ──

def test_communication_light_selects_compute_unit_direct():
    d = _plan("communication_light")["decision"]
    assert d["selected_execution_mode"] == "compute_unit"
    assert d["selected_storage_layout"] == "extents"
    assert d["selected_extent_io_mode"] == "direct"
    assert d["selected_mpi_window_execution"] == "off"


def test_tiny_local_fragments_avoid_compute_unit():
    # mixing_heavy is all rank-bit MPI gates → no local runs → tiny units
    d = _plan("mpi_nonlocal_mixing_heavy")["decision"]
    assert d["selected_execution_mode"] == "step"


# ── 5 & 6: phase-heavy + default heavy keep window off ──────────────────

def test_phase_heavy_window_off():
    d = _plan("mpi_nonlocal_phase_heavy")["decision"]
    assert d["selected_mpi_window_execution"] == "off"


def test_default_mpi_heavy_window_off():
    d = _plan("mpi_nonlocal_heavy")["decision"]
    assert d["selected_mpi_window_execution"] == "off"


# ── 7 & 8: window chosen only when predicted faster ─────────────────────

def test_window_chosen_when_collectives_cheap():
    cheap = {"collective_gather_gbps": 50.0, "collective_scatter_gbps": 50.0,
             "leader_compute_amps_per_sec": 1e12, "segment_latency_ms": 0.0,
             "pairwise_mpi_gbps": 0.05, "commit_ms": 50.0}
    d = _plan("mpi_nonlocal_mixing_heavy", calibration=cheap)["decision"]
    assert d["selected_mpi_window_execution"] == "safe"


def test_window_rejected_when_slower_despite_fewer_bytes():
    d = _plan("mpi_nonlocal_mixing_heavy")["decision"]   # i3en-like defaults
    assert d["selected_mpi_window_execution"] == "off"
    # a window candidate exists and was rejected for wall time, not bytes
    rej = d["reason_for_each_rejection"]
    win_rej = [r for c, r in rej.items() if "window_safe" in c]
    assert win_rej and ("slower" in win_rej[0] or "wall time" in win_rej[0])


# ── 9 & 10: backend decision + rejection records ────────────────────────

def test_backend_decision_present():
    d = _plan("communication_light")["decision"]
    assert d["selected_kernel_backend"] in ("numpy", "numba")
    assert d["backend_decision_reason"]


def test_records_rejected_candidates_and_reasons():
    d = _plan("mpi_nonlocal_mixing_heavy")["decision"]
    assert len(d["rejected_candidates"]) >= 1
    assert set(d["rejected_candidates"]) <= set(d["reason_for_each_rejection"])
    assert all(d["reason_for_each_rejection"].values())


def test_predicted_wall_time_by_candidate_complete():
    plan = _plan("mpi_nonlocal_mixing_heavy")
    pw = plan["decision"]["predicted_wall_time_by_candidate"]
    assert len(pw) == len(plan["candidates"])
    assert all(isinstance(v, (int, float)) for v in pw.values())


# ── real-MPI smokes (11–18) ─────────────────────────────────────────────

pytest.importorskip("mpi4py")
_mark = pytest.mark.skipif(shutil.which("mpirun") is None, reason="no mpirun")


def _bench(tmp, kind, tag, planner="recovery_aware_v2", n=12, depth=20):
    out = Path(tmp) / f"o_{tag}"; wd = Path(tmp) / f"w_{tag}"
    shutil.rmtree(out, ignore_errors=True); shutil.rmtree(wd, ignore_errors=True)
    cmd = ["mpirun", "-np", "4", sys.executable, "-m",
           "wenbo_engine.bench.communication_workloads", "--kind", kind,
           "--n", str(n), "--depth", str(depth), "--recovery", "generation",
           "--kernel-backend", "auto", "--auto-chunk-bits", "--ram-budget-gib",
           "21", "--verify", "--output-dir", str(out), "--work-dir", str(wd)]
    if planner:
        cmd += ["--planner", planner]
    r = subprocess.run(cmd, env=dict(os.environ, PYTHONPATH=REPO),
                       capture_output=True, timeout=400)
    assert r.returncode == 0, r.stderr.decode()[-2000:]
    d = Path(glob.glob(str(out / "**" / "final_summary.json"),
                       recursive=True)[0]).parent
    return json.load(open(d / "final_summary.json")), d


@pytest.fixture(scope="module")
def light_run(tmp_path_factory):
    if shutil.which("mpirun") is None:
        pytest.skip("no mpirun")
    return _bench(tmp_path_factory.mktemp("v2"), "communication_light", "light")


# 11: all four artifacts emitted
@_mark
def test_emits_four_artifacts(light_run):
    _, d = light_run
    for f in ("plan_v2.json", "candidate_strategies_v2.json",
              "decision_report_v2.json", "cost_report_v2.json"):
        assert (Path(d) / f).exists(), f


# 12,13,14,15: correctness + recovery invariants
@_mark
def test_invariants(light_run):
    fs, d = light_run
    assert fs["correct"] is True
    assert abs(fs["final_norm"] - 1.0) < 1e-5
    assert fs["recovery_mode"] == "generation"
    rev = json.load(open(Path(d) / "recovery_events.json"))
    assert rev["source_of_truth"] == "global_commit_record"
    assert rev["wal_json_present"] is False


# 16: unique work_dir already enforced by _bench (distinct per tag); planner ran
@_mark
def test_planner_v2_selected_in_summary(light_run):
    fs, _ = light_run
    assert fs["planner"] == "recovery_aware_v2"
    assert fs["selected_execution_mode"] == "compute_unit"
    assert fs["selected_mpi_window_execution"] == "off"


# 17: defaults unchanged unless --planner recovery_aware_v2
@_mark
def test_defaults_unchanged_without_v2(tmp_path):
    fs, _ = _bench(tmp_path, "mpi_nonlocal_mixing_heavy", "noplanner",
                   planner=None)
    assert fs.get("planner") != "recovery_aware_v2"
    assert fs["correct"] is True


# 18: prediction error reported honestly when actual metrics available
@_mark
def test_prediction_error_honest(light_run):
    _, d = light_run
    cr = json.load(open(Path(d) / "cost_report_v2.json"))
    assert cr["actual_metrics_available"] is True
    m = cr["metrics"]
    # wall_time was measured → error is a number; unmeasured terms → null+reason
    assert m["wall_time"]["actual"] is not None
    assert isinstance(m["wall_time"]["prediction_error_pct"], (int, float))
    for v in m.values():
        if v["actual"] is None:
            assert v["actual_reason"] and v["prediction_error_pct"] is None
