"""Recovery-aware hierarchical planner v1.

In-process tests cover candidate generation, the cost-term completeness, and
the deterministic selection rules.  Real-MPI smokes prove the planner path runs
end-to-end with generation recovery, produces the three artifact files, records
predicted-vs-actual metrics, and keeps every recovery invariant intact while
exercising extents + direct extent I/O + compute units + gate-aware MPI.
"""
import glob
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from wenbo_engine.bench.communication_workloads import (
    build_circuit, _default_chunk_bits,
)
from wenbo_engine.planner import (
    plan_recovery_aware, enumerate_candidates, estimate_candidate,
    build_plan_context, build_cost_report, selected_run_params,
)

REPO = str(Path(__file__).resolve().parent.parent.parent)

# Required estimate terms every candidate must carry.
_REQUIRED_TERMS = (
    "bytes_read", "bytes_written", "read_ops", "write_ops",
    "temporary_chunk_files_created", "mpi_bytes_sent", "sendrecv_count",
    "kernel_time", "commit_count", "commit_cost", "durable_checkpoint_cost",
    "expected_recomputation_cost", "layout_materialization_cost",
    "estimated_total_cost",
)


def _plan(kind, n=24, depth=20, ranks=4, recovery="generation", min_gates=4):
    cb = _default_chunk_bits(n, ranks)
    cd = build_circuit(kind, n, depth, cb, ranks, 42)
    return plan_recovery_aware(cd, n=n, chunk_bits=cb, num_ranks=ranks,
                               recovery=recovery, compute_unit_min_gates=min_gates)


# ── 1. candidate strategies are generated ───────────────────────────────

def test_candidates_generated():
    cands = enumerate_candidates()
    assert len(cands) == 4
    names = [c.name for c in cands]
    assert names == [
        "chunks+step+naive",
        "extents+step+gate_aware",
        "extents+compute_unit+materialize+gate_aware",
        "extents+compute_unit+direct+gate_aware",
    ]


# ── 2. candidate costs contain all required terms ───────────────────────

def test_candidate_costs_have_all_terms():
    plan = _plan("communication_light")
    assert len(plan["candidates"]) == 4
    for c in plan["candidates"]:
        for term in _REQUIRED_TERMS:
            assert term in c, (c["candidate"], term)
        # the full named cost breakdown is present too
        for term in ("nvme_read_write_cost", "mpi_exchange_cost", "kernel_cost",
                     "extent_materialization_cost", "direct_extent_io_cost",
                     "layout_materialization_cost", "commit_cost",
                     "durable_checkpoint_cost", "expected_recomputation_cost",
                     "estimated_total_cost"):
            assert term in c["cost_terms"], (c["candidate"], term)


# ── 3. selected strategy is deterministic ───────────────────────────────

def test_selection_is_deterministic():
    a = _plan("mixed_staged")
    b = _plan("mixed_staged")
    assert (a["decision"]["selected_strategy"]
            == b["decision"]["selected_strategy"])
    assert (a["decision"]["predicted_cost_by_candidate"]
            == b["decision"]["predicted_cost_by_candidate"])


# ── 4. compute_unit selected for long local runs ────────────────────────

def test_compute_unit_for_long_local_run():
    plan = _plan("communication_light")           # all-local, long local run
    assert plan["context_summary"]["has_fused_local_unit"] is True
    assert plan["decision"]["selected_strategy"]["execution_mode"] == "compute_unit"


# ── 5. compute_unit rejected for short local fragments ──────────────────

def test_compute_unit_rejected_for_short_local():
    # raise the threshold above the available local run → no fused unit
    plan = _plan("communication_light", min_gates=10_000)
    assert plan["context_summary"]["has_fused_local_unit"] is False
    assert plan["decision"]["selected_strategy"]["execution_mode"] == "step"
    # and at least one compute_unit candidate is rejected citing rule 1
    reasons = " ".join(r["reasons"][0] for r in
                       plan["decision"]["rejected_candidates"])
    assert "rule 1" in reasons


# ── 6. gate_aware selected for MPI-heavy workloads ──────────────────────

def test_gate_aware_for_mpi_heavy():
    plan = _plan("mpi_nonlocal_heavy")
    assert plan["context_summary"]["total_mpi_nonlocal_ops"] > 0
    assert plan["decision"]["selected_strategy"]["mpi_exchange_mode"] == "gate_aware"


# ── 7. direct extent I/O selected for local compute-unit + extents ──────

def test_direct_extent_io_for_local_compute_unit():
    plan = _plan("communication_light")
    sel = plan["decision"]["selected_strategy"]
    assert sel["storage_layout"] == "extents"
    assert sel["execution_mode"] == "compute_unit"
    assert sel["extent_io_mode"] == "direct"


# ── 8. layout materialization cost is represented SEPARATELY ─────────────

def test_layout_materialization_cost_separate():
    plan = _plan("communication_light")
    for c in plan["candidates"]:
        ct = c["cost_terms"]
        # three distinct, non-aliased terms exist
        assert "layout_materialization_cost" in ct
        assert "extent_materialization_cost" in ct
        assert "direct_extent_io_cost" in ct
    # the materialize candidate pays extent materialization; the direct one
    # pays direct-extent I/O instead (mutually exclusive contributions)
    by = {c["candidate"]: c["cost_terms"] for c in plan["candidates"]}
    mat = by["extents+compute_unit+materialize+gate_aware"]
    direct = by["extents+compute_unit+direct+gate_aware"]
    assert mat["extent_materialization_cost"] > 0
    assert direct["extent_materialization_cost"] == 0
    assert direct["direct_extent_io_cost"] > 0


# ── 9. commit cost is represented ───────────────────────────────────────

def test_commit_cost_represented():
    plan = _plan("communication_light")
    for c in plan["candidates"]:
        assert c["commit_cost"] > 0
        assert c["commit_count"] >= 1


# ── 10. expected recomputation cost is represented ──────────────────────

def test_expected_recomputation_cost_represented():
    plan = _plan("communication_light")
    for c in plan["candidates"]:
        assert "expected_recomputation_cost" in c
        assert c["expected_recomputation_cost"] > 0    # planner_failure_prob > 0


# ── selection rule-6 explanation present when a cheaper candidate rejected ─

def test_reason_explains_rejected_cheaper_candidate():
    plan = _plan("mpi_nonlocal_heavy")   # naive (cheaper) rejected for gate_aware
    reason = plan["decision"]["reason_for_selection"]
    assert "rule 3" in reason or "gate_aware" in reason


# ── cost report (in-process): predicted + actual + error ────────────────

def test_cost_report_records_predicted_and_actual():
    plan = _plan("communication_light")
    # synthesize an "actual" measurement and build the report
    actual = {"bytes_read": 100, "bytes_written": 100, "read_ops": 0,
              "write_ops": 0, "temporary_chunk_files_created": 0,
              "mpi_bytes_sent": 0, "sendrecv_count": 0, "kernel_time": 0.01,
              "commit_count": 2, "wall_time": 1.0, "work_time": 0.5}
    rep = build_cost_report(plan, actual)
    for key in ("read_ops", "mpi_bytes_sent", "commit_count"):
        m = rep["metrics"][key]
        assert "predicted" in m and "actual" in m
        assert m["prediction_error_pct"] is not None   # 15. error computed


# ── real-MPI smokes (cover artifacts + invariants + correctness) ────────

pytest.importorskip("mpi4py")
_mark = pytest.mark.skipif(shutil.which("mpirun") is None, reason="no mpirun")


def _run_bench(tmp, kind, n, depth, ranks=2):
    out = Path(tmp) / f"out_{kind}"
    wd = Path(tmp) / f"wd_{kind}"
    for p in (out, wd):
        shutil.rmtree(p, ignore_errors=True)
    cmd = ["mpirun", "-np", str(ranks), sys.executable, "-m",
           "wenbo_engine.bench.communication_workloads", "--kind", kind,
           "--n", str(n), "--depth", str(depth), "--recovery", "generation",
           "--planner", "recovery_aware_v1", "--verify",
           "--output-dir", str(out), "--work-dir", str(wd)]
    r = subprocess.run(cmd, env=dict(os.environ, PYTHONPATH=REPO),
                       capture_output=True, timeout=400)
    assert r.returncode == 0, r.stderr.decode()[-2000:]
    def load(name):
        return json.load(open(glob.glob(str(out / "**" / name),
                                        recursive=True)[0]))
    return out, load


@pytest.fixture(scope="module")
def light_run(tmp_path_factory):
    if shutil.which("mpirun") is None:
        pytest.skip("no mpirun")
    tmp = tmp_path_factory.mktemp("rap_light")
    out, load = _run_bench(tmp, "communication_light", 14, 16)
    return out, load


@pytest.fixture(scope="module")
def mpi_run(tmp_path_factory):
    if shutil.which("mpirun") is None:
        pytest.skip("no mpirun")
    tmp = tmp_path_factory.mktemp("rap_mpi")
    out, load = _run_bench(tmp, "mpi_nonlocal_heavy", 8, 12)
    return out, load


# 11/12/13. the three artifacts are produced
@_mark
def test_artifacts_produced(light_run):
    out, _ = light_run
    for name in ("plan.json", "candidate_strategies.json", "cost_report.json"):
        assert glob.glob(str(out / "**" / name), recursive=True), name


# 16. final state matches reference for small circuits
# 17. works with generation recovery
@_mark
def test_correct_and_generation_recovery(light_run):
    _, load = light_run
    fs = load("final_summary.json")
    assert fs["correct"] is True
    assert fs["recovery_mode"] == "generation"
    assert abs(fs["final_norm"] - 1.0) < 1e-5
    assert fs["planner"] == "recovery_aware_v1"


# 18/19. recovery invariants intact
@_mark
def test_recovery_invariants(light_run):
    out, load = light_run
    rev = load("recovery_events.json")
    assert rev["source_of_truth"] == "global_commit_record"
    assert rev["wal_json_present"] is False
    assert not glob.glob(str(out / "**" / "wal.json"), recursive=True)


# 14/15. predicted and actual metrics both recorded; error computed
@_mark
def test_predicted_vs_actual_recorded(light_run):
    _, load = light_run
    cr = load("cost_report.json")
    assert cr["selected_strategy"]
    for key in ("read_ops", "write_ops", "mpi_bytes_sent", "sendrecv_count",
                "kernel_time", "commit_count"):
        m = cr["metrics"][key]
        assert m["predicted"] is not None and m["actual"] is not None
        assert m["prediction_error_pct"] is not None


# 20. works with extents + direct extent I/O + compute units (local workload)
@_mark
def test_light_selects_extents_direct_compute_unit(light_run):
    _, load = light_run
    fs = load("final_summary.json")
    assert fs["selected_strategy"] == "extents+compute_unit+direct+gate_aware"
    assert fs["storage_layout"] == "extents"
    assert fs["execution_mode"] == "compute_unit"
    assert fs["extent_io_mode"] == "direct"
    assert fs["temporary_chunk_files_created"] == 0   # direct → no temp files


# 20 (cont). works with gate-aware MPI + preserves MPI stress
@_mark
def test_mpi_heavy_selects_gate_aware_and_preserves_stress(mpi_run):
    _, load = mpi_run
    fs = load("final_summary.json")
    assert fs["correct"] is True
    assert fs["recovery_mode"] == "generation"
    assert fs["measured_mpi_nonlocal_ops"] > 0        # MPI stress preserved
    assert fs["mpi_bytes_sent"] > 0
    sel = fs["selected_strategy"]
    assert "gate_aware" in sel
    rev = load("recovery_events.json")
    assert rev["source_of_truth"] == "global_commit_record"
    assert rev["wal_json_present"] is False
