"""Calibrated cost-model telemetry: pure calibration + real-MPI timing fields.

Measurement-only feature: verifies the new timing fields are present and honest
(null-with-reason when a primitive was not exercised), that calibration
constants are derived from the measured run, and that enabling the telemetry
changes neither the final state nor the recovery invariants.
"""
import glob
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from wenbo_engine.planner.stage_cost_model import build_calibration
from wenbo_engine.planner.cost_report import build_timing_report, _TIMING_FIELDS

REPO = str(Path(__file__).resolve().parent.parent.parent)


# ── pure: calibration math + null-with-reason ───────────────────────────

def test_calibration_measured_values():
    cal = build_calibration({
        "bytes_read": 2_000_000_000, "read_sec": 1.0,
        "bytes_written": 1_000_000_000, "write_sec": 1.0,
        "mpi_bytes_sent": 3_000_000_000, "mpi_pairwise_sendrecv_time": 1.0,
        "gather_bytes": 4_000_000_000, "gather_time": 1.0,
        "scatter_bytes": 5_000_000_000, "scatter_time": 1.0,
        "local_gates_applied": 100, "local_kernel_time": 0.5,
        "nonlocal_gates_applied": 200, "nonlocal_kernel_time": 0.5,
        "commit_time": 0.2, "commit_count": 10,
        "state_bytes": 8_000_000_000, "norm_time": 1.0,
        "numba_speedup_factor": None,
    })
    assert cal["nvme_read_gbps"] == 2.0
    assert cal["nvme_write_gbps"] == 1.0
    assert cal["pairwise_mpi_gbps"] == 3.0
    assert cal["collective_gather_gbps"] == 4.0
    assert cal["collective_scatter_gbps"] == 5.0
    assert cal["local_kernel_gates_per_sec"] == 200.0
    assert cal["nonlocal_kernel_gates_per_sec"] == 400.0
    assert cal["commit_ms"] == 20.0
    assert cal["norm_scan_gbps"] == 8.0
    # every constant has a sibling reason key (null when measured)
    assert cal["nvme_read_gbps_reason"] is None


def test_calibration_unavailable_is_null_with_reason():
    cal = build_calibration({})   # nothing measured
    for k in ("nvme_read_gbps", "pairwise_mpi_gbps", "collective_gather_gbps",
              "local_kernel_gates_per_sec", "nonlocal_kernel_gates_per_sec",
              "commit_ms", "norm_scan_gbps", "numba_speedup_factor"):
        assert cal[k] is None, k
        assert isinstance(cal[k + "_reason"], str) and cal[k + "_reason"], k


def test_numba_speedup_needs_ab():
    cal = build_calibration({"numba_speedup_factor": None})
    assert cal["numba_speedup_factor"] is None
    assert "A/B" in cal["numba_speedup_factor_reason"]


def test_timing_report_predicted_null_actual_measured():
    rep = build_timing_report({"commit_time": 0.5, "norm_time": 0.1})
    assert rep["commit_time"]["actual"] == 0.5
    assert rep["commit_time"]["predicted"] is None
    assert rep["commit_time"]["predicted_reason"]          # explained
    # an absent measured field is null with its own reason (never omitted)
    assert rep["numba_compile_time"]["actual"] is None
    assert rep["numba_compile_time"]["actual_reason"]
    for f in _TIMING_FIELDS:
        assert f in rep


# ── real-MPI smokes ─────────────────────────────────────────────────────

pytest.importorskip("mpi4py")
_mark = pytest.mark.skipif(shutil.which("mpirun") is None, reason="no mpirun")

_FINAL_TIMING = (
    "local_kernel_time", "nonlocal_kernel_time", "mpi_pairwise_sendrecv_time",
    "mpi_collective_gather_time", "mpi_collective_scatter_time",
    "mpi_window_leader_compute_time", "mpi_window_segment_time",
    "direct_extent_read_time", "direct_extent_write_time",
    "extent_materialize_time", "extent_pack_time", "commit_time", "norm_time",
    "numba_compile_time", "rss_peak_gib", "overlay_peak_ram_gib",
    "remote_buffer_peak_gib",
)


def _bench(tmp, kind, extra, tag, n=12, depth=20, np_=4):
    out = Path(tmp) / f"o_{tag}"; wd = Path(tmp) / f"w_{tag}"
    shutil.rmtree(out, ignore_errors=True); shutil.rmtree(wd, ignore_errors=True)
    cmd = ["mpirun", "-np", str(np_), sys.executable, "-m",
           "wenbo_engine.bench.communication_workloads", "--kind", kind,
           "--n", str(n), "--depth", str(depth), "--recovery", "generation",
           "--mpi-exchange-mode", "gate_aware", "--ram-budget-gib", "21",
           "--verify", *extra, "--output-dir", str(out), "--work-dir", str(wd)]
    r = subprocess.run(cmd, env=dict(os.environ, PYTHONPATH=REPO),
                       capture_output=True, timeout=400)
    assert r.returncode == 0, r.stderr.decode()[-2000:]
    d = Path(glob.glob(str(out / "**" / "final_summary.json"),
                       recursive=True)[0]).parent
    return json.load(open(d / "final_summary.json")), d


@pytest.fixture(scope="module")
def window_run(tmp_path_factory):
    if shutil.which("mpirun") is None:
        pytest.skip("no mpirun")
    return _bench(tmp_path_factory.mktemp("cal"), "mpi_nonlocal_mixing_heavy",
                  ["--planner", "recovery_aware_v1",
                   "--mpi-window-execution", "safe"], "winsafe")


# 1: timing fields present in final_summary; 3: null → reason present
@_mark
def test_final_summary_has_timing_fields(window_run):
    fs, _ = window_run
    for f in _FINAL_TIMING:
        assert f in fs, f
        if fs[f] is None:
            assert fs.get(f + "_reason"), f   # explicit reason, never silent


# 2: timing block present in cost_report.json (recovery_aware run)
@_mark
def test_cost_report_has_timing_block(window_run):
    _, d = window_run
    cr = json.load(open(d / "cost_report.json"))
    assert "timing" in cr
    assert "commit_time" in cr["timing"]
    assert cr["timing"]["commit_time"]["predicted"] is None
    assert cr["timing"]["commit_time"]["predicted_reason"]


# 4: window run reports gather/scatter/leader timing
@_mark
def test_window_reports_collective_timing(window_run):
    fs, _ = window_run
    assert fs["mpi_windows_executed"] >= 1
    assert fs["mpi_collective_gather_time"] > 0
    assert fs["mpi_collective_scatter_time"] > 0
    assert fs["mpi_window_leader_compute_time"] > 0
    assert fs["mpi_window_segment_time"] > 0


# 7 & 8 & 9 & 10: commit_time, norm_time, recovery invariants, correctness
@_mark
def test_commit_norm_and_recovery_invariants(window_run):
    fs, d = window_run
    assert fs["commit_time"] > 0
    assert fs["norm_time"] > 0
    assert fs["correct"] is True
    assert abs(fs["final_norm"] - 1.0) < 1e-5
    assert fs["recovery_mode"] == "generation"
    rev = json.load(open(d / "recovery_events.json"))
    assert rev["source_of_truth"] == "global_commit_record"
    assert rev["wal_json_present"] is False


# 5: MPI-heavy gate_aware run reports nonlocal kernel time (pair kernels)
@_mark
def test_mpi_heavy_reports_nonlocal_kernel(tmp_path):
    fs, _ = _bench(tmp_path, "mpi_nonlocal_mixing_heavy",
                   ["--mpi-window-execution", "off"], "winoff")
    assert fs["nonlocal_kernel_time"] > 0          # pair kernels measured
    cal = json.load(open(glob.glob(str(tmp_path / "**" / "cost_model.json"),
                                   recursive=True)[0]))["constants"]
    assert cal["nonlocal_kernel_gates_per_sec"] is not None


# 6: direct extent path reports direct read/write timing
@_mark
def test_direct_extent_reports_io_timing(tmp_path):
    fs, _ = _bench(tmp_path, "communication_light",
                   ["--storage-layout", "extents", "--extent-io-mode", "direct",
                    "--execution-mode", "compute_unit"], "direct", depth=30)
    assert fs["correct"] is True
    assert fs["direct_extent_read_time"] is not None
    assert fs["direct_extent_write_time"] is not None
    assert fs["direct_extent_read_time"] >= 0 and fs["direct_extent_write_time"] > 0
