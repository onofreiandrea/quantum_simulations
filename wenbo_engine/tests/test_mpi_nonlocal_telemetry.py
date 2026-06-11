"""MPI-nonlocal telemetry + diagonal fast path (real-MPI) + workload validity."""
import glob
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from wenbo_engine.bench.communication_workloads import (
    build_circuit, circuit_clifford_stats, classify_mpi_gates,
)

REPO = str(Path(__file__).resolve().parent.parent.parent)


# ── 11,12: new workloads are non-stabilizer ─────────────────────────────

def test_phase_heavy_non_stabilizer():
    cd = build_circuit("mpi_nonlocal_phase_heavy", 24, 20, 20, 4, 42)
    assert circuit_clifford_stats(cd)["is_stabilizer"] is False


def test_mixing_heavy_non_stabilizer():
    cd = build_circuit("mpi_nonlocal_mixing_heavy", 24, 20, 20, 4, 42)
    assert circuit_clifford_stats(cd)["is_stabilizer"] is False


def test_phase_heavy_all_diagonal_mixing_heavy_all_mixing():
    ph = classify_mpi_gates(build_circuit("mpi_nonlocal_phase_heavy", 24, 20, 20, 4, 42), 20, 4)
    mx = classify_mpi_gates(build_circuit("mpi_nonlocal_mixing_heavy", 24, 20, 20, 4, 42), 20, 4)
    assert ph["diagonal_mpi_nonlocal_gate_count"] > 0
    assert ph["true_mixing_mpi_nonlocal_gate_count"] == 0
    assert ph["skipped_mpi_exchange_gate_count"] == ph["diagonal_mpi_nonlocal_gate_count"]
    assert mx["true_mixing_mpi_nonlocal_gate_count"] > 0
    assert mx["diagonal_mpi_nonlocal_gate_count"] == 0
    assert mx["requires_remote_amplitudes_gate_count"] > 0


# ── real-MPI smokes ─────────────────────────────────────────────────────

pytest.importorskip("mpi4py")
_mark = pytest.mark.skipif(shutil.which("mpirun") is None, reason="no mpirun")


def _bench(tmp, kind, n=12, depth=20):
    out = Path(tmp) / f"o_{kind}"; wd = Path(tmp) / f"w_{kind}"
    for p in (out, wd):
        shutil.rmtree(p, ignore_errors=True)
    cmd = ["mpirun", "-np", "2", sys.executable, "-m",
           "wenbo_engine.bench.communication_workloads", "--kind", kind,
           "--n", str(n), "--depth", str(depth), "--recovery", "generation",
           "--planner", "recovery_aware_v1", "--mpi-exchange-mode", "gate_aware",
           "--verify", "--output-dir", str(out), "--work-dir", str(wd)]
    r = subprocess.run(cmd, env=dict(os.environ, PYTHONPATH=REPO),
                       capture_output=True, timeout=400)
    assert r.returncode == 0, r.stderr.decode()[-2000:]
    return json.load(open(glob.glob(str(out / "**" / "final_summary.json"),
                                    recursive=True)[0]))


@pytest.fixture(scope="module")
def runs(tmp_path_factory):
    if shutil.which("mpirun") is None:
        pytest.skip("no mpirun")
    t = tmp_path_factory.mktemp("mpitel")
    return {
        "phase": _bench(t, "mpi_nonlocal_phase_heavy"),
        "mixing": _bench(t, "mpi_nonlocal_mixing_heavy"),
    }


# 1: remote-cache telemetry surfaced; 2: scope=step
@_mark
def test_cache_telemetry_surfaced(runs):
    fs = runs["mixing"]
    for k in ("remote_buffer_cache_hits", "remote_buffer_cache_misses",
              "remote_buffer_cache_hit_rate", "distinct_remote_chunks_per_rank",
              "repeated_remote_chunk_fetches",
              "repeated_remote_chunk_fetches_adjacent_steps", "mpi_steps",
              "mpi_gates_per_step", "remote_cache_scope"):
        assert k in fs, k
    assert fs["remote_cache_scope"] == "step"


# 3: repeated cross-step remote chunks are counted (true-mixing has redundancy)
@_mark
def test_repeated_adjacent_fetches_counted(runs):
    fs = runs["mixing"]
    assert fs["distinct_remote_chunks_per_rank"] > 0
    assert fs["repeated_remote_chunk_fetches_adjacent_steps"] > 0


# 9: diagonal fast path reduces MPI for phase-heavy
@_mark
def test_phase_heavy_skips_mpi(runs):
    fs = runs["phase"]
    assert fs["diagonal_mpi_nonlocal_gate_count"] > 0
    assert fs["skipped_mpi_exchange_gate_count"] == fs["diagonal_mpi_nonlocal_gate_count"]
    assert fs["sendrecv_count"] == 0          # nothing exchanged
    assert fs["mpi_bytes_sent"] == 0
    assert fs["correct"] is True
    assert abs(fs["final_norm"] - 1.0) < 1e-5


# 10: true-mixing still sends MPI data
@_mark
def test_mixing_heavy_still_exchanges(runs):
    fs = runs["mixing"]
    assert fs["true_mixing_mpi_nonlocal_gate_count"] > 0
    assert fs["sendrecv_count"] > 0
    assert fs["mpi_bytes_sent"] > 0
    assert fs["correct"] is True
    assert abs(fs["final_norm"] - 1.0) < 1e-5


# 13,14,15,16: recovery invariants + correctness, both workloads
@_mark
def test_recovery_invariants_both(runs, tmp_path):
    for fs in (runs["phase"], runs["mixing"]):
        assert fs["recovery_mode"] == "generation"
        assert abs(fs["final_norm"] - 1.0) < 1e-5
    # recovery_events for one run
    out = glob.glob("/tmp/**/recovery_events.json", recursive=True)  # not used
    # re-read via a fresh small run to check source_of_truth/wal
    fs2 = _bench(tmp_path, "mpi_nonlocal_phase_heavy", n=10, depth=8)
    rev = json.load(open(glob.glob(str(Path(tmp_path) / "**" / "recovery_events.json"),
                                   recursive=True)[0]))
    assert rev["source_of_truth"] == "global_commit_record"
    assert rev["wal_json_present"] is False
