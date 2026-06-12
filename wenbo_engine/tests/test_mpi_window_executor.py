"""MPI true-mixing window executor: selection (pure) + real-MPI behaviour.

Every MPI test uses a UNIQUE work_dir (per kind/mode/run) so a stale committed
generation can never produce a false-correct result.  The executor is off by
default and must not change behaviour when --mpi-window-execution=off.
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
from wenbo_engine.circuit.io import validate_circuit_dict, levelize
from wenbo_engine.mpi.mpi_runner import _compile_steps
from wenbo_engine.mpi.window_executor import (
    plan_executable_windows, apply_window_to_group_buffer, _rank_pattern,
)

REPO = str(Path(__file__).resolve().parent.parent.parent)
N, DEPTH, RANKS = 24, 20, 4
CB = _default_chunk_bits(N, RANKS)
NLB = N - CB - 2  # p = log2(4) = 2


def _steps(kind, n=N, depth=DEPTH, cb=CB, ranks=RANKS):
    cd = validate_circuit_dict(build_circuit(kind, n, depth, cb, ranks, 42))
    return _compile_steps(levelize(cd), cb, n - cb - (ranks.bit_length() - 1))


# ── pure selection logic (no MPI) ───────────────────────────────────────

def test_mixing_window_selected():
    wins, rejs = plan_executable_windows(_steps("mpi_nonlocal_mixing_heavy"),
                                         CB, NLB, RANKS, 1 << CB, 21.0)
    assert len(wins) == 1
    w = wins[0]
    assert w.n_steps >= 2 and w.n_gates > 0
    assert w.group_size == 4
    assert w.estimated_ram_gib > 0


def test_phase_heavy_no_windows():
    wins, rejs = plan_executable_windows(_steps("mpi_nonlocal_phase_heavy"),
                                         CB, NLB, RANKS, 1 << CB, 21.0)
    assert wins == []                       # all diagonal → nothing to fuse


def test_default_mpi_heavy_no_windows():
    wins, rejs = plan_executable_windows(_steps("mpi_nonlocal_heavy"),
                                         CB, NLB, RANKS, 1 << CB, 21.0)
    assert wins == []                       # permutation/mixed → not pure-mixing


def test_small_ram_budget_rejects():
    wins, rejs = plan_executable_windows(_steps("mpi_nonlocal_mixing_heavy"),
                                         CB, NLB, RANKS, 1 << CB, 1e-9)
    assert wins == []
    assert rejs and "ram_budget_gib" in rejs[0][2]


def test_window_gate_math_matches_dense():
    # n=5: k=1, n_local_bits=2, p=2 → rank qubits 3,4; verify fused window math.
    from wenbo_engine.kernel import gates as g
    from wenbo_engine.kernel.ref_dense import simulate
    n, k, nlb, ranks = 5, 1, 2, 4
    cs = 1 << k
    base_gates = [{"qubits": [q], "gate": "H"} for q in range(n)]
    win_gates = [
        {"qubits": [3], "gate": "RX", "params": {"theta": 0.5}},
        {"qubits": [4], "gate": "RY", "params": {"theta": 0.9}},
        {"qubits": [3], "gate": "H"},
        {"qubits": [4], "gate": "RX", "params": {"theta": 0.3}},
    ]
    ref = simulate({"number_of_qubits": n,
                    "gates": base_gates + win_gates}).astype(np.complex64)
    state = simulate({"number_of_qubits": n,
                      "gates": base_gates}).astype(np.complex64)
    wg = [((gt["qubits"][0] - k - nlb),
           g.gate_matrix(gt["gate"], gt.get("params", {})).astype(np.complex64))
          for gt in win_gates]
    sorted_bits = [0, 1]
    ncr = (1 << (n - k)) // ranks
    new = state.copy()
    for ci in range(ncr):
        by_pat = np.empty((ranks, cs), dtype=np.complex64)
        for r in range(ranks):
            base = (r << (k + nlb)) | (ci << k)
            by_pat[_rank_pattern(r, sorted_bits)] = state[base:base + cs]
        out = apply_window_to_group_buffer(by_pat, sorted_bits, wg)
        for r in range(ranks):
            base = (r << (k + nlb)) | (ci << k)
            new[base:base + cs] = out[_rank_pattern(r, sorted_bits)]
    assert np.allclose(new, ref, atol=1e-5)


# ── real-MPI behaviour ──────────────────────────────────────────────────

pytest.importorskip("mpi4py")
_mark = pytest.mark.skipif(shutil.which("mpirun") is None, reason="no mpirun")


def _run(tmp, kind, mode, n=12, depth=20, env_extra=None, verify=True,
         expect_ok=True):
    """Run one workload with a UNIQUE work_dir; return (final_summary, rc)."""
    tag = f"{kind}_{mode}_{n}_{abs(hash((kind, mode, n, str(env_extra)))) % 99999}"
    out = Path(tmp) / f"o_{tag}"
    wd = Path(tmp) / f"w_{tag}"          # unique work_dir per run (rule 15/16)
    if env_extra is None:               # only wipe on a fresh (non-resume) run
        shutil.rmtree(out, ignore_errors=True)
        shutil.rmtree(wd, ignore_errors=True)
    cmd = ["mpirun", "-np", "2", sys.executable, "-m",
           "wenbo_engine.bench.communication_workloads", "--kind", kind,
           "--n", str(n), "--depth", str(depth), "--recovery", "generation",
           "--planner", "recovery_aware_v1", "--mpi-exchange-mode", "gate_aware",
           "--mpi-window-execution", mode, "--ram-budget-gib", "21",
           "--output-dir", str(out), "--work-dir", str(wd)]
    if verify:
        cmd.append("--verify")
    env = dict(os.environ, PYTHONPATH=REPO)
    if env_extra:
        env.update(env_extra)
    r = subprocess.run(cmd, env=env, capture_output=True, timeout=400)
    if expect_ok:
        assert r.returncode == 0, r.stderr.decode()[-2000:]
    fs_files = glob.glob(str(out / "**" / "final_summary.json"), recursive=True)
    fs = json.load(open(fs_files[0])) if fs_files else None
    return fs, r.returncode, out, wd


@pytest.fixture(scope="module")
def tmp(tmp_path_factory):
    return tmp_path_factory.mktemp("winexec")


# 1 & 17: off by default / off does not run the executor
@_mark
def test_off_by_default(tmp):
    # no --mpi-window-execution flag at all → defaults to off
    out = Path(tmp) / "o_default"; wd = Path(tmp) / "w_default"
    shutil.rmtree(out, ignore_errors=True); shutil.rmtree(wd, ignore_errors=True)
    cmd = ["mpirun", "-np", "2", sys.executable, "-m",
           "wenbo_engine.bench.communication_workloads", "--kind",
           "mpi_nonlocal_mixing_heavy", "--n", "12", "--depth", "20",
           "--recovery", "generation", "--planner", "recovery_aware_v1",
           "--mpi-exchange-mode", "gate_aware", "--verify",
           "--output-dir", str(out), "--work-dir", str(wd)]
    r = subprocess.run(cmd, env=dict(os.environ, PYTHONPATH=REPO),
                       capture_output=True, timeout=400)
    assert r.returncode == 0, r.stderr.decode()[-2000:]
    fs = json.load(open(glob.glob(str(out / "**" / "final_summary.json"),
                                  recursive=True)[0]))
    assert fs["mpi_window_execution_mode"] == "off"
    assert fs["mpi_windows_executed"] == 0
    assert fs["sendrecv_count"] > 0          # per-step exchange still happens


@_mark
def test_off_explicit_no_executor(tmp):
    fs, _, _, _ = _run(tmp, "mpi_nonlocal_mixing_heavy", "off")
    assert fs["mpi_window_execution_mode"] == "off"
    assert fs["mpi_windows_executed"] == 0


# 2: safe mode executes a true-mixing window
@_mark
def test_safe_executes_window(tmp):
    fs, _, _, _ = _run(tmp, "mpi_nonlocal_mixing_heavy", "safe")
    assert fs["mpi_window_execution_mode"] == "safe"
    assert fs["mpi_windows_executed"] >= 1
    assert fs["mpi_window_gates_executed"] > 0
    assert fs["mpi_window_steps_executed"] >= 2


# 3: phase-heavy executes zero windows
@_mark
def test_phase_heavy_zero_windows(tmp):
    fs, _, _, _ = _run(tmp, "mpi_nonlocal_phase_heavy", "safe")
    assert fs["mpi_windows_executed"] == 0
    assert fs["correct"] is True


# 4: default mpi_nonlocal_heavy executes zero windows (falls back)
@_mark
def test_default_heavy_zero_windows(tmp):
    fs, _, _, _ = _run(tmp, "mpi_nonlocal_heavy", "safe")
    assert fs["mpi_windows_executed"] == 0
    assert fs["correct"] is True


# 6 & 9 & 11: tight RAM budget is respected — the leader gathers in segments so
# the window stays within budget (no OOM) rather than blowing up; result correct.
@_mark
def test_tight_ram_budget_respected(tmp):
    out = Path(tmp) / "o_smallram"; wd = Path(tmp) / "w_smallram"
    shutil.rmtree(out, ignore_errors=True); shutil.rmtree(wd, ignore_errors=True)
    # A tight (but runnable) budget: the window must segment its gather to fit,
    # never exceeding the budget, and still produce the correct state.
    budget = 0.02
    cmd = ["mpirun", "-np", "4", sys.executable, "-m",
           "wenbo_engine.bench.communication_workloads", "--kind",
           "mpi_nonlocal_mixing_heavy", "--n", "12", "--depth", "20",
           "--recovery", "generation", "--planner", "recovery_aware_v1",
           "--mpi-exchange-mode", "gate_aware", "--mpi-window-execution", "safe",
           "--chunk-bits", "8", "--ram-budget-gib", str(budget), "--verify",
           "--output-dir", str(out), "--work-dir", str(wd)]
    r = subprocess.run(cmd, env=dict(os.environ, PYTHONPATH=REPO),
                       capture_output=True, timeout=400)
    assert r.returncode == 0, r.stderr.decode()[-2000:]   # no OOM / crash
    fs = json.load(open(glob.glob(str(out / "**" / "final_summary.json"),
                                  recursive=True)[0]))
    assert fs["correct"] is True
    # RAM budget respected: estimated leader peak <= budget; ran or fell back.
    assert fs["mpi_window_estimated_ram_gib"] <= budget
    assert (fs["mpi_windows_executed"] >= 1) or (fs["mpi_window_fallbacks"] >= 1)


# 7 & 8 & 9: window output == gate-aware step output == dense ref; norm ~ 1
@_mark
def test_window_matches_baseline_and_dense(tmp):
    off, _, _, _ = _run(tmp, "mpi_nonlocal_mixing_heavy", "off", n=10)
    safe, _, _, _ = _run(tmp, "mpi_nonlocal_mixing_heavy", "safe", n=10)
    # both verified bit-close to the dense reference (atol 1e-5) — this is the
    # real "window output == gate-aware step output == dense" equivalence (7,8).
    assert off["correct"] is True and safe["correct"] is True
    assert abs(off["final_norm"] - safe["final_norm"]) < 1e-5   # match (7)
    assert abs(safe["final_norm"] - 1.0) < 1e-5                 # norm ~ 1 (9)


# 4 (acceptance): window reduces sendrecv / MPI bytes vs baseline
@_mark
def test_window_reduces_communication(tmp):
    off, _, _, _ = _run(tmp, "mpi_nonlocal_mixing_heavy", "off")
    safe, _, _, _ = _run(tmp, "mpi_nonlocal_mixing_heavy", "safe")
    assert (safe["sendrecv_count"] < off["sendrecv_count"]
            or safe["mpi_bytes_sent"] < off["mpi_bytes_sent"])


# 10,11,12: recovery invariants unchanged
@_mark
def test_recovery_invariants(tmp):
    fs, _, out, _ = _run(tmp, "mpi_nonlocal_mixing_heavy", "safe")
    assert fs["recovery_mode"] == "generation"
    rev_files = glob.glob(str(Path(out) / "**" / "recovery_events.json"),
                          recursive=True)
    rev = json.load(open(rev_files[0]))
    assert rev["source_of_truth"] == "global_commit_record"
    assert rev["wal_json_present"] is False


# 13: crash BEFORE window commit rolls back to previous generation
@_mark
def test_crash_before_window_commit_rolls_back(tmp):
    kind = "mpi_nonlocal_mixing_heavy"
    out = Path(tmp) / "o_crashbefore"; wd = Path(tmp) / "w_crashbefore"
    shutil.rmtree(out, ignore_errors=True); shutil.rmtree(wd, ignore_errors=True)

    def _cmd(verify):
        c = ["mpirun", "-np", "2", sys.executable, "-m",
             "wenbo_engine.bench.communication_workloads", "--kind", kind,
             "--n", "12", "--depth", "20", "--recovery", "generation",
             "--planner", "recovery_aware_v1", "--mpi-exchange-mode",
             "gate_aware", "--mpi-window-execution", "safe", "--ram-budget-gib",
             "21", "--output-dir", str(out), "--work-dir", str(wd)]
        return c + (["--verify"] if verify else [])
    env = dict(os.environ, PYTHONPATH=REPO)
    # crash inside the window commit (before the global commit of the window
    # generation).  In window mode the only AFTER_PARTIAL_WRITE is the window's
    # own writer, so no stage filter is needed (layout-independent).
    crash_env = dict(env, WE_FAULT_POINT="AFTER_PARTIAL_WRITE",
                     WE_FAULT_RANK="0")
    r1 = subprocess.run(_cmd(False), env=crash_env, capture_output=True,
                        timeout=400)
    assert r1.returncode != 0                       # it crashed
    # no window generation committed → resume rolls back, recomputes, correct
    r2 = subprocess.run(_cmd(True), env=env, capture_output=True, timeout=400)
    assert r2.returncode == 0, r2.stderr.decode()[-2000:]
    fs = json.load(open(glob.glob(str(out / "**" / "final_summary.json"),
                                  recursive=True)[0]))
    assert fs["correct"] is True
    assert fs["mpi_windows_executed"] >= 1          # re-executed after rollback


# 14: crash AFTER window commit recovers the window generation
@_mark
def test_crash_after_window_commit_recovers(tmp):
    kind = "mpi_nonlocal_mixing_heavy"
    out = Path(tmp) / "o_crashafter"; wd = Path(tmp) / "w_crashafter"
    shutil.rmtree(out, ignore_errors=True); shutil.rmtree(wd, ignore_errors=True)

    def _cmd(verify):
        c = ["mpirun", "-np", "2", sys.executable, "-m",
             "wenbo_engine.bench.communication_workloads", "--kind", kind,
             "--n", "12", "--depth", "20", "--recovery", "generation",
             "--planner", "recovery_aware_v1", "--mpi-exchange-mode",
             "gate_aware", "--mpi-window-execution", "safe", "--ram-budget-gib",
             "21", "--output-dir", str(out), "--work-dir", str(wd)]
        return c + (["--verify"] if verify else [])
    env = dict(os.environ, PYTHONPATH=REPO)
    # crash right after the window commits its generation (fires once committed)
    r1 = subprocess.run(_cmd(False), env=dict(env, WE_CRASH_AFTER_STEP="1"),
                        capture_output=True, timeout=400)
    assert r1.returncode != 0
    # resume must recover the committed window generation, NOT re-run it
    r2 = subprocess.run(_cmd(True), env=env, capture_output=True, timeout=400)
    assert r2.returncode == 0, r2.stderr.decode()[-2000:]
    assert b"resuming from gen" in r2.stdout + r2.stderr   # recovered a committed gen
    fs = json.load(open(glob.glob(str(out / "**" / "final_summary.json"),
                                  recursive=True)[0]))
    assert fs["correct"] is True
    assert fs["mpi_windows_executed"] == 0          # recovered, not re-run


# 18: metrics are written to final_summary.json
@_mark
def test_metrics_in_final_summary(tmp):
    fs, _, _, _ = _run(tmp, "mpi_nonlocal_mixing_heavy", "safe")
    for k in ("mpi_window_execution_mode", "mpi_windows_executed",
              "mpi_window_steps_executed", "mpi_window_gates_executed",
              "mpi_window_gather_bytes", "mpi_window_scatter_bytes",
              "mpi_window_sendrecv_count", "mpi_window_commits_saved",
              "mpi_window_estimated_ram_gib", "mpi_window_fallbacks",
              "mpi_window_fallback_reasons",
              "expected_recomputation_cost_increase", "final_norm"):
        assert k in fs, k


# calibrated-cost-model telemetry: a window run reports its gather/scatter/
# leader/segment timing (explains why SAFE can cut bytes yet be slower).
@_mark
def test_window_reports_timing_breakdown(tmp):
    fs, _, _, _ = _run(tmp, "mpi_nonlocal_mixing_heavy", "safe")
    assert fs["mpi_windows_executed"] >= 1
    for k in ("mpi_collective_gather_time", "mpi_collective_scatter_time",
              "mpi_window_leader_compute_time", "mpi_window_segment_time"):
        assert k in fs and fs[k] > 0, k
    # the window's own collective traffic is reported separately from pairwise
    assert fs["mpi_window_gather_bytes"] > 0
