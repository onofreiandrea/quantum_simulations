"""Selectable numerical backend (numpy / numba / auto).

In-process tests pin backend selection, safe fallback, numpy==numba
equivalence (when numba is installed), and that precision (complex64) never
changes. Real-MPI smokes confirm the runner records the backend fields and
that compute_unit / gate-aware MPI / generation recovery all work, with both
backends producing the same final state.
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

from wenbo_engine.kernel import backend, numba_kernels
from wenbo_engine.kernel.cpu_batched import apply_1q, apply_2q
from wenbo_engine.kernel.ref_dense import simulate
from wenbo_engine.kernel import gates as gmod

REPO = str(Path(__file__).resolve().parent.parent.parent)
DTYPE = np.complex64
_HAVE_NUMBA = backend.numba_available()


@pytest.fixture(autouse=True)
def _reset_backend():
    yield
    backend._AVAILABLE_CACHE = None      # re-detect
    backend.set_backend("numpy")         # restore safe default


def _apply(circ, n, kernel_backend):
    backend.set_backend(kernel_backend)
    state = np.zeros(1 << n, dtype=DTYPE); state[0] = 1.0
    for g in circ["gates"]:
        U = gmod.gate_matrix(g["gate"], g.get("params", {})).astype(DTYPE)
        qs = g["qubits"]
        if len(qs) == 1:
            apply_1q(state, qs[0], U)
        else:
            apply_2q(state, qs[0], qs[1], U)
    return state


_CIRC = {"number_of_qubits": 5, "gates": [
    {"qubits": [0], "gate": "H"}, {"qubits": [1], "gate": "RZ", "params": {"theta": 0.37}},
    {"qubits": [0, 2], "gate": "CR", "params": {"k": 3}},   # non-Clifford 2q
    {"qubits": [3], "gate": "T"}, {"qubits": [1, 4], "gate": "CNOT"},
    {"qubits": [2], "gate": "H"}]}


# ── 1. numpy backend matches the dense reference ────────────────────────

def test_numpy_matches_reference():
    got = _apply(_CIRC, 5, "numpy")
    ref = simulate(_CIRC).astype(DTYPE)
    assert np.allclose(got, ref, atol=1e-5)
    assert backend.backend_info()["used"] == "numpy"


# ── 2. numba matches numpy within tolerance (if installed) ──────────────

@pytest.mark.skipif(not _HAVE_NUMBA, reason="numba not installed")
def test_numba_matches_numpy():
    a = _apply(_CIRC, 5, "numpy")
    b = _apply(_CIRC, 5, "numba")
    assert backend.backend_info()["used"] == "numba"
    assert np.allclose(a, b, atol=1e-5)


# ── 3. auto / explicit-numba fall back safely when numba unavailable ────

def test_auto_falls_back_to_numpy_when_unavailable():
    backend._AVAILABLE_CACHE = False                  # simulate no numba
    info = backend.set_backend("auto")
    assert info["used"] == "numpy"
    assert info["available"] is False
    assert info["fallback_reason"]                     # reason recorded


def test_explicit_numba_falls_back_when_unavailable():
    backend._AVAILABLE_CACHE = False
    info = backend.set_backend("numba")
    assert info["used"] == "numpy"
    assert "not installed" in info["fallback_reason"]


def test_set_backend_rejects_unknown():
    with pytest.raises(ValueError):
        backend.set_backend("cuda")


# ── 9. no backend silently changes precision ────────────────────────────

def test_precision_unchanged_complex64():
    for be in ("numpy", "numba") if _HAVE_NUMBA else ("numpy",):
        out = _apply(_CIRC, 5, be)
        assert out.dtype == np.complex64


def test_backend_info_has_all_fields():
    backend.set_backend("auto")
    info = backend.backend_info()
    for k in ("requested", "used", "available", "compile_time", "fallback_reason"):
        assert k in info
    assert numba_kernels.available() == backend.numba_available()


# ── real-MPI smokes ─────────────────────────────────────────────────────

pytest.importorskip("mpi4py")
_mark = pytest.mark.skipif(shutil.which("mpirun") is None, reason="no mpirun")


def _bench(tmp, kind, be, n=12, depth=12):
    out = Path(tmp) / f"out_{kind}_{be}"
    wd = Path(tmp) / f"wd_{kind}_{be}"
    for p in (out, wd):
        shutil.rmtree(p, ignore_errors=True)
    cmd = ["mpirun", "-np", "2", sys.executable, "-m",
           "wenbo_engine.bench.communication_workloads", "--kind", kind,
           "--n", str(n), "--depth", str(depth), "--recovery", "generation",
           "--planner", "recovery_aware_v1", "--kernel-backend", be, "--verify",
           "--output-dir", str(out), "--work-dir", str(wd)]
    r = subprocess.run(cmd, env=dict(os.environ, PYTHONPATH=REPO),
                       capture_output=True, timeout=400)
    assert r.returncode == 0, r.stderr.decode()[-2000:]
    fs = json.load(open(glob.glob(str(out / "**" / "final_summary.json"),
                                  recursive=True)[0]))
    rev = json.load(open(glob.glob(str(out / "**" / "recovery_events.json"),
                                   recursive=True)[0]))
    return fs, rev


# 4 + 5 + 7 + 8: final_summary backend fields; compute_unit; recovery; non-stab
@_mark
def test_mpi_compute_unit_backend_fields_and_correct(tmp_path):
    fs, rev = _bench(tmp_path, "communication_light", "numpy")
    for k in ("kernel_backend_requested", "kernel_backend_used",
              "kernel_backend_available", "numba_compile_time",
              "backend_fallback_reason", "kernel_time"):
        assert k in fs, k
    assert fs["kernel_backend_requested"] == "numpy"
    assert fs["kernel_backend_used"] == "numpy"
    assert fs["correct"] is True
    assert fs["is_stabilizer"] is False                 # case 8
    assert fs["execution_mode"] == "compute_unit"        # case 5
    assert fs["recovery_mode"] == "generation"           # case 7
    assert rev["source_of_truth"] == "global_commit_record"
    assert rev["wal_json_present"] is False


# 2 + 6: numba vs numpy same final state; gate-aware MPI with backend selection
@_mark
def test_mpi_numpy_vs_auto_same_state(tmp_path):
    npy, _ = _bench(tmp_path, "communication_light", "numpy")
    aut, _ = _bench(tmp_path, "communication_light", "auto")
    assert npy["correct"] and aut["correct"]
    assert abs(npy["final_norm"] - aut["final_norm"]) < 1e-6
    if _HAVE_NUMBA:
        assert aut["kernel_backend_used"] == "numba"
    # gate-aware MPI path with a chosen backend stays correct
    mpi_fs, mpi_rev = _bench(tmp_path, "mpi_nonlocal_heavy", "auto", n=8)
    assert mpi_fs["correct"] is True
    assert "gate_aware" in mpi_fs["selected_strategy"]
    assert mpi_fs["measured_mpi_nonlocal_ops"] > 0
    assert mpi_rev["wal_json_present"] is False
