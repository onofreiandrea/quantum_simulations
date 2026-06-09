"""Tests for the MPI-nonlocal communication benchmark suite.

Pure (no-MPI) tests cover the generators and the static gate classifier.
The end-to-end correctness test launches a real 2-rank ``mpirun`` and
checks the resulting state against the ``ref_dense`` oracle, plus asserts
that the run produced nonzero MPI traffic and a profile artifact.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from wenbo_engine.bench.communication_workloads import (
    classify_circuit,
    communication_light,
    mixed_staged,
    mpi_nonlocal_heavy,
    rank_nonlocal_heavy,
    build_circuit,
)
from wenbo_engine.circuit.io import validate_circuit_dict

REPO_ROOT = Path(__file__).resolve().parents[2]

HAS_MPI = shutil.which("mpirun") is not None
try:
    import mpi4py  # noqa: F401
    HAS_MPI4PY = True
except ImportError:
    HAS_MPI4PY = False


def _all_qubits(cd):
    return [q for g in cd["gates"] for q in g["qubits"]]


# ── 1. determinism ──────────────────────────────────────────────────────

@pytest.mark.parametrize("kind", [
    "communication_light", "rank_nonlocal_heavy",
    "mpi_nonlocal_heavy", "mixed_staged",
])
def test_generator_deterministic(kind):
    a = build_circuit(kind, n=16, depth=30, chunk_bits=8, num_ranks=4, seed=7)
    b = build_circuit(kind, n=16, depth=30, chunk_bits=8, num_ranks=4, seed=7)
    c = build_circuit(kind, n=16, depth=30, chunk_bits=8, num_ranks=4, seed=8)
    dump = lambda cd: json.dumps(cd, sort_keys=True)
    assert dump(a) == dump(b), "same seed must produce identical circuit"
    assert dump(a) != dump(c), "different seed must produce different circuit"


@pytest.mark.parametrize("kind", [
    "communication_light", "rank_nonlocal_heavy",
    "mpi_nonlocal_heavy", "mixed_staged",
])
def test_generated_circuits_valid(kind):
    cd = build_circuit(kind, n=16, depth=30, chunk_bits=8, num_ranks=4, seed=1)
    validate_circuit_dict(cd)  # raises on malformed gates
    assert len(cd["gates"]) >= 1


# ── 2. communication_light: mostly low-bit gates ────────────────────────

def test_communication_light_is_low_bit():
    n, depth = 16, 60
    cd = communication_light(n, depth, seed=3)
    low = max(1, n // 4)
    # Every gate acts only on the lowest `low` qubits by construction.
    assert all(q < low for q in _all_qubits(cd))
    # Under a partition where those qubits are all chunk-local, zero MPI.
    info = classify_circuit(cd, chunk_bits=8, num_ranks=4)
    assert info["mpi_nonlocal_gate_count"] == 0
    assert info["rank_nonlocal_gate_count"] == 0
    assert info["local_gate_count"] == depth


# ── 3. mpi_nonlocal_heavy: includes rank-bit gates ──────────────────────

def test_mpi_nonlocal_heavy_has_rank_bit_gates():
    n, depth, chunk_bits, num_ranks = 12, 40, 8, 4
    cd = mpi_nonlocal_heavy(n, depth, chunk_bits, num_ranks, seed=5)
    info = classify_circuit(cd, chunk_bits, num_ranks)
    # Every gate was deliberately placed on a rank bit.
    assert info["mpi_nonlocal_gate_count"] == depth
    assert info["rank_nonlocal_gate_count"] == 0
    # p = log2(4) = 2 rank bits -> the two highest qubits.
    p = 2
    assert any(q >= n - p for q in _all_qubits(cd))
    # Communication actually spans ranks.
    assert info["partner_rank_pairs"] > 0


def test_rank_nonlocal_heavy_stays_on_rank():
    n, depth, chunk_bits, num_ranks = 12, 40, 8, 4
    cd = rank_nonlocal_heavy(n, depth, chunk_bits, seed=5)
    info = classify_circuit(cd, chunk_bits, num_ranks)
    assert info["rank_nonlocal_gate_count"] == depth
    assert info["mpi_nonlocal_gate_count"] == 0
    assert info["partner_rank_pairs"] == 0


def test_mixed_staged_spans_all_classes():
    n, depth, chunk_bits, num_ranks = 12, 30, 8, 4
    cd = mixed_staged(n, depth, chunk_bits, num_ranks, seed=2)
    info = classify_circuit(cd, chunk_bits, num_ranks)
    assert info["local_gate_count"] > 0
    assert info["rank_nonlocal_gate_count"] > 0
    assert info["mpi_nonlocal_gate_count"] > 0


# ── 4. small generated circuit runs against the reference ───────────────

def _run_cli(tmp, kind, n, depth, chunk_bits, np_ranks):
    """Launch the benchmark CLI under mpirun, return the parsed artifact."""
    out = Path(tmp) / "profile.json"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    cmd = [
        "mpirun", "--oversubscribe", "-np", str(np_ranks),
        sys.executable, "-m", "wenbo_engine.bench.communication_workloads",
        "--kind", kind, "--n", str(n), "--depth", str(depth),
        "--chunk-bits", str(chunk_bits),
        "--work-dir", str(Path(tmp) / "work"),
        "--output", str(out), "--verify",
    ]
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True,
                          timeout=300)
    assert proc.returncode == 0, (
        f"mpirun failed (code {proc.returncode}):\n"
        f"STDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}")
    assert out.exists(), f"profile artifact not written\nSTDERR:\n{proc.stderr}"
    return json.loads(out.read_text())


@pytest.mark.skipif(not (HAS_MPI and HAS_MPI4PY),
                    reason="mpirun / mpi4py not available")
def test_mpi_nonlocal_heavy_runs_correctly():
    # n=5, chunk_bits=2, 2 ranks -> p=1, n_local_bits=2, qubit 4 is a rank bit.
    with tempfile.TemporaryDirectory() as tmp:
        res = _run_cli(tmp, "mpi_nonlocal_heavy", n=5, depth=8,
                       chunk_bits=2, np_ranks=2)
    assert res["correct"] is True, "final state must match ref_dense"
    assert res["mpi_nonlocal_gate_count"] > 0
    assert res["aggregate"]["mpi_bytes_sent"] > 0, "expected nonzero MPI traffic"
    assert res["aggregate"]["sendrecv_count"] > 0
    assert res["partner_rank_pairs"] > 0


@pytest.mark.skipif(not (HAS_MPI and HAS_MPI4PY),
                    reason="mpirun / mpi4py not available")
def test_mixed_staged_runs_correctly():
    with tempfile.TemporaryDirectory() as tmp:
        res = _run_cli(tmp, "mixed_staged", n=6, depth=12,
                       chunk_bits=3, np_ranks=2)
    assert res["correct"] is True
    assert res["mpi_nonlocal_gate_count"] > 0
    assert res["aggregate"]["mpi_bytes_sent"] > 0
    # I/O and kernels were exercised through the instrumented runner.
    assert res["aggregate"]["bytes_written"] > 0
    assert res["aggregate"]["kernel_time"] >= 0.0
