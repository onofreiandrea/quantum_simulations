"""Benchmark validity: the communication workloads must be non-stabilizer.

A circuit built only from Clifford gates (H, S, CNOT, CZ, …) is a *stabilizer*
circuit — efficiently classically simulable via Gottesman–Knill — and therefore
a weak stress test for a full state-vector engine.  These tests pin:

  * every workload generator produces a non-stabilizer (non-Clifford) circuit;
  * ``mixed_staged`` carries non-Clifford gates in all three locality phases;
  * the Clifford classifier's R(k) / CR(k) boundaries are correct;
  * the non-Clifford injection preserves each workload's locality class
    (rank-nonlocal stays rank-nonlocal, MPI-nonlocal stays MPI-nonlocal);
  * ``final_summary.json`` carries the stabilizer metadata.
"""
import glob
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from wenbo_engine.bench.communication_workloads import (
    build_circuit, communication_light, rank_nonlocal_heavy,
    mpi_nonlocal_heavy, mixed_staged, classify_circuit, classify_gate,
    circuit_clifford_stats, is_clifford_gate, _layout,
)

REPO = str(Path(__file__).resolve().parent.parent.parent)

# A representative layout used across the in-process tests.
N, DEPTH, CHUNK_BITS, NUM_RANKS, SEED = 8, 30, 2, 4, 42


# ── per-workload: non-stabilizer ────────────────────────────────────────

@pytest.mark.parametrize("kind", [
    "communication_light", "rank_nonlocal_heavy",
    "mpi_nonlocal_heavy", "mixed_staged",
])
def test_workload_is_non_stabilizer(kind):
    cd = build_circuit(kind, N, DEPTH, CHUNK_BITS, NUM_RANKS, SEED)
    stats = circuit_clifford_stats(cd)
    assert stats["is_stabilizer"] is False, kind
    assert stats["non_clifford_gate_count"] > 0, kind
    assert stats["non_clifford_gate_types"], kind


# ── communication_light: DETERMINISTICALLY non-stabilizer (every seed) ──

@pytest.mark.parametrize("seed", [0, 1, 2, 7, 13, 42, 99, 123, 1000, 2024])
@pytest.mark.parametrize("depth", [1, 2, 5, 20, 50])
def test_communication_light_deterministic_non_stabilizer(seed, depth):
    """Non-stabilizer for ANY seed/depth, not just the default ones."""
    cd = communication_light(N, depth, seed=seed)
    stats = circuit_clifford_stats(cd)
    assert stats["is_stabilizer"] is False, (seed, depth)
    assert stats["non_clifford_gate_count"] > 0, (seed, depth)
    # locality unchanged: all-local, zero MPI traffic, gate count preserved
    loc = classify_circuit(cd, CHUNK_BITS, NUM_RANKS)
    assert loc["mpi_nonlocal_gate_count"] == 0, (seed, depth)
    assert loc["local_gate_count"] == depth, (seed, depth)


# ── mixed_staged: non-Clifford in EACH locality phase ───────────────────

def test_mixed_staged_non_clifford_in_every_phase():
    """local, rank-nonlocal AND mpi-nonlocal phases each contain non-Clifford."""
    cd = mixed_staged(N, DEPTH, CHUNK_BITS, num_ranks=NUM_RANKS, seed=SEED)
    k, p, n_local_bits = _layout(N, CHUNK_BITS, NUM_RANKS)
    by_class = {"local": [], "rank_nonlocal": [], "mpi_nonlocal": []}
    for g in cd["gates"]:
        by_class[classify_gate(g["qubits"], k, n_local_bits)].append(g)
    for cls, gates in by_class.items():
        assert gates, f"no gates in {cls} phase"
        n_nc = sum(0 if is_clifford_gate(g) else 1 for g in gates)
        assert n_nc > 0, f"{cls} phase is Clifford-only"


# ── Clifford classifier: R(k) / CR(k) boundaries ────────────────────────

def test_cr1_is_clifford():
    # CR(1) = CZ
    assert is_clifford_gate({"qubits": [0, 1], "gate": "CR", "params": {"k": 1}})


def test_cr2_is_non_clifford():
    # CR(2) = controlled-S — a controlled-Clifford is NOT itself Clifford
    assert not is_clifford_gate({"qubits": [0, 1], "gate": "CR", "params": {"k": 2}})


@pytest.mark.parametrize("k", [3, 4, 5])
def test_cr3plus_is_non_clifford(k):
    assert not is_clifford_gate({"qubits": [0, 1], "gate": "CR", "params": {"k": k}})


@pytest.mark.parametrize("k", [1, 2])
def test_r1_r2_are_clifford(k):
    # R(1) = Z, R(2) = S
    assert is_clifford_gate({"qubits": [0], "gate": "R", "params": {"k": k}})


@pytest.mark.parametrize("k", [3, 4, 5])
def test_r3plus_is_non_clifford(k):
    # R(3) = T, …
    assert not is_clifford_gate({"qubits": [0], "gate": "R", "params": {"k": k}})


def test_fixed_gate_clifford_classification():
    for g in ("H", "X", "Y", "Z", "S", "CNOT", "CZ", "CY", "SWAP"):
        assert is_clifford_gate({"qubits": [0], "gate": g}), g
    for g in ("T", "TDG"):
        assert not is_clifford_gate({"qubits": [0], "gate": g}), g


def test_rotation_clifford_only_at_multiples_of_half_pi():
    assert is_clifford_gate({"qubits": [0], "gate": "RZ", "params": {"theta": math.pi / 2}})
    assert is_clifford_gate({"qubits": [0], "gate": "RZ", "params": {"theta": math.pi}})
    assert not is_clifford_gate({"qubits": [0], "gate": "RZ", "params": {"theta": math.pi / 4}})


# ── locality class preserved by the non-Clifford injection ──────────────

def test_rank_nonlocal_heavy_locality_preserved():
    cd = rank_nonlocal_heavy(N, DEPTH, CHUNK_BITS, seed=SEED, num_ranks=NUM_RANKS)
    info = classify_circuit(cd, CHUNK_BITS, NUM_RANKS)
    assert info["rank_nonlocal_gate_count"] > 0
    assert info["mpi_nonlocal_gate_count"] == 0
    assert info["local_gate_count"] == 0


def test_mpi_nonlocal_heavy_locality_preserved():
    cd = mpi_nonlocal_heavy(N, DEPTH, CHUNK_BITS, NUM_RANKS, seed=SEED)
    info = classify_circuit(cd, CHUNK_BITS, NUM_RANKS)
    assert info["mpi_nonlocal_gate_count"] > 0
    assert info["rank_nonlocal_gate_count"] == 0
    assert info["local_gate_count"] == 0


def test_communication_light_stays_mpi_light():
    cd = communication_light(N, DEPTH, seed=SEED)
    info = classify_circuit(cd, CHUNK_BITS, NUM_RANKS)
    assert info["mpi_nonlocal_gate_count"] == 0
    assert info["local_gate_count"] == DEPTH


# ── real-MPI: final_summary carries stabilizer metadata + MPI stress ────

pytest.importorskip("mpi4py")
_mark = pytest.mark.skipif(shutil.which("mpirun") is None, reason="no mpirun")


def _bench(kind, tmp, *, n=6, depth=12, ranks=2, exch="gate_aware", verify=False):
    out = Path(tmp) / f"out_{kind}"
    wd = Path(tmp) / f"wd_{kind}"
    for p in (out, wd):
        shutil.rmtree(p, ignore_errors=True)
    cmd = ["mpirun", "-np", str(ranks), sys.executable, "-m",
           "wenbo_engine.bench.communication_workloads", "--kind", kind,
           "--n", str(n), "--depth", str(depth), "--recovery", "generation",
           "--mpi-exchange-mode", exch, "--output-dir", str(out),
           "--work-dir", str(wd)]
    if verify:
        cmd.append("--verify")
    r = subprocess.run(cmd, env=dict(os.environ, PYTHONPATH=REPO),
                       capture_output=True, timeout=400)
    assert r.returncode == 0, r.stderr.decode()[-1500:]
    fs = glob.glob(str(out / "**" / "final_summary.json"), recursive=True)[0]
    return json.load(open(fs))


@_mark
def test_final_summary_has_stabilizer_metadata(tmp_path):
    s = _bench("communication_light", tmp_path, verify=True)
    assert s["is_stabilizer"] is False
    assert s["non_clifford_gate_count"] > 0
    assert isinstance(s["non_clifford_gate_types"], list) and s["non_clifford_gate_types"]
    assert s["clifford_gate_count"] + s["non_clifford_gate_count"] == s["total_gate_count"]
    assert s["correct"] is True                         # still matches ref_dense
    assert abs(s["final_norm"] - 1.0) < 1e-5


@_mark
def test_mpi_heavy_stays_mpi_heavy_and_non_stabilizer(tmp_path):
    s = _bench("mpi_nonlocal_heavy", tmp_path)
    assert s["is_stabilizer"] is False
    assert s["non_clifford_gate_count"] > 0
    assert s["measured_mpi_nonlocal_ops"] > 0            # MPI stress preserved
    assert s["mpi_bytes_sent"] > 0                       # real Sendrecv traffic
    assert abs(s["final_norm"] - 1.0) < 1e-5
