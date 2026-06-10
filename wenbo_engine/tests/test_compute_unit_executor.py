"""Compute-unit executor: correctness + recovery + MPI equivalence."""
import glob
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from wenbo_engine.storage.block_store import (
    write_chunk_atomic, read_chunk, chunk_filename, DTYPE,
)
from wenbo_engine.kernel import gates as gmod
from wenbo_engine.kernel.ref_dense import simulate
from wenbo_engine.mpi.mpi_runner import _apply_local_ops
from wenbo_engine.runtime.compute_unit import ComputeUnit, execute_local_unit
from wenbo_engine.runtime.overlay_scheduler import build_compute_units

REPO = str(Path(__file__).resolve().parent.parent.parent)


# ── 5. a compute unit applies multiple gates and matches reference ──────

def test_compute_unit_matches_reference(tmp_path):
    # 2-qubit circuit, whole state in one chunk; all gates chunk-local.
    n = 2
    circ = {"number_of_qubits": n, "gates": [
        {"qubits": [0], "gate": "H"}, {"qubits": [1], "gate": "H"},
        {"qubits": [0, 1], "gate": "CZ"}, {"qubits": [0], "gate": "H"}]}
    ref = simulate(circ).astype(DTYPE)
    ops = [(g["qubits"], gmod.gate_matrix(g["gate"], g.get("params", {})).astype(DTYPE))
           for g in circ["gates"]]
    src = tmp_path / "g0" / "chunks"
    src.mkdir(parents=True)
    init = np.zeros(1 << n, dtype=DTYPE); init[0] = 1.0
    write_chunk_atomic(src / chunk_filename(0), init)

    unit = ComputeUnit(compute_unit_id=0, kind="local", src_generation=0,
                       dst_generation=1, rank=0, chunk_ids=[0], local_ops=ops)
    ov = execute_local_unit(unit, src, tmp_path / "g1" / "chunks", _apply_local_ops)
    out = read_chunk(tmp_path / "g1" / "chunks" / chunk_filename(0))
    assert np.allclose(out, ref, atol=1e-6)
    # one load + one writeback for the whole multi-gate unit
    assert ov.load_count == 1 and ov.writeback_count == 1


def test_compute_unit_equals_sequential_steps(tmp_path):
    # multi-chunk: fused unit must equal applying each step's ops per chunk.
    n_chunks, cs = 4, 8
    rng = np.random.default_rng(0)
    src = tmp_path / "g0" / "chunks"; src.mkdir(parents=True)
    base = {}
    for c in range(n_chunks):
        a = (rng.standard_normal(cs) + 1j * rng.standard_normal(cs)).astype(DTYPE)
        base[c] = a.copy()
        write_chunk_atomic(src / chunk_filename(c), a)
    H = gmod.gate_matrix("H", {}).astype(DTYPE)
    ops = [([0], H), ([1], H), ([2], H)]   # all chunk-local (cs=2^3)
    unit = ComputeUnit(compute_unit_id=0, kind="local", src_generation=0,
                       dst_generation=1, rank=0,
                       chunk_ids=list(range(n_chunks)), local_ops=ops)
    execute_local_unit(unit, src, tmp_path / "g1" / "chunks", _apply_local_ops)
    for c in range(n_chunks):
        expect = base[c].copy()
        _apply_local_ops(expect, ops)       # same kernels, applied directly
        got = read_chunk(tmp_path / "g1" / "chunks" / chunk_filename(c))
        assert np.allclose(got, expect, atol=1e-6)


# ── scheduler fuses consecutive local-only steps, keeps nonlocal separate ─

def test_scheduler_fuses_local_runs():
    steps = [
        {"local_ops": [([0], None)], "rank_nonlocal_ops": [], "mpi_nonlocal_ops": []},
        {"local_ops": [([1], None)], "rank_nonlocal_ops": [], "mpi_nonlocal_ops": []},
        {"local_ops": [], "rank_nonlocal_ops": [], "mpi_nonlocal_ops": [([5], None)]},
        {"local_ops": [([0], None)], "rank_nonlocal_ops": [], "mpi_nonlocal_ops": []},
    ]
    units = build_compute_units(steps, rank=0, n_chunks_per_rank=2, start_gen=0)
    assert [u.kind for u in units] == ["local", "step", "local"]
    assert units[0].n_steps == 2 and len(units[0].local_ops) == 2   # fused
    assert units[1].kind == "step"
    assert [u.dst_generation for u in units] == [1, 2, 3]


# ── 12. partial output is not committed without a global commit ─────────

def test_partial_unit_output_not_committed(tmp_path):
    from wenbo_engine.recovery import (
        RankManifest, ChunkRecord, GlobalCommitRecord, RecoveryScanner,
        commits_dir, gen_dir,
    )
    from wenbo_engine.recovery.generation_manager import (
        RunMetadata, write_json_atomic, run_json_path, sha256_file,
    )
    CH = "cu0001"
    write_json_atomic(run_json_path(tmp_path), RunMetadata(
        circuit_hash=CH, n_ranks=1, n_qubits=4, chunk_size=4, created=1.0).to_dict())
    # commit gen 0
    g0 = gen_dir(tmp_path, 0, 0); (g0 / "chunks").mkdir(parents=True)
    p = g0 / "chunks" / chunk_filename(0)
    write_chunk_atomic(p, np.zeros(4, dtype=DTYPE))
    rm = RankManifest(rank=0, generation=0, n_chunks=1, chunk_size=4,
                      dtype="complex64", circuit_hash=CH, parent_generation=-1,
                      stage_id=-1, chunks=[ChunkRecord(0, chunk_filename(0),
                      p.stat().st_size, sha256_file(p))], created=1.0)
    rm.write_atomic(g0)
    GlobalCommitRecord(generation=0, n_ranks=1, circuit_hash=CH, step_index=-1,
                       parent_generation=-1, rank_manifest_hashes={0: rm.manifest_hash},
                       created=1.0).write_atomic(commits_dir(tmp_path))
    # a compute unit writes gen 1 chunks via the overlay but NEVER commits
    unit = ComputeUnit(compute_unit_id=0, kind="local", src_generation=0,
                       dst_generation=1, rank=0, chunk_ids=[0],
                       local_ops=[([0], gmod.gate_matrix("X", {}).astype(DTYPE))])
    execute_local_unit(unit, g0 / "chunks", gen_dir(tmp_path, 0, 1) / "chunks",
                       _apply_local_ops)
    assert (gen_dir(tmp_path, 0, 1) / "chunks" / chunk_filename(0)).exists()
    # recovery ignores the uncommitted gen 1 → recovers gen 0
    res = RecoveryScanner(tmp_path).scan()
    assert res.generation == 0


# ── real-MPI: compute_unit == step, + recovery/extents/gate-aware ───────

pytest.importorskip("mpi4py")
_mark = pytest.mark.skipif(shutil.which("mpirun") is None, reason="no mpirun")


def _bench(execmode, tmp, *, kind="communication_light", n=6, depth=10,
           layout="chunks", exch="gate_aware", verify=False):
    out = Path(tmp) / f"out_{execmode}_{layout}"
    wd = Path(tmp) / f"wd_{execmode}_{layout}"
    for p in (out, wd):
        shutil.rmtree(p, ignore_errors=True)
    cmd = ["mpirun", "-np", "2", sys.executable, "-m",
           "wenbo_engine.bench.communication_workloads", "--kind", kind,
           "--n", str(n), "--depth", str(depth), "--recovery", "generation",
           "--mpi-exchange-mode", exch, "--storage-layout", layout,
           "--execution-mode", execmode, "--output-dir", str(out),
           "--work-dir", str(wd)]
    if verify:
        cmd.append("--verify")
    r = subprocess.run(cmd, env=dict(os.environ, PYTHONPATH=REPO),
                       capture_output=True, timeout=400)
    assert r.returncode == 0, r.stderr.decode()[-1500:]
    return json.load(open(glob.glob(str(out / "**" / "final_summary.json"),
                                    recursive=True)[0]))


@_mark
def test_compute_unit_equals_step_mpi(tmp_path):
    s = _bench("step", tmp_path, verify=True)
    c = _bench("compute_unit", tmp_path, verify=True)
    assert s["correct"] is True and c["correct"] is True
    assert abs(s["final_norm"] - c["final_norm"]) < 1e-9


@_mark
def test_compute_unit_recovery_extents_gate_aware(tmp_path):
    c = _bench("compute_unit", tmp_path, layout="extents", exch="gate_aware")
    assert abs(c["final_norm"] - 1.0) < 1e-5
    assert c["recovery_mode"] == "generation"
    assert c.get("execution_mode") == "compute_unit"
    assert c.get("compute_units_executed", 0) >= 1
