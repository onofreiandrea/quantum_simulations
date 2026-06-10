"""Direct extent-backed overlay I/O (``--extent-io-mode direct``).

In compute_unit mode with the extents layout, ``direct`` reads logical chunks
straight from their source extent slices and writes dirty chunks straight into
destination extent files, skipping the materialize→chunks→pack round trip used
by ``materialize`` (the default).  These tests pin:

  * direct extent reads return the same bytes as a materialized read;
  * the memory overlay writes each dirty chunk exactly once and never writes a
    clean chunk;
  * a directly-written generation recovers (and is rejected when its extents
    are missing / short / corrupt) — the on-disk format is identical to the
    pack-written extent layout, so recovery semantics are unchanged;
  * the default extent-io-mode is ``materialize`` and chunks mode is untouched;
  * under real MPI: direct == materialize final state, generation recovery +
    gate-aware MPI are preserved, the adaptive fallback still engages, direct
    creates zero temporary chunk files, and MPI-heavy stays MPI-heavy.
"""
import glob
import json
import math
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from wenbo_engine.storage.block_store import (
    read_chunk, chunk_filename, DTYPE,
)
from wenbo_engine.storage.extent_store import (
    write_generation_extents, materialize_to_chunk_files,
    read_chunk_from_extent, read_logical_chunk, ExtentWriter, EXTENTS_DIRNAME,
)
from wenbo_engine.storage.extent_manifest import extent_filename
from wenbo_engine.runtime.memory_overlay import MemoryOverlay
from wenbo_engine.runtime.compute_unit import (
    ComputeUnit, execute_local_unit, execute_local_unit_direct,
)
from wenbo_engine.mpi.mpi_runner import _apply_local_ops
from wenbo_engine.kernel import gates as gmod
from wenbo_engine.recovery import (
    RankManifest, ChunkRecord, GlobalCommitRecord, RecoveryScanner,
    commits_dir, gen_dir,
)
from wenbo_engine.recovery.generation_manager import (
    RunMetadata, write_json_atomic, run_json_path,
)

REPO = str(Path(__file__).resolve().parent.parent.parent)
CH = "directcafe00001"
CS = 4          # chunk size (elements)
NC = 4          # chunks per rank


def _src_records(man):
    """rank-manifest ChunkRecords (extent_id/extent_offset/size_bytes) from an
    ExtentManifest."""
    return {ci: ChunkRecord(index=ci, filename=chunk_filename(ci),
                            size_bytes=man.records[ci].length,
                            checksum=man.records[ci].checksum,
                            extent_id=man.records[ci].extent_id,
                            extent_offset=man.records[ci].offset)
            for ci in man.records}


def _build_src_extents(gen, chunks):
    man = write_generation_extents(gen, chunks, chunk_size=CS)
    return man, _src_records(man)


# ── 1. direct read == materialized read ─────────────────────────────────

def test_direct_read_equals_materialized_read(tmp_path):
    rng = np.random.default_rng(1)
    chunks = {c: (rng.standard_normal(CS) + 1j * rng.standard_normal(CS)).astype(DTYPE)
              for c in range(NC)}
    g = tmp_path / "g0"
    man, recs = _build_src_extents(g, chunks)
    # direct slice read
    for c in range(NC):
        r = recs[c]
        got = read_chunk_from_extent(g, r.extent_id, r.extent_offset, r.size_bytes)
        assert np.array_equal(got, chunks[c])
    # materialized read (unpack to chunk files) returns identical bytes
    materialize_to_chunk_files(g, recs.values(), force=True)
    for c in range(NC):
        mat = read_chunk(g / "chunks" / chunk_filename(c))
        assert np.array_equal(mat, chunks[c])


# ── 2. direct unit final state matches the materialize path ─────────────

def test_direct_unit_matches_materialize(tmp_path):
    rng = np.random.default_rng(2)
    chunks = {c: (rng.standard_normal(CS) + 1j * rng.standard_normal(CS)).astype(DTYPE)
              for c in range(NC)}
    H = gmod.gate_matrix("H", {}).astype(DTYPE)
    ops = [([0], H), ([1], H)]              # CS = 2^2 → both chunk-local
    unit = ComputeUnit(compute_unit_id=0, kind="local", src_generation=0,
                       dst_generation=1, rank=0,
                       chunk_ids=list(range(NC)), local_ops=ops)

    # direct: extents → overlay → extents
    gd = tmp_path / "direct_src"
    _man, recs = _build_src_extents(gd, chunks)
    dst_direct = tmp_path / "direct_dst"
    _ov, out_man = execute_local_unit_direct(
        unit, gd, dst_direct, recs, _apply_local_ops, chunk_size=CS)
    direct_out = {c: read_logical_chunk(dst_direct, out_man, c) for c in range(NC)}

    # materialize: extents → chunk files → overlay → chunk files
    gm = tmp_path / "mat_src"
    _man2, recs2 = _build_src_extents(gm, chunks)
    materialize_to_chunk_files(gm, recs2.values(), force=True)
    dst_mat = tmp_path / "mat_dst" / "chunks"
    execute_local_unit(unit, gm / "chunks", dst_mat, _apply_local_ops)
    for c in range(NC):
        mat = read_chunk(dst_mat / chunk_filename(c))
        assert np.allclose(direct_out[c], mat, atol=1e-6)


# ── 3. overlay writes each dirty chunk once; never writes a clean chunk ──

def test_overlay_writes_dirty_once_clean_never():
    data = {0: np.arange(CS, dtype=DTYPE), 1: np.arange(CS, dtype=DTYPE) + 10}
    writes = []

    def reader(cid):
        return data[cid].copy()

    def writer(cid, arr):
        writes.append(cid)

    ov = MemoryOverlay(reader=reader, writer=writer)
    a = ov.get(0)
    ov.mark_dirty(0)
    ov.writeback(0)
    ov.writeback(0)            # idempotent: already flushed
    _b = ov.get(1)             # loaded but never marked dirty
    ov.writeback(1)            # clean → no write
    assert writes == [0]
    assert ov.load_count == 2 and ov.writeback_count == 1


# ── build a directly-written committed generation (1 rank) ──────────────

def _write_run(work):
    write_json_atomic(run_json_path(work), RunMetadata(
        circuit_hash=CH, n_ranks=1, n_qubits=4, chunk_size=CS,
        created=1.0).to_dict())


def _commit_direct_gen(work, gen, *, fill, parent, stage):
    """Commit a generation whose payload was written by ExtentWriter (direct)."""
    gdir = gen_dir(work, 0, gen)
    gdir.mkdir(parents=True, exist_ok=True)
    ew = ExtentWriter(gdir, CS)
    for ci in range(NC):
        ew.append(ci, np.full(CS, fill + ci, dtype=DTYPE))
    man = ew.finalize()
    recs = list(_src_records(man).values())
    rm = RankManifest(rank=0, generation=gen, n_chunks=NC, chunk_size=CS,
                      dtype="complex64", circuit_hash=CH,
                      parent_generation=parent, stage_id=stage,
                      chunks=recs, created=1.0)
    rm.write_atomic(gdir)
    GlobalCommitRecord(generation=gen, n_ranks=1, circuit_hash=CH,
                       step_index=stage, parent_generation=parent,
                       rank_manifest_hashes={0: rm.manifest_hash},
                       created=1.0).write_atomic(commits_dir(work))
    return rm


# ── 4. direct-written generation recovers ───────────────────────────────

def test_direct_written_generation_recovers(tmp_path):
    _write_run(tmp_path)
    _commit_direct_gen(tmp_path, 0, fill=1.0, parent=-1, stage=-1)
    g0 = gen_dir(tmp_path, 0, 0)
    assert (g0 / EXTENTS_DIRNAME).exists()          # extent-backed
    res = RecoveryScanner(tmp_path).scan(check_checksums=True)
    assert res.recovered and res.generation == 0


# ── 5–7. corrupt / missing / short direct extent is rejected ────────────

def test_direct_missing_extent_rejected(tmp_path):
    _write_run(tmp_path)
    _commit_direct_gen(tmp_path, 0, fill=1.0, parent=-1, stage=-1)
    (gen_dir(tmp_path, 0, 0) / EXTENTS_DIRNAME / extent_filename(0)).unlink()
    assert RecoveryScanner(tmp_path).scan().recovered is False


def test_direct_short_extent_rejected(tmp_path):
    _write_run(tmp_path)
    _commit_direct_gen(tmp_path, 0, fill=1.0, parent=-1, stage=-1)
    ext = gen_dir(tmp_path, 0, 0) / EXTENTS_DIRNAME / extent_filename(0)
    ext.write_bytes(ext.read_bytes()[:8])           # truncate
    assert RecoveryScanner(tmp_path).scan().recovered is False


def test_direct_corrupt_extent_rejected(tmp_path):
    _write_run(tmp_path)
    _commit_direct_gen(tmp_path, 0, fill=1.0, parent=-1, stage=-1)
    ext = gen_dir(tmp_path, 0, 0) / EXTENTS_DIRNAME / extent_filename(0)
    raw = bytearray(ext.read_bytes()); raw[0] ^= 0xFF; ext.write_bytes(bytes(raw))
    assert RecoveryScanner(tmp_path).scan(check_checksums=True).recovered is False


# ── 8. default extent-io-mode is materialize (chunks/step untouched) ─────

def test_default_extent_io_mode_is_materialize():
    import inspect
    from wenbo_engine.mpi.mpi_runner import run
    assert inspect.signature(run).parameters["extent_io_mode"].default == "materialize"


# ── real-MPI ────────────────────────────────────────────────────────────

pytest.importorskip("mpi4py")
_mark = pytest.mark.skipif(shutil.which("mpirun") is None, reason="no mpirun")


def _bench(io_mode, tmp, *, kind="communication_light", n=6, depth=12,
           ranks=2, exch="gate_aware", verify=False):
    out = Path(tmp) / f"out_{io_mode}_{kind}"
    wd = Path(tmp) / f"wd_{io_mode}_{kind}"
    for p in (out, wd):
        shutil.rmtree(p, ignore_errors=True)
    cmd = ["mpirun", "-np", str(ranks), sys.executable, "-m",
           "wenbo_engine.bench.communication_workloads", "--kind", kind,
           "--n", str(n), "--depth", str(depth), "--recovery", "generation",
           "--mpi-exchange-mode", exch, "--storage-layout", "extents",
           "--execution-mode", "compute_unit", "--extent-io-mode", io_mode,
           "--output-dir", str(out), "--work-dir", str(wd)]
    if verify:
        cmd.append("--verify")
    r = subprocess.run(cmd, env=dict(os.environ, PYTHONPATH=REPO),
                       capture_output=True, timeout=400)
    assert r.returncode == 0, r.stderr.decode()[-1500:]
    fs = glob.glob(str(out / "**" / "final_summary.json"), recursive=True)[0]
    return json.load(open(fs)), wd


@_mark
def test_direct_matches_materialize_mpi(tmp_path):
    m, _ = _bench("materialize", tmp_path, verify=True)
    d, _ = _bench("direct", tmp_path, verify=True)
    assert m["correct"] is True and d["correct"] is True
    assert abs(m["final_norm"] - d["final_norm"]) < 1e-9


@_mark
def test_direct_recovers_with_gate_aware_mpi(tmp_path):
    d, wd = _bench("direct", tmp_path, kind="mpi_nonlocal_heavy", n=8, depth=12)
    assert abs(d["final_norm"] - 1.0) < 1e-5
    assert d["recovery_mode"] == "generation"
    assert d["measured_mpi_nonlocal_ops"] > 0           # gate-aware MPI preserved
    assert d["mpi_bytes_sent"] > 0
    # generation-recovery invariants
    rev = json.load(open(glob.glob(str(wd.parent / "out_direct_mpi_nonlocal_heavy"
                                       / "**" / "recovery_events.json"),
                                   recursive=True)[0]))
    assert rev["source_of_truth"] == "global_commit_record"
    assert rev["wal_json_present"] is False
    # node-local: each rank's committed generation dir holds its own extents
    assert glob.glob(str(wd / "**" / EXTENTS_DIRNAME / "extent_*.dat"),
                     recursive=True)
    assert not glob.glob(str(wd / "**" / "wal.json"), recursive=True)


@_mark
def test_direct_creates_no_temp_chunk_files(tmp_path):
    m, _ = _bench("materialize", tmp_path, kind="communication_light", n=6, depth=12)
    d, _ = _bench("direct", tmp_path, kind="communication_light", n=6, depth=12)
    # all-local workload → direct path eliminates every temp chunk file
    assert d["temporary_chunk_files_created"] == 0
    assert d["read_ops"] == 0 and d["write_ops"] == 0
    assert m["temporary_chunk_files_created"] > 0       # materialize round-trips


@_mark
def test_direct_adaptive_fallback_engages(tmp_path):
    d, _ = _bench("direct", tmp_path, kind="mixed_staged", n=8, depth=16)
    assert abs(d["final_norm"] - 1.0) < 1e-5
    assert d.get("compute_unit_fallbacks", 0) >= 1      # short local runs fall back
    assert d.get("execution_mode") == "compute_unit"
