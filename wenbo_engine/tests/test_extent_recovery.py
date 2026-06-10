"""Recovery + durable + MPI tests for the extent storage layout."""
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
from wenbo_engine.storage.extent_store import pack_chunk_files, EXTENTS_DIRNAME
from wenbo_engine.storage.extent_manifest import extent_filename
from wenbo_engine.recovery import (
    RankManifest, ChunkRecord, GlobalCommitRecord, RecoveryScanner,
    LocalCoordinator, commits_dir, gen_dir, gen_chunks_dir,
)
from wenbo_engine.recovery.generation_manager import (
    RunMetadata, write_json_atomic, run_json_path,
)
from wenbo_engine.durable import (
    LocalPathBackend, DurableCheckpointManager, DurableRestoreManager,
)

REPO = str(Path(__file__).resolve().parent.parent.parent)
CH = "extentcafe000001"
CS = 4
NC = 3


# ── build an extent-backed committed generation in-process (1 rank) ──────

def _write_run(work, n_ranks=1):
    write_json_atomic(run_json_path(work), RunMetadata(
        circuit_hash=CH, n_ranks=n_ranks, n_qubits=4, chunk_size=CS,
        created=1.0).to_dict())


def _build_extent_gen(work, gen, *, fill, parent, stage):
    gdir = gen_dir(work, 0, gen)
    cdir = gdir / "chunks"
    cdir.mkdir(parents=True, exist_ok=True)
    for ci in range(NC):
        write_chunk_atomic(cdir / chunk_filename(ci),
                           np.full(CS, fill + ci, dtype=DTYPE))
    man_ext = pack_chunk_files(gdir, NC, chunk_size=CS)   # chunks -> extents
    recs = [ChunkRecord(index=ci, filename=chunk_filename(ci),
                        size_bytes=man_ext.records[ci].length,
                        checksum=man_ext.records[ci].checksum,
                        extent_id=man_ext.records[ci].extent_id,
                        extent_offset=man_ext.records[ci].offset)
            for ci in range(NC)]
    rm = RankManifest(rank=0, generation=gen, n_chunks=NC, chunk_size=CS,
                      dtype="complex64", circuit_hash=CH,
                      parent_generation=parent, stage_id=stage,
                      chunks=recs, created=1.0)
    rm.write_atomic(gdir)
    rec = GlobalCommitRecord(generation=gen, n_ranks=1, circuit_hash=CH,
                             step_index=stage, parent_generation=parent,
                             rank_manifest_hashes={0: rm.manifest_hash},
                             created=1.0)
    rec.write_atomic(commits_dir(work))
    return rm


# ── recovery scanner validates / rejects extent generations ─────────────

def test_extent_generation_recovers(tmp_path):
    _write_run(tmp_path)
    _build_extent_gen(tmp_path, 0, fill=1.0, parent=-1, stage=-1)
    # the committed gen is extent-backed (extents/, no chunk files)
    g0 = gen_dir(tmp_path, 0, 0)
    assert (g0 / EXTENTS_DIRNAME).exists() and not (g0 / "chunks").exists()
    res = RecoveryScanner(tmp_path).scan(check_checksums=True)
    assert res.recovered and res.generation == 0


def test_missing_extent_rejected(tmp_path):
    _write_run(tmp_path)
    _build_extent_gen(tmp_path, 0, fill=1.0, parent=-1, stage=-1)
    (gen_dir(tmp_path, 0, 0) / EXTENTS_DIRNAME / extent_filename(0)).unlink()
    res = RecoveryScanner(tmp_path).scan()
    assert not res.recovered          # missing extent -> generation rejected


def test_wrong_extent_length_rejected(tmp_path):
    _write_run(tmp_path)
    _build_extent_gen(tmp_path, 0, fill=1.0, parent=-1, stage=-1)
    ext = gen_dir(tmp_path, 0, 0) / EXTENTS_DIRNAME / extent_filename(0)
    ext.write_bytes(ext.read_bytes()[:4])   # truncate
    res = RecoveryScanner(tmp_path).scan()
    assert not res.recovered


def test_extent_checksum_corruption_rejected(tmp_path):
    _write_run(tmp_path)
    _build_extent_gen(tmp_path, 0, fill=1.0, parent=-1, stage=-1)
    ext = gen_dir(tmp_path, 0, 0) / EXTENTS_DIRNAME / extent_filename(0)
    raw = bytearray(ext.read_bytes()); raw[0] ^= 0xFF; ext.write_bytes(bytes(raw))
    assert RecoveryScanner(tmp_path).scan(check_checksums=True).recovered is False


def test_newest_extent_invalid_rolls_back(tmp_path):
    _write_run(tmp_path)
    _build_extent_gen(tmp_path, 0, fill=1.0, parent=-1, stage=-1)
    _build_extent_gen(tmp_path, 1, fill=9.0, parent=0, stage=1)
    (gen_dir(tmp_path, 0, 1) / EXTENTS_DIRNAME / extent_filename(0)).unlink()
    res = RecoveryScanner(tmp_path).scan()
    assert res.generation == 0        # rolls back to the valid extent gen


# ── chunks mode unchanged (regression) ──────────────────────────────────

def test_chunks_mode_still_validates(tmp_path):
    _write_run(tmp_path)
    gdir = gen_dir(tmp_path, 0, 0)
    cdir = gdir / "chunks"; cdir.mkdir(parents=True)
    recs = []
    from wenbo_engine.recovery.generation_manager import sha256_file
    for ci in range(NC):
        p = cdir / chunk_filename(ci)
        write_chunk_atomic(p, np.full(CS, ci, dtype=DTYPE))
        recs.append(ChunkRecord(index=ci, filename=chunk_filename(ci),
                                size_bytes=p.stat().st_size,
                                checksum=sha256_file(p)))   # no extent fields
    rm = RankManifest(rank=0, generation=0, n_chunks=NC, chunk_size=CS,
                      dtype="complex64", circuit_hash=CH, parent_generation=-1,
                      stage_id=-1, chunks=recs, created=1.0)
    rm.write_atomic(gdir)
    GlobalCommitRecord(generation=0, n_ranks=1, circuit_hash=CH, step_index=-1,
                       parent_generation=-1, rank_manifest_hashes={0: rm.manifest_hash},
                       created=1.0).write_atomic(commits_dir(tmp_path))
    assert all(not c.is_extent for c in rm.chunks)
    assert RecoveryScanner(tmp_path).scan(check_checksums=True).recovered


# ── durable promote/restore of an extent-backed generation ──────────────

def test_durable_promote_restore_extents(tmp_path):
    work = tmp_path / "work"
    _write_run(work)
    _build_extent_gen(work, 0, fill=1.0, parent=-1, stage=-1)
    coord = LocalCoordinator()
    backend = LocalPathBackend(tmp_path / "durable")
    cm = DurableCheckpointManager(work, "extent_run", backend, coord)
    cm.upload_run_metadata()
    assert cm.promote(0) is not None

    # capture original logical chunk bytes
    from wenbo_engine.storage.extent_store import materialize_to_chunk_files
    man = RankManifest.read(gen_dir(work, 0, 0))
    materialize_to_chunk_files(gen_dir(work, 0, 0), man.chunks, force=True)
    orig = [read_chunk(gen_chunks_dir(work, 0, 0) / chunk_filename(ci)).copy()
            for ci in range(NC)]

    shutil.rmtree(work)
    rm = DurableRestoreManager(work, "extent_run", backend, coord)
    assert rm.restore_latest(check_checksums=True).generation == 0
    # restored generation is extent-backed and recovers
    g0 = gen_dir(work, 0, 0)
    assert (g0 / EXTENTS_DIRNAME).exists()
    assert RecoveryScanner(work).scan(check_checksums=True).recovered
    # restored logical chunks are byte-identical
    materialize_to_chunk_files(g0, RankManifest.read(g0).chunks, force=True)
    for ci in range(NC):
        got = read_chunk(gen_chunks_dir(work, 0, 0) / chunk_filename(ci))
        assert np.array_equal(got, orig[ci])


# ── real-MPI: generation recovery + gate-aware + extents ────────────────

pytest.importorskip("mpi4py")
_HAVE = shutil.which("mpirun") is not None
_mark = pytest.mark.skipif(not _HAVE, reason="mpirun not available")


def _bench(layout, tmp, *, kind="mpi_nonlocal_heavy", n=6, depth=6,
           exch="gate_aware", verify=False):
    out = Path(tmp) / f"out_{layout}"
    wd = Path(tmp) / f"wd_{layout}"
    for p in (out, wd):
        shutil.rmtree(p, ignore_errors=True)
    cmd = ["mpirun", "-np", "2", sys.executable, "-m",
           "wenbo_engine.bench.communication_workloads", "--kind", kind,
           "--n", str(n), "--depth", str(depth), "--recovery", "generation",
           "--mpi-exchange-mode", exch, "--storage-layout", layout,
           "--output-dir", str(out), "--work-dir", str(wd)]
    if verify:
        cmd.append("--verify")
    r = subprocess.run(cmd, env=dict(os.environ, PYTHONPATH=REPO),
                       capture_output=True, timeout=400)
    assert r.returncode == 0, r.stderr.decode()[-1500:]
    s = json.load(open(glob.glob(str(out / "**" / "final_summary.json"),
                                 recursive=True)[0]))
    return s, wd


@_mark
def test_mpi_generation_recovery_with_extents(tmp_path):
    s, wd = _bench("extents", tmp_path)
    assert abs(s["final_norm"] - 1.0) < 1e-5
    assert s["recovery_mode"] == "generation"
    assert s["measured_mpi_nonlocal_ops"] > 0           # gate-aware MPI kept
    # committed generations are extent-backed: extent files exist, and the
    # final committed gen has no per-chunk files
    assert glob.glob(str(wd / "**" / EXTENTS_DIRNAME / "extent_*.dat"),
                     recursive=True)
    assert not glob.glob(str(wd / "**" / "wal.json"), recursive=True)


@_mark
def test_extents_match_chunks_final_state(tmp_path):
    chunks, _ = _bench("chunks", tmp_path, verify=True)
    extents, _ = _bench("extents", tmp_path, verify=True)
    assert chunks["correct"] is True and extents["correct"] is True
    assert abs(chunks["final_norm"] - extents["final_norm"]) < 1e-9


@_mark
def test_extents_reduce_file_count(tmp_path):
    _, wd_c = _bench("chunks", tmp_path)
    _, wd_e = _bench("extents", tmp_path)
    # count committed-generation payload files (chunk .bin vs extent .dat)
    c_files = glob.glob(str(wd_c / "**" / "chunk_*.bin"), recursive=True)
    e_files = glob.glob(str(wd_e / "**" / "extent_*.dat"), recursive=True)
    assert len(e_files) < len(c_files)     # extents = fewer physical files
