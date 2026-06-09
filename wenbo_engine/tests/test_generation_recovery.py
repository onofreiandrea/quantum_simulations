"""Distributed generation-based recovery scenarios.

Required cases:
  1. valid generation commit
  2. missing global commit means rollback
  3. missing chunk means rollback
  4. wrong chunk size means rollback
  5. corrupted manifest hash means rollback
  6. crash after manifest before global commit recovers previous generation
  7. crash after global commit recovers new generation

Plus: no commits → fresh start, quarantine of incomplete generations,
checksum mismatch, and a manager-driven end-to-end commit/scan round-trip.
"""
import json

import numpy as np
import pytest

from wenbo_engine.storage.block_store import write_chunk_atomic, chunk_filename
from wenbo_engine.recovery import (
    RankManifest, GlobalCommitRecord, RecoveryScanner, GenerationManager,
    LocalCoordinator, EventType, commits_dir, gen_dir, gen_chunks_dir,
    quarantine_dir, read_run_metadata, commit_filename,
)
from wenbo_engine.recovery.generation_manager import (
    RunMetadata, write_json_atomic, run_json_path, Coordinator, _RankStatus,
)

CIRCUIT_HASH = "deadbeefcafe0001"
CHUNK_SIZE = 4              # 4 complex64 amplitudes
N_CHUNKS_PER_RANK = 2
CHUNK_BYTES = CHUNK_SIZE * 8


# ── layout builders (drive the real write path for N ranks) ────────────

def _write_run(work_dir, n_ranks):
    meta = RunMetadata(circuit_hash=CIRCUIT_HASH, n_ranks=n_ranks,
                       n_qubits=4, chunk_size=CHUNK_SIZE, created=1.0)
    write_json_atomic(run_json_path(work_dir), meta.to_dict())


def _lineage(generation):
    """Canonical (parent_generation, stage_id) for a generation in tests."""
    return generation - 1, generation


def _write_rank_generation(work_dir, rank, generation, *, fill, checksum=False,
                           parent_generation=None, stage_id=None):
    """Write one rank's chunks + sealed manifest for a generation.

    Returns the sealed RankManifest (so a caller can build the commit record).
    """
    from wenbo_engine.recovery.generation_manager import sha256_file
    parent, stage = _lineage(generation)
    if parent_generation is not None:
        parent = parent_generation
    if stage_id is not None:
        stage = stage_id
    cdir = gen_chunks_dir(work_dir, rank, generation)
    cdir.mkdir(parents=True, exist_ok=True)
    chunks = []
    for ci in range(N_CHUNKS_PER_RANK):
        data = np.full(CHUNK_SIZE, fill + ci + rank * 100, dtype=np.complex64)
        path = cdir / chunk_filename(ci)
        write_chunk_atomic(path, data)
        from wenbo_engine.recovery import ChunkRecord
        chunks.append(ChunkRecord(
            index=ci, filename=chunk_filename(ci),
            size_bytes=path.stat().st_size,
            checksum=sha256_file(path) if checksum else None,
        ))
    man = RankManifest(
        rank=rank, generation=generation,
        parent_generation=parent, stage_id=stage,
        n_chunks=len(chunks),
        chunk_size=CHUNK_SIZE, dtype="complex64", circuit_hash=CIRCUIT_HASH,
        chunks=chunks, created=1.0,
    )
    man.write_atomic(gen_dir(work_dir, rank, generation))
    return man


def _commit_generation(work_dir, generation, n_ranks, *, fill,
                       checksum=False, write_commit=True,
                       rank_overrides=None):
    """Write all ranks' generation; optionally write the global commit record.

    ``rank_overrides`` maps rank -> kwargs forwarded to _write_rank_generation
    (e.g. to make one rank diverge for negative tests).
    """
    parent, stage = _lineage(generation)
    rank_overrides = rank_overrides or {}
    hashes = {}
    for rank in range(n_ranks):
        man = _write_rank_generation(work_dir, rank, generation,
                                     fill=fill, checksum=checksum,
                                     **rank_overrides.get(rank, {}))
        hashes[rank] = man.manifest_hash
    if write_commit:
        rec = GlobalCommitRecord(
            generation=generation, n_ranks=n_ranks, circuit_hash=CIRCUIT_HASH,
            step_index=stage, parent_generation=parent,
            rank_manifest_hashes=hashes, created=1.0,
        )
        rec.write_atomic(commits_dir(work_dir))
        return rec
    return None


# ── 1. valid generation commit ─────────────────────────────────────────

def test_valid_generation_commit(tmp_path):
    _write_run(tmp_path, n_ranks=2)
    _commit_generation(tmp_path, 0, 2, fill=0)
    _commit_generation(tmp_path, 1, 2, fill=10)

    res = RecoveryScanner(tmp_path).scan()
    assert res.recovered
    assert res.generation == 1
    assert res.record.circuit_hash == CIRCUIT_HASH


# ── 2. missing global commit means rollback ────────────────────────────

def test_missing_global_commit_rolls_back(tmp_path):
    _write_run(tmp_path, n_ranks=2)
    _commit_generation(tmp_path, 0, 2, fill=0)
    # Generation 1 fully written on disk, but NO commit record.
    _commit_generation(tmp_path, 1, 2, fill=10, write_commit=False)

    res = RecoveryScanner(tmp_path).scan()
    assert res.generation == 0          # rolled back to last committed gen
    # gen 1 (uncommitted) must be quarantined for every rank
    assert len(res.quarantined) == 2
    assert not gen_dir(tmp_path, 0, 1).exists()
    assert (quarantine_dir(tmp_path, 0) / "gen_000001").exists()


def test_no_commits_means_fresh_start(tmp_path):
    _write_run(tmp_path, n_ranks=2)
    # Only an uncommitted generation 0 exists.
    _commit_generation(tmp_path, 0, 2, fill=0, write_commit=False)

    res = RecoveryScanner(tmp_path).scan()
    assert res.generation is None       # no committed generation at all
    assert res.events.has(EventType.NO_COMMITS)
    # the uncommitted generation is quarantined
    assert (quarantine_dir(tmp_path, 0) / "gen_000000").exists()


def test_newest_invalid_selects_previous_valid(tmp_path):
    """Three committed generations; newest is corrupt → pick the one before it.

    Proves the scanner walks back to the *immediately preceding* valid
    generation, not merely the oldest one.
    """
    _write_run(tmp_path, n_ranks=2)
    _commit_generation(tmp_path, 0, 2, fill=0)
    _commit_generation(tmp_path, 1, 2, fill=10)
    _commit_generation(tmp_path, 2, 2, fill=20)

    # Corrupt generation 2 (the newest committed): drop a chunk.
    (gen_chunks_dir(tmp_path, 1, 2) / chunk_filename(0)).unlink()

    res = RecoveryScanner(tmp_path).scan(quarantine=False)
    assert res.generation == 1          # previous valid, not 0
    assert res.events.has(EventType.ROLLBACK)


def test_manifests_alone_do_not_commit(tmp_path):
    """Commit invariant: perfect manifests + chunks but no commit record → None.

    Manifests and chunk files alone must never make recovery accept a gen.
    """
    _write_run(tmp_path, n_ranks=2)
    # Fully valid generation 0 on disk for every rank — but NO commits/ dir.
    _commit_generation(tmp_path, 0, 2, fill=0, write_commit=False)
    assert not (commits_dir(tmp_path)).exists() or not list(
        commits_dir(tmp_path).glob("commit_*.json"))

    res = RecoveryScanner(tmp_path).scan(quarantine=False)
    assert res.generation is None
    assert res.events.has(EventType.NO_COMMITS)


# ── 3. missing chunk means rollback ────────────────────────────────────

def test_missing_chunk_rolls_back(tmp_path):
    _write_run(tmp_path, n_ranks=2)
    _commit_generation(tmp_path, 0, 2, fill=0)
    _commit_generation(tmp_path, 1, 2, fill=10)

    # Delete one chunk file from generation 1, rank 1.
    (gen_chunks_dir(tmp_path, 1, 1) / chunk_filename(0)).unlink()

    res = RecoveryScanner(tmp_path).scan(quarantine=False)
    assert res.generation == 0
    assert res.events.has(EventType.CHUNK_MISSING)


# ── 4. wrong chunk size means rollback ─────────────────────────────────

def test_wrong_chunk_size_rolls_back(tmp_path):
    _write_run(tmp_path, n_ranks=2)
    _commit_generation(tmp_path, 0, 2, fill=0)
    _commit_generation(tmp_path, 1, 2, fill=10)

    # Truncate a chunk in generation 1.
    bad = gen_chunks_dir(tmp_path, 0, 1) / chunk_filename(1)
    with open(bad, "r+b") as f:
        f.truncate(CHUNK_BYTES // 2)

    res = RecoveryScanner(tmp_path).scan(quarantine=False)
    assert res.generation == 0
    assert res.events.has(EventType.CHUNK_SIZE_MISMATCH)


# ── 5. corrupted manifest hash means rollback ──────────────────────────

def test_corrupted_manifest_hash_rolls_back(tmp_path):
    _write_run(tmp_path, n_ranks=2)
    _commit_generation(tmp_path, 0, 2, fill=0)
    _commit_generation(tmp_path, 1, 2, fill=10)

    # Tamper the on-disk manifest content WITHOUT updating its hash field:
    # recomputed hash no longer matches the stored (and commit-recorded) hash.
    mpath = gen_dir(tmp_path, 1, 1) / "manifest.json"
    data = json.loads(mpath.read_text())
    data["chunks"][0]["size_bytes"] = 999999     # content changed
    mpath.write_text(json.dumps(data))

    res = RecoveryScanner(tmp_path).scan(quarantine=False)
    assert res.generation == 0
    assert res.events.has(EventType.MANIFEST_HASH_MISMATCH)


def test_manifest_missing_rolls_back(tmp_path):
    _write_run(tmp_path, n_ranks=2)
    _commit_generation(tmp_path, 0, 2, fill=0)
    _commit_generation(tmp_path, 1, 2, fill=10)

    (gen_dir(tmp_path, 0, 1) / "manifest.json").unlink()

    res = RecoveryScanner(tmp_path).scan(quarantine=False)
    assert res.generation == 0
    assert res.events.has(EventType.MANIFEST_MISSING)


# ── 6. crash after manifest, before global commit ──────────────────────

def test_crash_after_manifest_before_commit(tmp_path):
    """Rank prepared g+1 (manifest written) but coordinator never committed."""
    coord = LocalCoordinator()
    gm = GenerationManager(tmp_path, coord, circuit_hash=CIRCUIT_HASH,
                           chunk_size=CHUNK_SIZE)
    gm.init_run(n_qubits=4)

    def writer(fill):
        def w(cdir):
            recs = []
            for ci in range(N_CHUNKS_PER_RANK):
                data = np.full(CHUNK_SIZE, fill + ci, dtype=np.complex64)
                write_chunk_atomic(cdir / chunk_filename(ci), data)
                recs.append(gm.chunk_record(cdir, ci))
            return recs
        return w

    # Commit generation 0 fully.
    assert gm.commit_step(0, 0, writer(0)) is not None
    # Prepare generation 1 (steps 1–7) but DO NOT commit (crash at step ~8).
    gm.prepare(1, writer(10))
    assert (gen_dir(tmp_path, 0, 1) / "manifest.json").exists()
    assert not (commits_dir(tmp_path) / "commit_000001.json").exists()

    res = RecoveryScanner(tmp_path).scan()
    assert res.generation == 0          # previous generation recovered
    assert (quarantine_dir(tmp_path, 0) / "gen_000001").exists()


# ── 7. crash after global commit ───────────────────────────────────────

def test_crash_after_global_commit(tmp_path):
    """Commit record for g+1 is on disk → recovery picks up the new gen."""
    coord = LocalCoordinator()
    gm = GenerationManager(tmp_path, coord, circuit_hash=CIRCUIT_HASH,
                           chunk_size=CHUNK_SIZE)
    gm.init_run(n_qubits=4)

    def writer(fill):
        def w(cdir):
            recs = []
            for ci in range(N_CHUNKS_PER_RANK):
                data = np.full(CHUNK_SIZE, fill + ci, dtype=np.complex64)
                write_chunk_atomic(cdir / chunk_filename(ci), data)
                recs.append(gm.chunk_record(cdir, ci))
            return recs
        return w

    assert gm.commit_step(0, 0, writer(0)) is not None
    assert gm.commit_step(1, 1, writer(10)) is not None   # crash right after

    res = RecoveryScanner(tmp_path).scan()
    assert res.generation == 1          # newest committed generation


# ── checksum verification (opt-in) ─────────────────────────────────────

def test_checksum_mismatch_rolls_back(tmp_path):
    _write_run(tmp_path, n_ranks=1)
    _commit_generation(tmp_path, 0, 1, fill=0, checksum=True)
    _commit_generation(tmp_path, 1, 1, fill=10, checksum=True)

    # Corrupt bytes in a gen-1 chunk WITHOUT changing its size.
    bad = gen_chunks_dir(tmp_path, 0, 1) / chunk_filename(0)
    raw = bytearray(bad.read_bytes())
    raw[0] ^= 0xFF
    bad.write_bytes(bytes(raw))

    # Size check passes; checksum check catches the corruption.
    res_sizes = RecoveryScanner(tmp_path).scan(quarantine=False,
                                               check_checksums=False)
    assert res_sizes.generation == 1   # size-only scan can't see it
    res_csum = RecoveryScanner(tmp_path).scan(quarantine=False,
                                              check_checksums=True)
    assert res_csum.generation == 0
    assert res_csum.events.has(EventType.CHUNK_CHECKSUM_MISMATCH)


# ── corrupted commit file itself ───────────────────────────────────────

def test_corrupted_commit_file_rolls_back(tmp_path):
    _write_run(tmp_path, n_ranks=1)
    _commit_generation(tmp_path, 0, 1, fill=0)
    _commit_generation(tmp_path, 1, 1, fill=10)

    cpath = commits_dir(tmp_path) / "commit_000001.json"
    data = json.loads(cpath.read_text())
    data["step_index"] = 999            # break self-hash
    cpath.write_text(json.dumps(data))

    res = RecoveryScanner(tmp_path).scan(quarantine=False)
    assert res.generation == 0
    assert res.events.has(EventType.COMMIT_INVALID)


# ── end-to-end via manager + run.json metadata ─────────────────────────

def test_manager_e2e_and_run_metadata(tmp_path):
    coord = LocalCoordinator()
    gm = GenerationManager(tmp_path, coord, circuit_hash=CIRCUIT_HASH,
                           chunk_size=CHUNK_SIZE)
    gm.init_run(n_qubits=4, plan={"steps": 3}, cost_model={"flops": 1})

    meta = read_run_metadata(tmp_path)
    assert meta is not None and meta.n_ranks == 1
    assert meta.recovery_mode == "generation"
    assert (tmp_path / "plan.json").exists()
    assert (tmp_path / "cost_model.json").exists()

    def writer(fill):
        def w(cdir):
            recs = []
            for ci in range(N_CHUNKS_PER_RANK):
                data = np.full(CHUNK_SIZE, fill, dtype=np.complex64)
                write_chunk_atomic(cdir / chunk_filename(ci), data)
                recs.append(gm.chunk_record(cdir, ci, checksum=True))
            return recs
        return w

    for g in range(3):
        assert gm.commit_step(g, g, writer(g)) is not None

    assert gm.resume_generation(check_checksums=True) == 2


# ════════════════════════════════════════════════════════════════════════
# Strict proof-check additions: multi-rank write-side + read-side invariants
# ════════════════════════════════════════════════════════════════════════

class _SimCoord(Coordinator):
    """Single-process stand-in for one rank of an N-rank communicator.

    prepare() only needs ``rank``/``is_coordinator``; commit() collectives are
    driven explicitly by the test harness, so gather/broadcast are no-ops.
    """
    def __init__(self, rank, n_ranks):
        self.rank = rank
        self.n_ranks = n_ranks

    def gather(self, obj):
        return [obj] if self.is_coordinator else None

    def broadcast(self, obj):
        return obj

    def barrier(self):
        pass


def _mgr(work_dir, rank, n_ranks):
    return GenerationManager(work_dir, _SimCoord(rank, n_ranks),
                             circuit_hash=CIRCUIT_HASH, chunk_size=CHUNK_SIZE,
                             dtype="complex64")


def _prepare_all(work_dir, generation, n_ranks, *, fill, parent, stage,
                 overrides=None):
    """Drive real prepare() for every rank; return list of _RankStatus."""
    overrides = overrides or {}
    statuses = []
    for r in range(n_ranks):
        ov = overrides.get(r, {})
        gm = _mgr(work_dir, r, n_ranks)
        if ov.get("skip_manifest"):
            # Rank wrote chunks but never produced a manifest (mid-crash).
            cdir = gen_chunks_dir(work_dir, r, generation)
            cdir.mkdir(parents=True, exist_ok=True)
            for ci in range(N_CHUNKS_PER_RANK):
                write_chunk_atomic(cdir / chunk_filename(ci),
                                   np.full(CHUNK_SIZE, r, dtype=np.complex64))
            statuses.append(_RankStatus(
                rank=r, generation=generation, prepared=False,
                manifest_hash="", parent_generation=parent, stage_id=stage))
            continue

        def writer(cdir, rr=r):
            recs = []
            for ci in range(N_CHUNKS_PER_RANK):
                write_chunk_atomic(cdir / chunk_filename(ci),
                                   np.full(CHUNK_SIZE, fill + rr + ci,
                                           dtype=np.complex64))
                recs.append(gm.chunk_record(cdir, ci))
            return recs

        p = ov.get("parent", parent)
        s = ov.get("stage", stage)
        man = gm.prepare(generation, writer, parent_generation=p, stage_id=s)
        statuses.append(_RankStatus(
            rank=r, generation=generation, prepared=True,
            manifest_hash=man.manifest_hash, parent_generation=p, stage_id=s))
    return statuses


def _drive_commit(work_dir, generation, n_ranks, *, fill, overrides=None):
    """Real prepare on all ranks + real coordinator commit. Returns record|None."""
    parent, stage = _lineage(generation)
    statuses = _prepare_all(work_dir, generation, n_ranks, fill=fill,
                            parent=parent, stage=stage, overrides=overrides)
    coord0 = _mgr(work_dir, 0, n_ranks)
    return coord0._coordinator_commit(generation, stage, parent, statuses)


# ── Check 3: partial-rank commit failure ───────────────────────────────

def test_partial_rank_no_manifest_aborts_commit(tmp_path):
    """Rank 0 valid; rank 1 never writes a manifest → NO global commit."""
    _write_run(tmp_path, n_ranks=2)
    assert _drive_commit(tmp_path, 0, 2, fill=0) is not None      # gen 0 commits
    rec = _drive_commit(tmp_path, 1, 2, fill=10,
                        overrides={1: {"skip_manifest": True}})
    assert rec is None                                            # aborted
    assert not (commits_dir(tmp_path) / commit_filename(1)).exists()
    # Recovery rolls back to the last committed generation.
    assert RecoveryScanner(tmp_path).scan(quarantine=False).generation == 0


def test_partial_rank_missing_chunk_rejected_on_read(tmp_path):
    """Commit for gen 1 exists, but rank 1 is missing a chunk → reject, pick 0."""
    _write_run(tmp_path, n_ranks=2)
    _commit_generation(tmp_path, 0, 2, fill=0)
    _commit_generation(tmp_path, 1, 2, fill=10)
    (gen_chunks_dir(tmp_path, 1, 1) / chunk_filename(0)).unlink()

    res = RecoveryScanner(tmp_path).scan(quarantine=False)
    assert res.generation == 0
    assert res.events.has(EventType.CHUNK_MISSING)


# ── Check 4: a global commit record cannot lie ─────────────────────────

def test_commit_lies_about_manifest_hash_rejected(tmp_path):
    """Commit names a rank manifest hash that doesn't match the real manifest."""
    _write_run(tmp_path, n_ranks=2)
    _commit_generation(tmp_path, 0, 2, fill=0)
    rec = _commit_generation(tmp_path, 1, 2, fill=10)

    # Rewrite the gen-1 commit with a wrong hash for rank 1, re-sealed so the
    # commit's OWN self-hash is valid (only the named manifest hash is a lie).
    rec.rank_manifest_hashes[1] = "0" * 32
    rec.commit_hash = ""
    rec.write_atomic(commits_dir(tmp_path))

    res = RecoveryScanner(tmp_path).scan(quarantine=False)
    assert res.generation == 0
    assert res.events.has(EventType.MANIFEST_HASH_MISMATCH)


def test_commit_with_divergent_parent_generation_rejected(tmp_path):
    """One rank's manifest has a different parent_generation → reject gen."""
    _write_run(tmp_path, n_ranks=2)
    _commit_generation(tmp_path, 0, 2, fill=0)
    # rank 1 prepared with a bogus parent; commit records its real hash.
    _commit_generation(tmp_path, 1, 2, fill=10,
                       rank_overrides={1: {"parent_generation": 999}})

    res = RecoveryScanner(tmp_path).scan(quarantine=False)
    assert res.generation == 0
    assert res.events.has(EventType.LINEAGE_MISMATCH)


def test_commit_with_divergent_stage_id_rejected(tmp_path):
    """One rank's manifest has a different stage_id → reject gen."""
    _write_run(tmp_path, n_ranks=2)
    _commit_generation(tmp_path, 0, 2, fill=0)
    _commit_generation(tmp_path, 1, 2, fill=10,
                       rank_overrides={1: {"stage_id": 777}})

    res = RecoveryScanner(tmp_path).scan(quarantine=False)
    assert res.generation == 0
    assert res.events.has(EventType.LINEAGE_MISMATCH)


def test_write_side_lineage_disagreement_aborts_commit(tmp_path):
    """If ranks prepare with disagreeing lineage, no commit is written."""
    _write_run(tmp_path, n_ranks=2)
    assert _drive_commit(tmp_path, 0, 2, fill=0) is not None
    rec = _drive_commit(tmp_path, 1, 2, fill=10,
                        overrides={1: {"parent": 999}})
    assert rec is None
    assert not (commits_dir(tmp_path) / commit_filename(1)).exists()


# ── Check 7: scanner must never select mixed-generation state ───────────

def test_mixed_generation_partitions_rejected(tmp_path):
    """rank 0 has gen 5, rank 1 only has gen 4; commit claims gen 5 → reject."""
    _write_run(tmp_path, n_ranks=2)
    _commit_generation(tmp_path, 4, 2, fill=0)

    # Write rank 0's gen-5 manifest/chunks only; rank 1 has NO gen 5.
    m0 = _write_rank_generation(tmp_path, 0, 5, fill=50)
    # Build a gen-5 commit naming both ranks (rank 1's hash is whatever; the
    # point is rank 1's gen-5 manifest does not exist on disk).
    parent, stage = _lineage(5)
    rec = GlobalCommitRecord(
        generation=5, n_ranks=2, circuit_hash=CIRCUIT_HASH, step_index=stage,
        parent_generation=parent,
        rank_manifest_hashes={0: m0.manifest_hash, 1: "f" * 32}, created=1.0,
    )
    rec.write_atomic(commits_dir(tmp_path))

    res = RecoveryScanner(tmp_path).scan(quarantine=False)
    assert res.generation == 4          # never a mix of rank0@5 + rank1@4
    assert res.events.has(EventType.MANIFEST_MISSING)


# ── Check 5: pruning safety ─────────────────────────────────────────────

def test_pruning_keeps_rollback_target(tmp_path):
    """gen0,1 valid, gen2 invalid → recovery still finds gen1 (kept by prune)."""
    _write_run(tmp_path, n_ranks=1)
    _commit_generation(tmp_path, 0, 1, fill=0)
    _commit_generation(tmp_path, 1, 1, fill=10)
    _commit_generation(tmp_path, 2, 1, fill=20)
    # Default prune keeps newest 3, so gen1 survives; corrupt gen2.
    (gen_chunks_dir(tmp_path, 0, 2) / chunk_filename(0)).unlink()

    res = RecoveryScanner(tmp_path).scan(quarantine=False)
    assert res.generation == 1


def test_pruning_removed_parent_falls_back_safely(tmp_path):
    """gen2 invalid AND gen1 pruned away → fall back to gen0 (explicit, safe)."""
    _write_run(tmp_path, n_ranks=1)
    _commit_generation(tmp_path, 0, 1, fill=0)
    _commit_generation(tmp_path, 1, 1, fill=10)
    _commit_generation(tmp_path, 2, 1, fill=20)
    # Simulate prune having removed gen1's directory; corrupt gen2.
    import shutil
    shutil.rmtree(gen_dir(tmp_path, 0, 1))
    (gen_chunks_dir(tmp_path, 0, 2) / chunk_filename(0)).unlink()

    res = RecoveryScanner(tmp_path).scan(quarantine=False)
    assert res.generation == 0          # not a mix, not a crash — older valid


def test_all_retained_invalid_is_explicit_fresh_start(tmp_path):
    """If every committed generation on disk is invalid → None (recompute)."""
    _write_run(tmp_path, n_ranks=1)
    _commit_generation(tmp_path, 0, 1, fill=0)
    _commit_generation(tmp_path, 1, 1, fill=10)
    # Corrupt both: remove a chunk from each.
    (gen_chunks_dir(tmp_path, 0, 0) / chunk_filename(0)).unlink()
    (gen_chunks_dir(tmp_path, 0, 1) / chunk_filename(0)).unlink()

    res = RecoveryScanner(tmp_path).scan(quarantine=False)
    assert res.generation is None       # explicit, safe: start fresh
    assert res.events.has(EventType.FRESH_START)


def test_prune_default_keeps_three(tmp_path):
    """Manager.prune default retains the newest three generations."""
    coord = LocalCoordinator()
    gm = GenerationManager(tmp_path, coord, circuit_hash=CIRCUIT_HASH,
                           chunk_size=CHUNK_SIZE)
    gm.init_run(n_qubits=4)

    def writer(cdir):
        recs = []
        for ci in range(N_CHUNKS_PER_RANK):
            write_chunk_atomic(cdir / chunk_filename(ci),
                               np.zeros(CHUNK_SIZE, dtype=np.complex64))
            recs.append(gm.chunk_record(cdir, ci))
        return recs

    for g in range(5):
        parent = g - 1
        gm.commit_step(g, g, writer, parent_generation=parent)
        gm.prune()                      # default keep=3
    remaining = sorted(int(d.name[4:]) for d in
                       (gen_dir(tmp_path, 0, 0).parent).glob("gen_*"))
    assert remaining == [2, 3, 4]       # newest three kept
