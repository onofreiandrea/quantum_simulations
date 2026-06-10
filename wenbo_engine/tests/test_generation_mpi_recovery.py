"""Distributed (node-local, multi-node) generation recovery.

These tests prove the fix for the real-EC2 finding: on true multi-node each
rank can read ONLY its own ``rank_<r>/`` subtree (node-local NVMe), so the
shared-FS scanner wrongly rejected every generation.  Here we *simulate*
node-local behavior on one machine by giving each rank a SEPARATE work_dir
root (rank r's files live only in ``roots[r]``; the global commit records +
run.json live only in the coordinator's ``roots[0]``), and we drive the real
:class:`DistributedRecoveryScanner` across ranks with a thread-backed
coordinator that implements true gather/broadcast/barrier semantics.

Covers the 8 required unit cases; an MPI variant (mpirun) is in the smoke runs.
"""
import threading

import numpy as np

from wenbo_engine.recovery import (
    RankManifest, ChunkRecord, GlobalCommitRecord,
    LocalCoordinator, commits_dir, gen_dir, gen_chunks_dir, quarantine_dir,
)
from wenbo_engine.recovery.generation_manager import (
    RunMetadata, write_json_atomic, run_json_path, Coordinator,
)
from wenbo_engine.recovery.recovery_scanner import DistributedRecoveryScanner
from wenbo_engine.storage.block_store import write_chunk_atomic, chunk_filename

CIRCUIT_HASH = "d15ea5e0c0ffee01"
CHUNK_SIZE = 4
N_CHUNKS = 2


# ── thread-backed coordinator (true collective semantics, in-process) ────

class _Bus:
    def __init__(self, n):
        self.n = n
        self.barrier = threading.Barrier(n)
        self.slots = [None] * n
        self.val = None


class ThreadCoordinator(Coordinator):
    """Coordinator whose gather/broadcast/barrier run across N threads."""

    def __init__(self, rank: int, bus: _Bus):
        self.rank = rank
        self.n_ranks = bus.n
        self._bus = bus

    def gather(self, obj):
        self._bus.slots[self.rank] = obj
        self._bus.barrier.wait()
        out = list(self._bus.slots) if self.rank == 0 else None
        self._bus.barrier.wait()
        return out

    def broadcast(self, obj):
        if self.rank == 0:
            self._bus.val = obj
        self._bus.barrier.wait()
        v = self._bus.val
        self._bus.barrier.wait()
        return v

    def barrier(self) -> None:
        self._bus.barrier.wait()


def run_distributed_scan(roots, n_ranks, *, check_checksums=False,
                         quarantine=False):
    """Run DistributedRecoveryScanner across n_ranks threads (node-local roots).

    Returns the list of recovered generation ids, one per rank.
    """
    bus = _Bus(n_ranks)
    results = [None] * n_ranks
    errors = [None] * n_ranks

    def worker(r):
        try:
            sc = DistributedRecoveryScanner(roots[r], ThreadCoordinator(r, bus))
            res = sc.find_latest_valid_generation(
                check_checksums=check_checksums, quarantine=quarantine)
            results[r] = res.generation
        except Exception as e:  # surface per-rank failure
            errors[r] = e

    threads = [threading.Thread(target=worker, args=(r,)) for r in range(n_ranks)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    for e in errors:
        if e is not None:
            raise e
    return results


# ── per-rank-root layout builders (simulate node-local NVMe) ─────────────

def _write_rank_gen(root, rank, generation, *, fill, parent, stage,
                    checksum=False, chunk_override=None):
    from wenbo_engine.recovery.generation_manager import sha256_file
    cdir = gen_chunks_dir(root, rank, generation)
    cdir.mkdir(parents=True, exist_ok=True)
    chunks = []
    for ci in range(N_CHUNKS):
        data = np.full(CHUNK_SIZE, fill + ci + rank * 100, dtype=np.complex64)
        path = cdir / chunk_filename(ci)
        write_chunk_atomic(path, data)
        size = path.stat().st_size
        if chunk_override and ci == 0:
            size = chunk_override  # record a wrong size to force a mismatch
        chunks.append(ChunkRecord(
            index=ci, filename=chunk_filename(ci), size_bytes=size,
            checksum=sha256_file(path) if checksum else None))
    man = RankManifest(
        rank=rank, generation=generation, parent_generation=parent,
        stage_id=stage, n_chunks=len(chunks), chunk_size=CHUNK_SIZE,
        dtype="complex64", circuit_hash=CIRCUIT_HASH, chunks=chunks, created=1.0)
    man.write_atomic(gen_dir(root, rank, generation))
    return man


def _make_generation(roots, generation, n_ranks, *, fill, checksum=False,
                     write_commit=True, skip_rank_manifest=None,
                     rank_overrides=None, commit_hash_overrides=None):
    """Write generation across per-rank roots; commit record into roots[0].

    skip_rank_manifest: a rank whose on-disk manifest is omitted (node-local
        torn upload).  commit_hash_overrides: rank->hash to record a wrong hash.
    """
    parent, stage = generation - 1, generation
    rank_overrides = rank_overrides or {}
    hashes = {}
    for rank in range(n_ranks):
        ov = dict(rank_overrides.get(rank, {}))
        p = ov.pop("parent", parent)
        s = ov.pop("stage", stage)
        man = _write_rank_gen(roots[rank], rank, generation, fill=fill,
                              parent=p, stage=s, checksum=checksum, **ov)
        hashes[rank] = man.manifest_hash
        if skip_rank_manifest == rank:
            # remove the manifest after the fact (chunks remain) -> torn
            (gen_dir(roots[rank], rank, generation) / "manifest.json").unlink()
    if commit_hash_overrides:
        hashes.update(commit_hash_overrides)
    if write_commit:
        rec = GlobalCommitRecord(
            generation=generation, n_ranks=n_ranks, circuit_hash=CIRCUIT_HASH,
            step_index=stage, parent_generation=parent,
            rank_manifest_hashes=hashes, created=1.0)
        rec.write_atomic(commits_dir(roots[0]))
        return rec
    return None


def _roots(tmp_path, n_ranks):
    roots = [tmp_path / f"node{r}_work" for r in range(n_ranks)]
    for r, root in enumerate(roots):
        root.mkdir(parents=True, exist_ok=True)
    # run.json is coordinator-visible only (node 0)
    write_json_atomic(run_json_path(roots[0]), RunMetadata(
        circuit_hash=CIRCUIT_HASH, n_ranks=n_ranks, n_qubits=4,
        chunk_size=CHUNK_SIZE, created=1.0).to_dict())
    return roots


# ── 1. rank-local validation succeeds ───────────────────────────────────

def test_rank_local_validation_succeeds(tmp_path):
    roots = _roots(tmp_path, 2)
    _make_generation(roots, 1, 2, fill=1.0)
    res = run_distributed_scan(roots, 2)
    assert res == [1, 1]                 # both ranks recover gen 1


# ── 2. one rank missing local manifest → rejection ──────────────────────

def test_missing_local_manifest_rejects(tmp_path):
    roots = _roots(tmp_path, 2)
    _make_generation(roots, 1, 2, fill=1.0, skip_rank_manifest=1)
    res = run_distributed_scan(roots, 2)
    assert res == [None, None]           # rank 1 can't validate -> all reject


# ── 3. one rank wrong chunk size → rejection ────────────────────────────

def test_wrong_chunk_size_rejects(tmp_path):
    roots = _roots(tmp_path, 2)
    _make_generation(roots, 1, 2, fill=1.0,
                     rank_overrides={1: {"chunk_override": 999}})
    res = run_distributed_scan(roots, 2)
    assert res == [None, None]


# ── 4. one rank manifest-hash mismatch → rejection ──────────────────────

def test_manifest_hash_mismatch_rejects(tmp_path):
    roots = _roots(tmp_path, 2)
    # commit records a bogus hash for rank 1; rank 1's real manifest won't match
    _make_generation(roots, 1, 2, fill=1.0,
                     commit_hash_overrides={1: "0" * 32})
    res = run_distributed_scan(roots, 2)
    assert res == [None, None]


# ── 5. one rank wrong parent_generation → rejection ─────────────────────

def test_wrong_parent_generation_rejects(tmp_path):
    roots = _roots(tmp_path, 2)
    # rank 1's manifest claims a different parent than the commit records
    _make_generation(roots, 2, 2, fill=2.0,
                     rank_overrides={1: {"parent": 0}})  # commit parent=1
    res = run_distributed_scan(roots, 2)
    assert res == [None, None]


# ── 6. rank0@gen5 + rank1@gen4 mixed state is rejected ──────────────────

def test_mixed_generation_state_rejected(tmp_path):
    roots = _roots(tmp_path, 2)
    # gen 4 fully valid on both ranks (fallback target)
    _make_generation(roots, 4, 2, fill=4.0)
    # gen 5 commit names both ranks, but rank 1 never wrote gen 5 (only gen 4).
    _make_generation(roots, 5, 2, fill=5.0, skip_rank_manifest=1)
    res = run_distributed_scan(roots, 2)
    # must NOT recover a mixed gen5(rank0)/gen4(rank1); falls back to gen 4 on all
    assert res == [4, 4]


# ── 7. newest invalid skipped, previous valid selected ──────────────────

def test_newest_invalid_skipped_previous_selected(tmp_path):
    roots = _roots(tmp_path, 2)
    _make_generation(roots, 1, 2, fill=1.0)                      # valid
    _make_generation(roots, 2, 2, fill=2.0, skip_rank_manifest=0)  # invalid
    res = run_distributed_scan(roots, 2)
    assert res == [1, 1]


# ── 8. all ranks return the same generation (incl. 4 ranks) ─────────────

def test_all_ranks_same_generation_4_ranks(tmp_path):
    roots = _roots(tmp_path, 4)
    _make_generation(roots, 3, 4, fill=3.0)
    res = run_distributed_scan(roots, 4)
    assert res == [3, 3, 3, 3]
    assert len(set(res)) == 1


def test_no_commit_record_fresh_start(tmp_path):
    roots = _roots(tmp_path, 2)
    _make_generation(roots, 1, 2, fill=1.0, write_commit=False)  # no commit
    res = run_distributed_scan(roots, 2)
    assert res == [None, None]           # no commit ⇒ no committed generation


def test_checksum_mismatch_rejects(tmp_path):
    roots = _roots(tmp_path, 2)
    rec = _make_generation(roots, 1, 2, fill=1.0, checksum=True)
    # corrupt rank 1's chunk bytes in place (same size) -> only checksum catches
    victim = gen_chunks_dir(roots[1], 1, 1) / chunk_filename(0)
    orig = victim.read_bytes()
    victim.write_bytes(bytes((b ^ 0xFF) for b in orig))
    assert run_distributed_scan(roots, 2, check_checksums=True) == [None, None]
    # without checksum validation the same-size corruption is not detected
    assert run_distributed_scan(roots, 2, check_checksums=False) == [1, 1]
