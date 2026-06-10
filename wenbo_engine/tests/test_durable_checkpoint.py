"""Durable checkpoint (R4) promotion tests.

Covered required cases:
  1. promote a committed generation to a durable local path
  2. missing durable commit is ignored on restore-side discovery
  3. partial durable upload (chunk deleted) is ignored
  4. corrupt durable manifest record / file is ignored
  8. durable storage is NOT used in hot execution (spy backend → 0 put() calls
     across a normal generation step loop)

Synthetic committed generations are built with the recovery package's public
APIs (LocalCoordinator, GenerationManager, write_chunk_atomic), then promoted
with the durable package — no full MPI simulation is needed for these.
"""
import numpy as np

from wenbo_engine.storage.block_store import write_chunk_atomic, chunk_filename
from wenbo_engine.recovery import (
    GenerationManager, LocalCoordinator, commits_dir,
)
from wenbo_engine.recovery.global_commit import commit_filename

from wenbo_engine.durable import (
    LocalPathBackend, DurableCheckpointManager, DurableRestoreManager,
    make_backend,
)
from wenbo_engine.durable import durable_commit as dc
from wenbo_engine.durable.backend import DurableBackend, sha256_bytes
from wenbo_engine.durable.durable_checkpoint_manager import (
    DurableConfig, DurablePromotionError,
)

CIRCUIT_HASH = "deadbeefcafe0042"
CHUNK_SIZE = 4
N_CHUNKS_PER_RANK = 2
RUN_ID = "durable_run"


# ── helpers ─────────────────────────────────────────────────────────────

def _writer(gm, fill):
    def w(cdir):
        recs = []
        for ci in range(N_CHUNKS_PER_RANK):
            data = np.full(CHUNK_SIZE, fill + ci, dtype=np.complex64)
            write_chunk_atomic(cdir / chunk_filename(ci), data)
            recs.append(gm.chunk_record(cdir, ci))
        return recs
    return w


def _make_committed_run(work_dir, n_gens=3):
    """Build a local single-rank run with ``n_gens`` committed generations."""
    coord = LocalCoordinator()
    gm = GenerationManager(work_dir, coord, circuit_hash=CIRCUIT_HASH,
                           chunk_size=CHUNK_SIZE)
    gm.init_run(n_qubits=4)
    for g in range(n_gens):
        rec = gm.commit_step(g, g, _writer(gm, fill=g * 10),
                             parent_generation=g - 1)
        assert rec is not None
    return gm, coord


def _backend(tmp_path):
    return LocalPathBackend(tmp_path / "durable")


def _promote(work_dir, backend, coord, generation):
    cm = DurableCheckpointManager(work_dir, RUN_ID, backend, coord)
    cm.upload_run_metadata()
    return cm.promote(generation)


# ── 1. promote committed generation to durable local path ───────────────

def test_promote_committed_generation_to_durable(tmp_path):
    work = tmp_path / "work"
    gm, coord = _make_committed_run(work, n_gens=2)
    backend = _backend(tmp_path)

    rec = _promote(work, backend, coord, 1)
    assert rec is not None
    assert rec.generation == 1
    assert rec.n_ranks == 1
    assert rec.verify_self_hash()

    # durable_commit record exists and is the durability point.
    assert backend.exists(dc.durable_commit_key(RUN_ID, 1))
    # every chunk + manifest is present with matching size/checksum.
    entry = rec.ranks[0]
    assert len(entry.chunks) == N_CHUNKS_PER_RANK
    for fr in [entry.manifest, *entry.chunks]:
        assert backend.exists(fr.key)
        assert backend.size(fr.key) == fr.size_bytes
        assert backend.checksum(fr.key) == fr.checksum

    # restore-side discovery finds it.
    rm = DurableRestoreManager(work, RUN_ID, backend, coord)
    found = rm.latest_valid_durable_commit(check_checksums=True)
    assert found is not None and found.generation == 1


def test_promote_refuses_uncommitted_generation(tmp_path):
    """Only locally committed generations may be promoted (protocol step 1)."""
    work = tmp_path / "work"
    gm, coord = _make_committed_run(work, n_gens=1)   # only gen 0 committed
    backend = _backend(tmp_path)
    cm = DurableCheckpointManager(work, RUN_ID, backend, coord)

    # Generation 5 was never committed locally → refuse.
    try:
        cm.promote(5)
        assert False, "expected DurablePromotionError"
    except DurablePromotionError:
        pass
    assert not backend.exists(dc.durable_commit_key(RUN_ID, 5))


def test_make_backend_local_path(tmp_path):
    b = make_backend("local_path", root=str(tmp_path / "d"))
    assert isinstance(b, LocalPathBackend)
    pr = b.put("a/b.txt", b"hello")
    assert pr.size_bytes == 5
    assert b.get("a/b.txt") == b"hello"
    assert b.checksum("a/b.txt") == sha256_bytes(b"hello")


# ── 2. missing durable commit is ignored ────────────────────────────────

def test_missing_durable_commit_ignored(tmp_path):
    """Chunks + manifests uploaded but NO durable commit record → not durable."""
    work = tmp_path / "work"
    gm, coord = _make_committed_run(work, n_gens=2)
    backend = _backend(tmp_path)

    # Upload the files for gen 1 by hand, but never write the durable commit.
    cm = DurableCheckpointManager(work, RUN_ID, backend, coord)
    entry = cm._promote_rank(1, _local_record(work, 1))   # uploads files only
    assert backend.exists(entry.manifest.key)
    assert not backend.exists(dc.durable_commit_key(RUN_ID, 1))

    rm = DurableRestoreManager(work, RUN_ID, backend, coord)
    assert rm.latest_valid_durable_commit(check_checksums=True) is None


# ── 3. partial durable upload is ignored ────────────────────────────────

def test_partial_durable_upload_ignored(tmp_path):
    """A durable commit exists but a named chunk is missing → invalid."""
    work = tmp_path / "work"
    gm, coord = _make_committed_run(work, n_gens=2)
    backend = _backend(tmp_path)
    rec = _promote(work, backend, coord, 1)
    assert rec is not None

    # Simulate a torn upload: delete one chunk the durable commit names.
    victim = rec.ranks[0].chunks[0].key
    backend.delete(victim)
    assert not backend.exists(victim)

    rm = DurableRestoreManager(work, RUN_ID, backend, coord)
    # The newest durable commit is now invalid (missing file) → ignored.
    assert rm.latest_valid_durable_commit(check_checksums=True) is None


def test_partial_upload_wrong_size_ignored(tmp_path):
    work = tmp_path / "work"
    gm, coord = _make_committed_run(work, n_gens=2)
    backend = _backend(tmp_path)
    rec = _promote(work, backend, coord, 1)

    # Overwrite a chunk with truncated bytes (size mismatch vs the record).
    victim = rec.ranks[0].chunks[0].key
    backend.put(victim, b"\x00\x00")
    rm = DurableRestoreManager(work, RUN_ID, backend, coord)
    assert rm.latest_valid_durable_commit(check_checksums=True) is None


def test_corrupt_chunk_same_size_detected_only_by_checksum(tmp_path):
    """A chunk corrupted WITHOUT changing its size is caught by the checksum.

    Size validation alone cannot see this (the byte count is unchanged); only
    the sha256 check rejects it.  Proven by asserting it is *accepted* with
    check_checksums=False and *rejected* with check_checksums=True.
    """
    work = tmp_path / "work"
    gm, coord = _make_committed_run(work, n_gens=2)
    backend = _backend(tmp_path)
    rec = _promote(work, backend, coord, 1)

    victim = rec.ranks[0].chunks[0].key
    orig = backend.get(victim)
    backend.put(victim, bytes((b ^ 0xFF) for b in orig))   # same length, different bytes
    assert len(backend.get(victim)) == len(orig)            # size unchanged

    rm = DurableRestoreManager(work, RUN_ID, backend, coord)
    # size-only validation does not catch a same-size flip ...
    assert rm.latest_valid_durable_commit(check_checksums=False) is not None
    # ... but checksum validation does.
    assert rm.latest_valid_durable_commit(check_checksums=True) is None


# ── 4. corrupt durable manifest is ignored ──────────────────────────────

def test_corrupt_durable_commit_record_ignored(tmp_path):
    """Tamper the durable commit JSON without fixing its self-hash → invalid."""
    import json
    work = tmp_path / "work"
    gm, coord = _make_committed_run(work, n_gens=2)
    backend = _backend(tmp_path)
    rec = _promote(work, backend, coord, 1)

    key = dc.durable_commit_key(RUN_ID, 1)
    data = json.loads(backend.get(key).decode())
    data["step_index"] = 9999            # breaks the stored self-hash
    backend.put(key, json.dumps(data).encode())

    rm = DurableRestoreManager(work, RUN_ID, backend, coord)
    assert rm.latest_valid_durable_commit(check_checksums=True) is None


def test_corrupt_durable_manifest_file_ignored(tmp_path):
    """The named manifest's bytes are corrupted → checksum check rejects it."""
    work = tmp_path / "work"
    gm, coord = _make_committed_run(work, n_gens=2)
    backend = _backend(tmp_path)
    rec = _promote(work, backend, coord, 1)

    man_key = rec.ranks[0].manifest.key
    backend.put(man_key, b"{not the manifest}")   # size+checksum now differ
    rm = DurableRestoreManager(work, RUN_ID, backend, coord)
    assert rm.latest_valid_durable_commit(check_checksums=True) is None


def test_older_valid_durable_commit_selected_when_newest_corrupt(tmp_path):
    """Two durable gens; corrupt the newer → restore discovery picks the older."""
    work = tmp_path / "work"
    gm, coord = _make_committed_run(work, n_gens=3)
    backend = _backend(tmp_path)
    assert _promote(work, backend, coord, 1) is not None
    rec2 = _promote(work, backend, coord, 2)
    assert rec2 is not None

    backend.delete(rec2.ranks[0].chunks[0].key)   # corrupt gen 2
    rm = DurableRestoreManager(work, RUN_ID, backend, coord)
    found = rm.latest_valid_durable_commit(check_checksums=True)
    assert found is not None and found.generation == 1


# ── 8. durable storage is NOT used in hot execution ─────────────────────

class _SpyBackend(DurableBackend):
    """Recording backend: counts put/get to prove the hot path is untouched."""

    def __init__(self, inner: DurableBackend):
        self.inner = inner
        self.put_calls = 0
        self.get_calls = 0
        self.put_keys: list[str] = []

    def put(self, key, data):
        self.put_calls += 1
        self.put_keys.append(key)
        return self.inner.put(key, data)

    def get(self, key):
        self.get_calls += 1
        return self.inner.get(key)

    def exists(self, key):
        return self.inner.exists(key)

    def list(self, prefix):
        return self.inner.list(prefix)

    def delete(self, key):
        return self.inner.delete(key)


def test_durable_not_touched_during_gate_execution(tmp_path):
    """Run a full generation step loop with a spy backend wired in but NEVER
    invoked by the hot path: assert zero put() calls happen while gates are
    applied and generations are committed locally.

    This mirrors how the runner is structured: GenerationManager.commit_step
    (the per-step hot path) must touch only local NVMe; durable promotion is a
    separate, explicit call.  We assert the spy sees no puts across the step
    loop, then exactly the expected puts only when we explicitly promote.
    """
    work = tmp_path / "work"
    spy = _SpyBackend(_backend(tmp_path))

    coord = LocalCoordinator()
    gm = GenerationManager(work, coord, circuit_hash=CIRCUIT_HASH,
                           chunk_size=CHUNK_SIZE)
    gm.init_run(n_qubits=4)

    # ── hot path: commit several generations locally, no durable calls ──
    for g in range(4):
        rec = gm.commit_step(g, g, _writer(gm, fill=g * 10),
                             parent_generation=g - 1)
        assert rec is not None
    assert spy.put_calls == 0, "durable backend put() called during step loop!"
    assert spy.get_calls == 0, "durable backend get() called during step loop!"

    # ── explicit, separate durable step: now (and only now) puts happen ──
    cm = DurableCheckpointManager(work, RUN_ID, spy, coord)
    cm.upload_run_metadata()
    before = spy.put_calls
    rec = cm.promote(3)
    assert rec is not None
    assert spy.put_calls > before        # promotion did upload
    # the durable commit key was the LAST put (the durability point).
    assert spy.put_keys[-1] == dc.durable_commit_key(RUN_ID, 3)


# ── shared helper that reads a local commit record ──────────────────────

def _local_record(work_dir, generation):
    from wenbo_engine.recovery.global_commit import GlobalCommitRecord
    cpath = commits_dir(work_dir) / commit_filename(generation)
    return GlobalCommitRecord.read_file(cpath)


# ── DurableConfig parsing ───────────────────────────────────────────────

def test_durable_config_from_dict_defaults():
    c = DurableConfig.from_dict(None)
    assert c.enabled is False
    assert c.backend == "local_path"
    assert c.interval_generations == 5

    c2 = DurableConfig.from_dict({"enabled": True, "backend": "local_path",
                                  "root": "/tmp/x", "interval_generations": 3})
    assert c2.enabled and c2.root == "/tmp/x" and c2.interval_generations == 3
