"""Durable restore (R4) tests.

Covered required cases:
  5. delete the local state, restore the durable generation, continue
  6. the restored final state / norm is valid
  7. durable checkpoint works WITH generation recovery (the local recovery
     scanner finds the restored generation and resume picks it up)

These use the smallest possible *real* generation run: a genuine normalized
state vector (built with the reference simulator) is partitioned into chunks
and committed through the recovery package's public commit protocol, so the
restored state is numerically meaningful (norm == 1).
"""
import shutil

import numpy as np

from wenbo_engine.kernel.ref_dense import simulate
from wenbo_engine.storage.block_store import (
    write_chunk_atomic, read_chunk, chunk_filename, DTYPE,
)
from wenbo_engine.recovery import (
    GenerationManager, LocalCoordinator, RecoveryScanner,
    gen_dir, gen_chunks_dir,
)

from wenbo_engine.durable import (
    LocalPathBackend, DurableCheckpointManager, DurableRestoreManager,
)

CIRCUIT_HASH = "feedface00001234"
RUN_ID = "durable_restore_run"


# ── real-state generation builder ───────────────────────────────────────

def _real_state(n_qubits=3):
    """A genuine normalized state vector from a small circuit (GHZ-like)."""
    gates = [{"qubits": [0], "gate": "H"}]
    for q in range(n_qubits - 1):
        gates.append({"qubits": [q, q + 1], "gate": "CNOT"})
    circ = {"number_of_qubits": n_qubits, "gates": gates}
    psi = simulate(circ).astype(DTYPE)
    return psi


def _commit_real_run(work_dir, psi, chunk_size):
    """Commit a single-rank run whose generations hold a real state vector.

    gen 0 = |0...0>, gen 1 = the real state ``psi``.  Returns the manager.
    """
    n_chunks = len(psi) // chunk_size
    coord = LocalCoordinator()
    gm = GenerationManager(work_dir, coord, circuit_hash=CIRCUIT_HASH,
                           chunk_size=chunk_size)
    gm.init_run(n_qubits=int(np.log2(len(psi))))

    def init_writer(cdir):
        recs = []
        for ci in range(n_chunks):
            data = np.zeros(chunk_size, dtype=DTYPE)
            if ci == 0:
                data[0] = 1.0
            write_chunk_atomic(cdir / chunk_filename(ci), data)
            recs.append(gm.chunk_record(cdir, ci, checksum=True))
        return recs

    assert gm.commit_step(0, -1, init_writer, parent_generation=-1) is not None

    def state_writer(cdir):
        recs = []
        for ci in range(n_chunks):
            seg = psi[ci * chunk_size:(ci + 1) * chunk_size]
            write_chunk_atomic(cdir / chunk_filename(ci), seg)
            recs.append(gm.chunk_record(cdir, ci, checksum=True))
        return recs

    assert gm.commit_step(1, 1, state_writer, parent_generation=0) is not None
    return gm, coord


def _read_state(work_dir, rank, generation, n_chunks):
    cdir = gen_chunks_dir(work_dir, rank, generation)
    return np.concatenate(
        [read_chunk(cdir / chunk_filename(ci)) for ci in range(n_chunks)]
    ).astype(np.complex128)


def _norm(state):
    return float(np.sqrt(np.sum(np.abs(state) ** 2)))


# ── 5 + 6: delete local state, restore durable generation, valid norm ───

def test_delete_local_restore_durable_and_valid_norm(tmp_path):
    chunk_size = 2
    psi = _real_state(n_qubits=3)             # length 8
    n_chunks = len(psi) // chunk_size
    work = tmp_path / "work"

    gm, coord = _commit_real_run(work, psi, chunk_size)
    backend = LocalPathBackend(tmp_path / "durable")

    # Promote the final committed generation (gen 1) to durable storage.
    cm = DurableCheckpointManager(work, RUN_ID, backend, coord)
    cm.upload_run_metadata()
    assert cm.promote(1) is not None

    # Capture the original state for comparison, then DESTROY the local work_dir.
    original = _read_state(work, 0, 1, n_chunks)
    assert abs(_norm(original) - 1.0) < 1e-6
    shutil.rmtree(work)
    assert not work.exists()

    # ── Restore from durable storage into a rebuilt local work_dir. ──
    rm = DurableRestoreManager(work, RUN_ID, backend, coord)
    result = rm.restore_latest(check_checksums=True)
    assert result.restored
    assert result.generation == 1

    # Case 6: restored state is byte-identical and still normalized.
    restored = _read_state(work, 0, 1, n_chunks)
    assert np.allclose(restored, original, atol=0)
    assert abs(_norm(restored) - 1.0) < 1e-6


# ── 7: durable checkpoint works WITH generation recovery ────────────────

def test_restore_then_generation_recovery_resumes(tmp_path):
    """After restore, the local recovery scanner must find the restored gen.

    Proves the restore re-publishes the local global commit record LAST, so the
    standard generation-recovery path (RecoveryScanner / resume_generation)
    picks up exactly the restored generation and execution can continue.
    """
    chunk_size = 2
    psi = _real_state(n_qubits=3)
    n_chunks = len(psi) // chunk_size
    work = tmp_path / "work"

    gm, coord = _commit_real_run(work, psi, chunk_size)
    backend = LocalPathBackend(tmp_path / "durable")
    cm = DurableCheckpointManager(work, RUN_ID, backend, coord)
    cm.upload_run_metadata()
    assert cm.promote(1) is not None

    shutil.rmtree(work)
    rm = DurableRestoreManager(work, RUN_ID, backend, coord)
    assert rm.restore_latest(check_checksums=True).generation == 1

    # The standard generation-recovery scanner now recovers the restored gen
    # locally — no durable access — and validates checksums.
    res = RecoveryScanner(work).scan(quarantine=False, check_checksums=True)
    assert res.recovered
    assert res.generation == 1
    assert res.record.circuit_hash == CIRCUIT_HASH

    # And a fresh GenerationManager.resume_generation agrees (resume point).
    gm2 = GenerationManager(work, LocalCoordinator(),
                            circuit_hash=CIRCUIT_HASH, chunk_size=chunk_size)
    assert gm2.resume_generation(check_checksums=True) == 1


def test_restore_no_durable_commit_returns_none(tmp_path):
    """Nothing durable available → restore is a no-op (caller starts fresh)."""
    work = tmp_path / "work"
    backend = LocalPathBackend(tmp_path / "durable")
    rm = DurableRestoreManager(work, RUN_ID, backend, LocalCoordinator())
    result = rm.restore_latest(check_checksums=True)
    assert not result.restored
    assert result.generation is None


def test_restore_picks_newest_valid_when_multiple(tmp_path):
    """Two durable generations promoted → restore picks the newest valid one."""
    chunk_size = 2
    psi = _real_state(n_qubits=3)
    n_chunks = len(psi) // chunk_size
    work = tmp_path / "work"

    gm, coord = _commit_real_run(work, psi, chunk_size)
    # Commit a third generation (copy of psi) so we have gens 0,1,2.
    def writer(cdir):
        recs = []
        for ci in range(n_chunks):
            seg = psi[ci * chunk_size:(ci + 1) * chunk_size]
            write_chunk_atomic(cdir / chunk_filename(ci), seg)
            recs.append(gm.chunk_record(cdir, ci, checksum=True))
        return recs
    assert gm.commit_step(2, 2, writer, parent_generation=1) is not None

    backend = LocalPathBackend(tmp_path / "durable")
    cm = DurableCheckpointManager(work, RUN_ID, backend, coord)
    cm.upload_run_metadata()
    assert cm.promote(1) is not None
    assert cm.promote(2) is not None

    shutil.rmtree(work)
    rm = DurableRestoreManager(work, RUN_ID, backend, coord)
    assert rm.restore_latest(check_checksums=True).generation == 2
