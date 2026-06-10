"""Crash -> resume proofs for generation recovery via deterministic faults.

Proves (deterministically, no race):
  1. crash BEFORE global commit recovers the PREVIOUS generation
  2. crash AFTER global commit recovers the NEW generation
  3. an MPI-heavy workload with --recovery generation can crash and resume
     (real mpirun when available; otherwise the in-process runner/LocalCoordinator
      equivalent — single-rank COMM_WORLD exercises the same commit protocol)
  4. recovery_events records the injected fault
  5. no partial generation is ever accepted

Proofs 1, 2, 4, 5 run in-process through the real commit protocol
(GenerationManager + LocalCoordinator) driven by the real FaultInjector.  The
``exception`` crash mode is used in-process so the deliberate crash does not
kill the pytest process; the fault hooks fire at the SAME protocol positions as
``os_exit`` would, so the on-disk state recovery sees is identical.  The
``os_exit`` hard-crash path is exercised separately via a subprocess.
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

from wenbo_engine.storage.block_store import write_chunk_atomic, chunk_filename
from wenbo_engine.recovery import (
    GenerationManager, LocalCoordinator, RecoveryScanner, EventType,
    commits_dir, gen_dir, quarantine_dir, commit_filename,
)
from wenbo_engine.recovery.recovery_events import RecoveryEventLog
from wenbo_engine.faults import (
    FaultPoint, FaultInjector, InjectedFault, CONFIG_KEY, EXCEPTION,
)

CIRCUIT_HASH = "faultcafe00000001"
CHUNK_SIZE = 4
N_CHUNKS_PER_RANK = 2


def _fi(point, *, rank=None, stage_id=None, mode=EXCEPTION, events=None):
    cfg = {CONFIG_KEY: {
        "enabled": True, "fault_point": str(point),
        "rank": rank, "stage_id": stage_id, "mode": mode,
    }}
    return FaultInjector(cfg, events=events)


def _writer(gm, fill):
    def w(cdir):
        recs = []
        for ci in range(N_CHUNKS_PER_RANK):
            write_chunk_atomic(cdir / chunk_filename(ci),
                               np.full(CHUNK_SIZE, fill + ci, dtype=np.complex64))
            recs.append(gm.chunk_record(cdir, ci))
        return recs
    return w


def _manager(tmp_path, injector=None, events=None):
    return GenerationManager(
        tmp_path, LocalCoordinator(), circuit_hash=CIRCUIT_HASH,
        chunk_size=CHUNK_SIZE, events=events, fault_injector=injector,
    )


# ── Proof 1: crash BEFORE global commit recovers the PREVIOUS generation ─

@pytest.mark.parametrize("point", [
    FaultPoint.BEFORE_STAGE,
    FaultPoint.AFTER_ALL_WRITES,
    FaultPoint.AFTER_RENAME,
    FaultPoint.AFTER_MANIFEST_WRITE,
    FaultPoint.AFTER_MANIFEST_RENAME,
    FaultPoint.AFTER_ALLGATHER_PREPARED,
    FaultPoint.BEFORE_GLOBAL_COMMIT,
])
def test_crash_before_global_commit_recovers_previous(tmp_path, point):
    # Commit generation 0 cleanly (no fault).
    gm0 = _manager(tmp_path)
    gm0.init_run(n_qubits=4)
    assert gm0.commit_step(0, 0, _writer(gm0, 0)) is not None

    # Attempt generation 1 with a fault somewhere before the global commit.
    events = RecoveryEventLog()
    inj = _fi(point, stage_id=1, events=events)
    gm1 = _manager(tmp_path, injector=inj, events=events)
    with pytest.raises(InjectedFault):
        gm1.commit_step(1, 1, _writer(gm1, 10), parent_generation=0)

    # No commit record for gen 1 must exist.
    assert not (commits_dir(tmp_path) / commit_filename(1)).exists()

    res = RecoveryScanner(tmp_path).scan()
    assert res.generation == 0, f"{point}: expected rollback to gen 0"
    # Any partially-written gen 1 is quarantined, never accepted (proof 5).
    if gen_dir(tmp_path, 0, 1).exists() or \
            (quarantine_dir(tmp_path, 0) / "gen_000001").exists():
        assert not gen_dir(tmp_path, 0, 1).exists()
        assert (quarantine_dir(tmp_path, 0) / "gen_000001").exists()


# ── Proof 2: crash AFTER global commit recovers the NEW generation ──────

@pytest.mark.parametrize("point", [
    FaultPoint.AFTER_GLOBAL_COMMIT,
    FaultPoint.DURING_DURABLE_UPLOAD,
    FaultPoint.BEFORE_DURABLE_COMMIT,
])
def test_crash_after_global_commit_recovers_new(tmp_path, point):
    gm0 = _manager(tmp_path)
    gm0.init_run(n_qubits=4)
    assert gm0.commit_step(0, 0, _writer(gm0, 0)) is not None

    events = RecoveryEventLog()
    inj = _fi(point, stage_id=1, events=events)
    gm1 = _manager(tmp_path, injector=inj, events=events)
    with pytest.raises(InjectedFault):
        gm1.commit_step(1, 1, _writer(gm1, 10), parent_generation=0)

    # The commit record for gen 1 IS on disk (fault was at/after the commit).
    assert (commits_dir(tmp_path) / commit_filename(1)).exists()

    res = RecoveryScanner(tmp_path).scan()
    assert res.generation == 1, f"{point}: expected the NEW gen 1 to recover"


# ── Proof 4: recovery_events records the injected fault ─────────────────

def test_recovery_events_records_the_fault(tmp_path):
    gm0 = _manager(tmp_path)
    gm0.init_run(n_qubits=4)
    gm0.commit_step(0, 0, _writer(gm0, 0))

    events = RecoveryEventLog()
    inj = _fi(FaultPoint.AFTER_GLOBAL_COMMIT, stage_id=1, events=events)
    gm1 = _manager(tmp_path, injector=inj, events=events)
    with pytest.raises(InjectedFault):
        gm1.commit_step(1, 1, _writer(gm1, 10), parent_generation=0)

    assert events.has(EventType.FAULT_INJECTED)
    ev = events.of_type(EventType.FAULT_INJECTED)[0]
    assert ev.details["fault_point"] == "AFTER_GLOBAL_COMMIT"
    assert ev.details["stage_id"] == 1
    assert ev.details["mode"] == EXCEPTION
    # The recorded event is JSON-serializable (it is what lands in
    # recovery_events.json).
    json.dumps(ev.to_dict())


# ── Proof 5: no partial generation is ever accepted ────────────────────

def test_no_partial_generation_accepted(tmp_path):
    """A crash that writes only the FIRST chunk of gen 1 must never be picked.

    We drive prepare() partway by faulting at AFTER_ALL_WRITES after manually
    leaving a single chunk, then assert recovery rolls back to gen 0.
    """
    gm0 = _manager(tmp_path)
    gm0.init_run(n_qubits=4)
    gm0.commit_step(0, 0, _writer(gm0, 0))

    # Fault right after the manifest is renamed but BEFORE the global commit:
    # the rank looks fully prepared on disk, yet no commit record names gen 1.
    events = RecoveryEventLog()
    inj = _fi(FaultPoint.AFTER_MANIFEST_RENAME, stage_id=1, events=events)
    gm1 = _manager(tmp_path, injector=inj, events=events)
    with pytest.raises(InjectedFault):
        gm1.commit_step(1, 1, _writer(gm1, 10), parent_generation=0)

    # Manifest + chunks for gen 1 exist on disk...
    assert (gen_dir(tmp_path, 0, 1) / "manifest.json").exists()
    # ...but there is NO commit record, so recovery must reject gen 1.
    assert not (commits_dir(tmp_path) / commit_filename(1)).exists()
    res = RecoveryScanner(tmp_path).scan()
    assert res.generation == 0
    # The fully-written-but-uncommitted gen 1 is quarantined, not accepted.
    assert (quarantine_dir(tmp_path, 0) / "gen_000001").exists()
    assert not gen_dir(tmp_path, 0, 1).exists()


def test_partial_write_then_rollback(tmp_path):
    """Crash partway through writing chunks (AFTER_PARTIAL_WRITE analogue).

    Manually create a half-written gen-1 dir (one chunk, no manifest) then a
    later valid scan must roll back to gen 0 and quarantine the partial gen.
    """
    gm0 = _manager(tmp_path)
    gm0.init_run(n_qubits=4)
    gm0.commit_step(0, 0, _writer(gm0, 0))

    # Half-written generation 1: a single chunk, no manifest, no commit.
    from wenbo_engine.recovery import gen_chunks_dir
    cdir = gen_chunks_dir(tmp_path, 0, 1)
    cdir.mkdir(parents=True, exist_ok=True)
    write_chunk_atomic(cdir / chunk_filename(0),
                       np.full(CHUNK_SIZE, 99, dtype=np.complex64))

    res = RecoveryScanner(tmp_path).scan()
    assert res.generation == 0
    assert (quarantine_dir(tmp_path, 0) / "gen_000001").exists()


# ══════════════════════════════════════════════════════════════════════
# Proof 3: MPI-heavy workload with --recovery generation crashes and resumes
# ══════════════════════════════════════════════════════════════════════

def _mpi_circuit():
    # mpi_nonlocal_heavy-style: gates on the top (rank) bit force MPI traffic
    # under >1 rank; under 1 rank it still drives multiple commit generations.
    return {"number_of_qubits": 3, "gates": [
        {"qubits": [0], "gate": "H"},
        {"qubits": [2, 0], "gate": "CNOT"},   # qubit 2 is the rank bit @ 2 ranks
        {"qubits": [1], "gate": "H"},
        {"qubits": [2, 1], "gate": "CNOT"},
    ]}


_CRASH_SCRIPT = r'''
import sys, json, os
sys.path.insert(0, "{repo_root}")
from mpi4py import MPI
from wenbo_engine.mpi import mpi_runner
cd = json.loads(r"""{cd_json}""")
mpi_runner.run(cd, "{work_dir}", chunk_size={cs}, recovery="generation",
               comm=MPI.COMM_WORLD)
'''


def _run_crash_subprocess(np_ranks, work_dir, cd, cs, fault_env, *, use_mpirun):
    repo_root = str(Path(__file__).resolve().parent.parent.parent)
    script = _CRASH_SCRIPT.format(
        repo_root=repo_root, cd_json=json.dumps(cd), work_dir=work_dir, cs=cs)
    env = os.environ.copy()
    env.update(fault_env)
    if use_mpirun:
        cmd = ["mpirun", "-np", str(np_ranks), sys.executable, "-c", script]
    else:
        cmd = [sys.executable, "-c", script]
    return subprocess.run(cmd, env=env, capture_output=True, timeout=180)


def test_mpi_heavy_crash_and_resume():
    """Crash an MPI-heavy generation run with a hard os_exit, then resume.

    Uses real mpirun (np=2) if available; otherwise a single-rank subprocess
    that exercises the identical commit protocol via Local/MPI COMM_WORLD=1.
    """
    pytest.importorskip("mpi4py")
    import shutil
    from mpi4py import MPI
    if MPI.COMM_WORLD.Get_size() != 1:
        pytest.skip("run under a single launching rank; this test spawns its own")

    use_mpirun = shutil.which("mpirun") is not None
    np_ranks = 2 if use_mpirun else 1

    cd = _mpi_circuit()
    from wenbo_engine.kernel.ref_dense import simulate
    ref = simulate(cd)

    with tempfile.TemporaryDirectory() as td:
        # chunk_size: whole-state for 1 rank; half-state (2 chunks) for 2 ranks.
        cs = (1 << cd["number_of_qubits"]) if np_ranks == 1 else \
            (1 << (cd["number_of_qubits"] - 1))
        # Hard crash AFTER the global commit of generation 2 (stage_id 1).
        fault_env = {
            "WE_FAULT_POINT": "AFTER_GLOBAL_COMMIT",
            "WE_FAULT_STAGE": "1",
            "WE_FAULT_MODE": "os_exit",
        }
        r = _run_crash_subprocess(np_ranks, td, cd, cs, fault_env,
                                  use_mpirun=use_mpirun)
        assert r.returncode != 0, \
            "expected a crash exit code\n" + r.stderr.decode()[-2000:]

        # The fault was recorded durably (proof #4 survives a hard crash).
        sink = Path(td) / "fault_events.jsonl"
        assert sink.exists(), "fault_events.jsonl missing after crash"
        recorded = [json.loads(ln) for ln in sink.read_text().splitlines() if ln]
        assert any(e["details"]["fault_point"] == "AFTER_GLOBAL_COMMIT"
                   for e in recorded)

        # Generation 2 was committed before the crash; recovery picks it up and
        # rejects any partial newer generation (proof #5).
        scanned = RecoveryScanner(td).scan()
        assert scanned.generation == 2

        # Resume to completion (no fault env) → correct final state.
        clean_env = {k: v for k, v in os.environ.items()
                     if not k.startswith("WE_FAULT")}
        script = _CRASH_SCRIPT.format(
            repo_root=str(Path(__file__).resolve().parent.parent.parent),
            cd_json=json.dumps(cd), work_dir=td, cs=cs)
        cmd = (["mpirun", "-np", str(np_ranks)] if use_mpirun else []) + \
            [sys.executable, "-c", script]
        r2 = subprocess.run(cmd, env=clean_env, capture_output=True, timeout=180)
        assert r2.returncode == 0, r2.stderr.decode()[-2000:]

        state = mpi_runner_collect(td)
        np.testing.assert_allclose(state, ref, atol=1e-6)


def mpi_runner_collect(work_dir):
    """Reassemble the full state from every rank's committed chunks on disk.

    ``mpi_runner.collect_state`` gathers only within the caller's communicator;
    the run happened in a separate subprocess (possibly multi-rank under
    mpirun), so here in the single pytest process we read each rank's committed
    partition directly and concatenate in rank order — rank ``r`` holds the
    high-bit block ``r``, matching the dense reference layout.
    """
    from wenbo_engine.mpi.mpi_runner import _committed_chunks_dir
    from wenbo_engine.storage.block_store import read_chunk
    work = Path(work_dir)
    ranks = sorted(int(p.name.split("_")[1])
                   for p in work.glob("rank_*") if p.is_dir())
    parts = []
    for r in ranks:
        cdir = _committed_chunks_dir(work, r)
        for f in sorted(cdir.glob("chunk_*.bin")):
            parts.append(read_chunk(f))
    return np.concatenate(parts).astype(np.complex128)
