"""Generation manager — layout, commit protocol, and coordinators.

Directory layout (``--recovery=generation``)::

    work_dir/
      run.json                         run metadata (circuit hash, n_ranks, ...)
      plan.json                        (optional) execution plan, passthrough
      cost_model.json                  (optional) cost model, passthrough
      commits/
        commit_000000.json             global commit records (durability point)
        commit_000001.json
      rank_0000/
        generations/
          gen_000000/
            manifest.json              per-rank manifest for this generation
            chunks/
              chunk_000000.bin
          gen_000001/
            ...
        quarantine/                    incomplete/newer gens moved here on recovery

Commit protocol (per simulation step, producing generation g+1 from g):

  1. Each rank reads from committed generation g.
  2. Each rank writes its output chunks into generation g+1.
  3. Each rank fsyncs its chunk files.
  4. Each rank atomically renames temp chunk files into place.
  5. Each rank writes manifest.tmp.
  6. Each rank fsyncs manifest.tmp.
  7. Each rank renames manifest.tmp -> manifest.json.
  8. MPI gathers rank manifest hashes + statuses to the coordinator.
  9. If all ranks prepared, coordinator writes commit_00xxxx.tmp.
 10. Coordinator fsyncs commit_00xxxx.tmp.
 11. Coordinator renames it to commit_00xxxx.json.       <-- generation committed
 12. Coordinator broadcasts COMMITTED(g+1).
 13. All ranks install generation g+1 as the current source.

Steps 1–4 are performed by the caller's ``write_chunks`` callback (it uses
:func:`wenbo_engine.storage.block_store.write_chunk_atomic`, which is
tmp+fsync+rename per file).  Steps 5–7 are :meth:`RankManifest.write_atomic`.
Steps 8–13 are :meth:`GenerationManager.commit`.

This module performs *no* numerical work (rule 4) — the kernels never touch
manifests, commits, or MPI.  The coordinator abstraction keeps the protocol
testable in-process (``LocalCoordinator``) and runnable under MPI
(``MPICoordinator``) without changing the protocol code.
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from wenbo_engine.storage.block_store import chunk_filename, write_chunk_atomic
from wenbo_engine.recovery.rank_manifest import ChunkRecord, RankManifest
from wenbo_engine.recovery.global_commit import (
    GlobalCommitRecord, COMMITS_DIRNAME,
)
from wenbo_engine.recovery.recovery_events import EventType, RecoveryEventLog
from wenbo_engine.recovery.global_commit import _fsync_dir, _fsync_file

log = logging.getLogger(__name__)

RUN_JSON = "run.json"
PLAN_JSON = "plan.json"
COST_MODEL_JSON = "cost_model.json"
GENERATIONS_DIRNAME = "generations"
QUARANTINE_DIRNAME = "quarantine"
CHUNKS_DIRNAME = "chunks"


# ── layout helpers ─────────────────────────────────────────────────────

def run_json_path(work_dir: str | Path) -> Path:
    return Path(work_dir) / RUN_JSON


def commits_dir(work_dir: str | Path) -> Path:
    return Path(work_dir) / COMMITS_DIRNAME


def rank_dir(work_dir: str | Path, rank: int) -> Path:
    return Path(work_dir) / f"rank_{rank:04d}"


def generations_dir(work_dir: str | Path, rank: int) -> Path:
    return rank_dir(work_dir, rank) / GENERATIONS_DIRNAME


def gen_dir(work_dir: str | Path, rank: int, generation: int) -> Path:
    return generations_dir(work_dir, rank) / f"gen_{generation:06d}"


def gen_chunks_dir(work_dir: str | Path, rank: int, generation: int) -> Path:
    return gen_dir(work_dir, rank, generation) / CHUNKS_DIRNAME


def quarantine_dir(work_dir: str | Path, rank: int) -> Path:
    return rank_dir(work_dir, rank) / QUARANTINE_DIRNAME


# ── checksums ──────────────────────────────────────────────────────────

def sha256_file(path: str | Path, *, bufsize: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(bufsize)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


# ── run metadata ───────────────────────────────────────────────────────

@dataclass
class RunMetadata:
    circuit_hash: str
    n_ranks: int
    n_qubits: int
    chunk_size: int
    dtype: str = "complex64"
    recovery_mode: str = "generation"
    created: float = 0.0

    def to_dict(self) -> dict:
        return {
            "circuit_hash": self.circuit_hash,
            "n_ranks": self.n_ranks,
            "n_qubits": self.n_qubits,
            "chunk_size": self.chunk_size,
            "dtype": self.dtype,
            "recovery_mode": self.recovery_mode,
            "created": self.created,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RunMetadata":
        return cls(
            circuit_hash=str(d["circuit_hash"]),
            n_ranks=int(d["n_ranks"]),
            n_qubits=int(d["n_qubits"]),
            chunk_size=int(d["chunk_size"]),
            dtype=str(d.get("dtype", "complex64")),
            recovery_mode=str(d.get("recovery_mode", "generation")),
            created=float(d.get("created", 0.0)),
        )


def write_json_atomic(path: str | Path, payload: dict) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w") as f:
        f.write(json.dumps(payload, indent=2, sort_keys=True))
        f.flush()
        os.fsync(f.fileno())
    os.replace(str(tmp), str(p))
    _fsync_dir(p.parent)   # make the rename durable (run.json/plan/cost_model)
    return p


def read_run_metadata(work_dir: str | Path) -> RunMetadata | None:
    p = run_json_path(work_dir)
    if not p.exists():
        return None
    with open(p) as f:
        return RunMetadata.from_dict(json.loads(f.read()))


# ── coordinators ───────────────────────────────────────────────────────

class Coordinator:
    """Cluster collective operations needed by the commit protocol.

    Subclasses provide gather/broadcast/barrier.  Rank 0 is the coordinator.
    """

    rank: int = 0
    n_ranks: int = 1

    @property
    def is_coordinator(self) -> bool:
        return self.rank == 0

    def gather(self, obj):
        """Gather ``obj`` from every rank to the coordinator.

        Returns a list (indexed by rank) on the coordinator, ``None`` elsewhere.
        """
        raise NotImplementedError

    def broadcast(self, obj):
        """Broadcast ``obj`` from the coordinator to all ranks; return value."""
        raise NotImplementedError

    def barrier(self) -> None:
        raise NotImplementedError


class LocalCoordinator(Coordinator):
    """Single-process coordinator (n_ranks == 1).  For tests / single node."""

    def __init__(self):
        self.rank = 0
        self.n_ranks = 1

    def gather(self, obj):
        return [obj]

    def broadcast(self, obj):
        return obj

    def barrier(self) -> None:
        pass


class MPICoordinator(Coordinator):
    """Coordinator backed by an mpi4py communicator."""

    def __init__(self, comm):
        self._comm = comm
        self.rank = comm.Get_rank()
        self.n_ranks = comm.Get_size()

    def gather(self, obj):
        return self._comm.gather(obj, root=0)

    def broadcast(self, obj):
        return self._comm.bcast(obj, root=0)

    def barrier(self) -> None:
        self._comm.Barrier()


# ── prepared-state passed through gather ───────────────────────────────

@dataclass
class _RankStatus:
    rank: int
    generation: int
    prepared: bool
    manifest_hash: str
    parent_generation: int = -1
    stage_id: int = -1
    reason: str = ""


# ── generation manager ─────────────────────────────────────────────────

# write_chunks callback: (chunks_dir) -> list[ChunkRecord]
WriteChunksFn = Callable[[Path], "list[ChunkRecord]"]


class GenerationManager:
    """Drives the generation commit protocol for one rank."""

    def __init__(self, work_dir: str | Path, coordinator: Coordinator, *,
                 circuit_hash: str, dtype: str = "complex64",
                 chunk_size: int = 0,
                 events: RecoveryEventLog | None = None,
                 fault_injector=None):
        self.work_dir = Path(work_dir)
        self.coord = coordinator
        self.circuit_hash = circuit_hash
        self.dtype = dtype
        self.chunk_size = chunk_size
        self.events = events or RecoveryEventLog()
        # Deterministic fault injection (crash testing).  Defaults to the
        # shared always-off injector so the hot path is a single ``enabled``
        # check and all behaviour is unchanged when no fault is configured.
        if fault_injector is None:
            from wenbo_engine.faults.fault_injector import NULL_INJECTOR
            fault_injector = NULL_INJECTOR
        self.faults = fault_injector

    # ── fault hook ─────────────────────────────────────────────────────

    def _fault(self, fault_point, stage_id: int) -> None:
        """Fire the configured fault iff it matches this protocol point.

        No-op (one ``enabled`` check) when fault injection is disabled.  In
        ``os_exit`` mode this does not return — it models a node dying at this
        exact step of the commit protocol.
        """
        self.faults.maybe_fire(fault_point, self.coord.rank, stage_id)

    # ── run init ───────────────────────────────────────────────────────

    def init_run(self, *, n_qubits: int,
                 plan: dict | None = None,
                 cost_model: dict | None = None,
                 recovery_mode: str = "generation") -> None:
        """Coordinator writes run.json (+ optional plan/cost_model)."""
        if self.coord.is_coordinator and not run_json_path(self.work_dir).exists():
            meta = RunMetadata(
                circuit_hash=self.circuit_hash,
                n_ranks=self.coord.n_ranks,
                n_qubits=n_qubits,
                chunk_size=self.chunk_size,
                dtype=self.dtype,
                recovery_mode=recovery_mode,
                created=time.time(),
            )
            write_json_atomic(run_json_path(self.work_dir), meta.to_dict())
            if plan is not None:
                write_json_atomic(self.work_dir / PLAN_JSON, plan)
            if cost_model is not None:
                write_json_atomic(self.work_dir / COST_MODEL_JSON, cost_model)
        self.coord.barrier()

    # ── helper for the common "build records by hashing files" path ────

    def chunk_record(self, chunks_dir: Path, index: int, *,
                     checksum: bool = False) -> ChunkRecord:
        fn = chunk_filename(index)
        path = chunks_dir / fn
        return ChunkRecord(
            index=index,
            filename=fn,
            size_bytes=os.path.getsize(path),
            checksum=sha256_file(path) if checksum else None,
        )

    # ── steps 1–7: prepare ─────────────────────────────────────────────

    def prepare(self, generation: int, write_chunks: WriteChunksFn, *,
                parent_generation: int = -1,
                stage_id: int = -1) -> RankManifest:
        """Write this rank's chunks + manifest for ``generation``.

        ``write_chunks(chunks_dir)`` must write the chunk files (using
        atomic writes) and return the list of :class:`ChunkRecord`.
        ``parent_generation``/``stage_id`` record this generation's lineage
        (the generation it derives from and the circuit step that produced it).
        Returns the sealed :class:`RankManifest` (manifest.json on disk).
        """
        from wenbo_engine.faults.fault_points import FaultPoint
        rank = self.coord.rank
        gdir = gen_dir(self.work_dir, rank, generation)
        cdir = gdir / CHUNKS_DIRNAME
        cdir.mkdir(parents=True, exist_ok=True)

        self._fault(FaultPoint.BEFORE_STAGE, stage_id)   # before step 1

        records = write_chunks(cdir)          # steps 1–2,4 (atomic chunk writes)
        self._fault(FaultPoint.AFTER_ALL_WRITES, stage_id)  # after step 2
        # Step 3: fsync chunk *data* to stable storage.  write_chunk_atomic
        # only renames (it does not fsync, to keep the WAL hot path fast), so
        # the commit protocol fsyncs the durable data here before the manifest.
        for rec in records:
            _fsync_file(cdir / rec.filename)
        _fsync_dir(cdir)                       # ensure chunk renames are durable
        self._fault(FaultPoint.AFTER_RENAME, stage_id)   # after steps 3–4

        man = RankManifest(
            rank=rank,
            generation=generation,
            parent_generation=parent_generation,
            stage_id=stage_id,
            n_chunks=len(records),
            chunk_size=self.chunk_size,
            dtype=self.dtype,
            circuit_hash=self.circuit_hash,
            chunks=sorted(records, key=lambda c: c.index),
            created=time.time(),
        )
        # Steps 5–7.  ``after_tmp_fsync`` fires AFTER_MANIFEST_WRITE between the
        # fsync of manifest.tmp (step 6) and its rename to manifest.json (step
        # 7): a crash here leaves manifest.tmp on disk but NO manifest.json, so
        # recovery must still roll back to the previous generation.
        man.write_atomic(
            gdir,
            after_tmp_fsync=lambda: self._fault(
                FaultPoint.AFTER_MANIFEST_WRITE, stage_id),
        )
        _fsync_dir(gdir)
        self._fault(FaultPoint.AFTER_MANIFEST_RENAME, stage_id)  # after step 7
        self.events.emit(EventType.GENERATION_PREPARED,
                         "rank prepared generation",
                         generation=generation, rank=rank,
                         manifest_hash=man.manifest_hash)
        return man

    # ── steps 8–13: commit ─────────────────────────────────────────────

    def commit(self, generation: int, step_index: int,
               prepared: RankManifest, *,
               parent_generation: int = -1) -> GlobalCommitRecord | None:
        """Gather prepared statuses; coordinator writes the commit record.

        Returns the broadcast :class:`GlobalCommitRecord` on every rank, or
        ``None`` if the commit was aborted (a rank failed to prepare, or the
        ranks disagree on lineage — see :meth:`_coordinator_commit`).
        """
        status = _RankStatus(
            rank=self.coord.rank,
            generation=generation,
            prepared=prepared is not None
            and prepared.generation == generation
            and prepared.verify_self_hash(),
            manifest_hash=prepared.manifest_hash if prepared else "",
            parent_generation=prepared.parent_generation if prepared else -2,
            stage_id=prepared.stage_id if prepared else -2,
        )

        from wenbo_engine.faults.fault_points import FaultPoint

        gathered = self.coord.gather(status)   # step 8
        # After the coordinator has gathered every rank's prepared status but
        # before any commit record is written.
        self._fault(FaultPoint.AFTER_ALLGATHER_PREPARED, step_index)

        record: GlobalCommitRecord | None = None
        if self.coord.is_coordinator:
            # Just before steps 9–11 write/rename the global commit record:
            # a crash here means the generation is NOT durable → roll back.
            self._fault(FaultPoint.BEFORE_GLOBAL_COMMIT, step_index)
            record = self._coordinator_commit(
                generation, step_index, parent_generation, gathered)
            # Steps 9–11 done: commit_XXXXXX.json is on disk and durable.  A
            # crash here means the new generation IS committed → recover it.
            if record is not None:
                self._fault(FaultPoint.AFTER_GLOBAL_COMMIT, step_index)

        # Step 12: broadcast the (committed) record to all ranks.
        self._fault(FaultPoint.DURING_DURABLE_UPLOAD, step_index)
        record = self.coord.broadcast(record)  # step 12
        # Just before step 13 installs g+1 as the live source on each rank.
        self._fault(FaultPoint.BEFORE_DURABLE_COMMIT, step_index)

        if record is not None:                 # step 13
            self.events.emit(EventType.GENERATION_INSTALLED,
                             "installed committed generation",
                             generation=generation, rank=self.coord.rank)
        else:
            self.events.emit(EventType.COMMIT_ABORTED,
                             "commit aborted; staying on previous generation",
                             generation=generation, rank=self.coord.rank)
        return record

    def _coordinator_commit(self, generation, step_index, parent_generation,
                            gathered: "list[_RankStatus]") -> GlobalCommitRecord | None:
        not_ready = [s for s in gathered if not s.prepared]
        if not_ready or len(gathered) != self.coord.n_ranks:
            self.events.emit(
                EventType.COMMIT_ABORTED,
                f"{len(not_ready)} rank(s) not prepared",
                generation=generation,
                ranks_not_ready=[s.rank for s in not_ready],
            )
            return None

        # Write-time lineage agreement: refuse to commit if ranks disagree on
        # parent generation or stage id (a divergent rank must not be sealed
        # into a global commit).  Recovery enforces the same invariant on read.
        parents = {s.parent_generation for s in gathered}
        stages = {s.stage_id for s in gathered}
        if len(parents) != 1 or len(stages) != 1:
            self.events.emit(
                EventType.COMMIT_ABORTED,
                f"ranks disagree on lineage parents={sorted(parents)} "
                f"stages={sorted(stages)}",
                generation=generation,
            )
            return None

        record = GlobalCommitRecord(
            generation=generation,
            n_ranks=self.coord.n_ranks,
            circuit_hash=self.circuit_hash,
            step_index=step_index,
            rank_manifest_hashes={s.rank: s.manifest_hash for s in gathered},
            parent_generation=parent_generation,
            created=time.time(),
        )
        record.write_atomic(commits_dir(self.work_dir))  # steps 9–11
        self.events.emit(EventType.GENERATION_COMMITTED,
                         "wrote global commit record",
                         generation=generation,
                         commit_hash=record.commit_hash)
        return record

    # ── convenience: prepare + commit in one call ──────────────────────

    def commit_step(self, generation: int, step_index: int,
                    write_chunks: WriteChunksFn, *,
                    parent_generation: int = -1) -> GlobalCommitRecord | None:
        prepared = self.prepare(generation, write_chunks,
                                parent_generation=parent_generation,
                                stage_id=step_index)
        return self.commit(generation, step_index, prepared,
                           parent_generation=parent_generation)

    # ── retention ──────────────────────────────────────────────────────

    def prune(self, keep_generations: int = 3) -> list[int]:
        """Delete this rank's oldest generation dirs, keeping the newest N.

        Generation directories are immutable once committed, so without
        reclamation the working set grows without bound (each generation is a
        full copy of the rank's partition).  The default keeps the newest
        **three** generations (the current one plus two rollback targets) so
        that even two consecutive bad/incomplete newer generations still leave
        a valid generation on disk to recover from; with only two kept, a
        double failure could erase the last good rollback target.  If every
        retained generation is invalid the scanner returns ``None`` (explicit
        fresh start / recompute) rather than constructing a corrupt state.
        Commit records are left in place (they are tiny).  Returns the deleted
        generation numbers.
        """
        if keep_generations < 1:
            return []
        import shutil
        rank = self.coord.rank
        gens = generations_dir(self.work_dir, rank)
        if not gens.exists():
            return []
        nums = sorted(
            int(d.name[4:]) for d in gens.glob("gen_*")
            if d.name[4:].isdigit()
        )
        to_delete = nums[:-keep_generations]
        for g in to_delete:
            shutil.rmtree(gen_dir(self.work_dir, rank, g), ignore_errors=True)
        return to_delete

    # ── recovery entry point ───────────────────────────────────────────

    def resume_generation(self, *, quarantine: bool = True,
                          check_checksums: bool = False) -> int | None:
        """Return the newest committed+valid generation, or None for fresh.

        Lazy-imports the scanner to avoid an import cycle.
        """
        from wenbo_engine.recovery.recovery_scanner import RecoveryScanner
        scanner = RecoveryScanner(self.work_dir, events=self.events)
        result = scanner.scan(quarantine=quarantine and self.coord.is_coordinator,
                              check_checksums=check_checksums)
        self.coord.barrier()
        return result.generation
