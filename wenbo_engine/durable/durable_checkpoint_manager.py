"""Durable promotion protocol — local committed generation → durable store.

Promotion protocol (per generation ``g`` being promoted):

  1. Validate ``g`` is **locally committed**: a global commit record names it,
     and the local generation validates (manifests + chunk sizes).  We never
     promote uncommitted / partial / corrupt local state.
  2. Each rank uploads its chunk files, then its durable manifest.
  3. Every uploaded file's size AND checksum are verified against what was put
     (re-read from the backend) before the rank is considered done.
  4. Each rank reports its :class:`DurableRankEntry` (manifest + chunk records).
  5. The coordinator writes ``durable_commit_XXXXXX.json`` **LAST** — this is
     the durability point.
  6. Restore IGNORES any durable generation without a valid durable commit.

This module runs only *between* committed generations, at the configured
interval — never on the hot gate-execution path.  It drives cluster collectives
through the same :class:`~wenbo_engine.recovery.generation_manager.Coordinator`
abstraction the recovery package uses, so it is testable in-process
(``LocalCoordinator``) and runnable under MPI (``MPICoordinator``).

No numpy, no kernel imports.  The only ``wenbo_engine`` couplings are to the
recovery package (read-only: validate + locate the local committed generation)
and the durable backend (byte movement).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path

from wenbo_engine.recovery.generation_manager import (
    Coordinator, gen_dir, gen_chunks_dir, sha256_file, commits_dir,
)
from wenbo_engine.recovery.rank_manifest import RankManifest
from wenbo_engine.recovery.global_commit import (
    GlobalCommitRecord, commit_filename,
)
from wenbo_engine.recovery.generation_validator import validate_generation

from wenbo_engine.durable.backend import DurableBackend
from wenbo_engine.durable import durable_commit as dc
from wenbo_engine.durable.durable_commit import (
    DurableCommitRecord, DurableRankEntry, DurableFileRecord,
)

log = logging.getLogger(__name__)

# How durable storage handles extent-backed generations in THIS branch:
# extents are materialized to chunk files and uploaded as chunk files (restore
# re-packs them).  Correct, but the durable store does NOT yet get the extent
# file-count reduction.  Surfaced so we never claim durable file-count savings.
DURABLE_EXTENT_MODE = "materialize_chunks_for_durable"


@dataclass
class DurableConfig:
    """Durable-checkpoint configuration (parsed from yaml/CLI)."""
    enabled: bool = False
    backend: str = "local_path"
    root: str = ""
    interval_generations: int = 5

    @classmethod
    def from_dict(cls, d: dict | None) -> "DurableConfig":
        d = dict(d or {})
        return cls(
            enabled=bool(d.get("enabled", False)),
            backend=str(d.get("backend", "local_path")),
            root=str(d.get("root", "") or ""),
            interval_generations=int(d.get("interval_generations", 5)),
        )

    def to_dict(self) -> dict:
        return {
            "enabled": self.enabled,
            "backend": self.backend,
            "root": self.root,
            "interval_generations": self.interval_generations,
        }

    def build_backend(self) -> DurableBackend:
        """Instantiate the configured backend (raises if misconfigured)."""
        from wenbo_engine.durable import make_backend
        if self.backend == "local_path":
            if not self.root:
                raise ValueError("durable.root required for backend=local_path")
            return make_backend("local_path", root=self.root)
        return make_backend(self.backend, root=self.root)


class DurablePromotionError(RuntimeError):
    """Raised when a generation cannot be safely promoted to durable storage."""


class DurableCheckpointManager:
    """Promotes locally committed generations into the durable store."""

    def __init__(self, work_dir: str | Path, run_id: str,
                 backend: DurableBackend, coordinator: Coordinator):
        self.work_dir = Path(work_dir)
        self.run_id = run_id
        self.backend = backend
        self.coord = coordinator

    # ── run-level metadata (once, by the coordinator) ───────────────────

    def upload_run_metadata(self) -> None:
        """Upload run.json (+ plan.json if present).  Coordinator only."""
        if not self.coord.is_coordinator:
            self.coord.barrier()
            return
        run_json = self.work_dir / "run.json"
        if run_json.exists():
            self.backend.put(dc.durable_run_json_key(self.run_id),
                             run_json.read_bytes())
        plan_json = self.work_dir / "plan.json"
        if plan_json.exists():
            self.backend.put(dc.durable_plan_json_key(self.run_id),
                             plan_json.read_bytes())
        self.coord.barrier()

    # ── pre-flight: is the generation already durable? ──────────────────

    def latest_durable_generation(self) -> int | None:
        """Newest VALID durable generation, or None (read-only)."""
        from wenbo_engine.durable.durable_restore_manager import (
            DurableRestoreManager,
        )
        rm = DurableRestoreManager(self.work_dir, self.run_id, self.backend,
                                   self.coord)
        rec = rm.latest_valid_durable_commit(check_checksums=True)
        return rec.generation if rec is not None else None

    # ── step 1: validate the local committed generation ─────────────────

    def _load_local_commit(self, generation: int) -> GlobalCommitRecord:
        cpath = commits_dir(self.work_dir) / commit_filename(generation)
        if not cpath.exists():
            raise DurablePromotionError(
                f"generation {generation} is not locally committed "
                f"(no {cpath.name})")
        record = GlobalCommitRecord.read_file(cpath)
        res = validate_generation(
            self.work_dir, record,
            check_sizes=True, check_checksums=False,
        )
        if not res.valid:
            raise DurablePromotionError(
                f"generation {generation} fails local validation: {res.reason}")
        return record

    # ── step 2–4: this rank uploads chunks + manifest, then verifies ────

    def _promote_rank(self, generation: int, record: GlobalCommitRecord
                      ) -> DurableRankEntry:
        rank = self.coord.rank
        gdir = gen_dir(self.work_dir, rank, generation)
        cdir = gen_chunks_dir(self.work_dir, rank, generation)
        man = RankManifest.read(gdir)

        # Extent-backed generation: unpack to chunk files so the existing
        # per-chunk upload path works unchanged.
        #
        # LIMITATION (explicit, by design in this branch): durable storage holds
        # CHUNK files even for extent-backed generations — durable_extent_mode ==
        # DURABLE_EXTENT_MODE ("materialize_chunks_for_durable").  Promote/restore
        # of extent generations is CORRECT (restore re-packs to the manifest's
        # exact extent layout), but the durable store does NOT get the
        # file-count reduction.  Extent-native durable upload is a future branch;
        # do not claim durable file-count reduction.
        if any(c.is_extent for c in man.chunks):
            from wenbo_engine.storage.extent_store import materialize_to_chunk_files
            materialize_to_chunk_files(gdir, man.chunks)
            if self.coord.is_coordinator:
                log.warning("durable: extent gen %d promoted via %s — durable "
                            "store holds chunk files (no durable file-count "
                            "reduction yet)", generation, DURABLE_EXTENT_MODE)

        # Upload + verify each chunk file.
        chunk_records: list[DurableFileRecord] = []
        for c in sorted(man.chunks, key=lambda x: x.index):
            src = cdir / c.filename
            key = dc.durable_chunk_key(self.run_id, generation, rank, c.filename)
            pr = self.backend.put_file(key, src)
            local_size = src.stat().st_size
            local_csum = sha256_file(src)
            # Verify what we uploaded matches what we read locally (step 3).
            if pr.size_bytes != local_size:
                raise DurablePromotionError(
                    f"rank {rank} gen {generation}: chunk {c.filename} uploaded "
                    f"size {pr.size_bytes} != local {local_size}")
            if pr.checksum != local_csum:
                raise DurablePromotionError(
                    f"rank {rank} gen {generation}: chunk {c.filename} uploaded "
                    f"checksum mismatch")
            # Re-read from the backend and re-verify (catches a lying put).
            if self.backend.size(key) != local_size:
                raise DurablePromotionError(
                    f"rank {rank} gen {generation}: chunk {c.filename} durable "
                    f"size mismatch after upload")
            if self.backend.checksum(key) != local_csum:
                raise DurablePromotionError(
                    f"rank {rank} gen {generation}: chunk {c.filename} durable "
                    f"checksum mismatch after upload")
            chunk_records.append(DurableFileRecord(
                key=key, size_bytes=local_size, checksum=local_csum))

        # Upload + verify the manifest (after its chunks — chunks-before-manifest
        # mirrors the local commit ordering).
        man_key = dc.durable_manifest_key(self.run_id, generation, rank)
        man_path = gdir / RankManifest.MANIFEST_NAME
        man_pr = self.backend.put_file(man_key, man_path)
        man_local_csum = sha256_file(man_path)
        if man_pr.checksum != man_local_csum or \
                self.backend.checksum(man_key) != man_local_csum:
            raise DurablePromotionError(
                f"rank {rank} gen {generation}: manifest durable checksum mismatch")

        return DurableRankEntry(
            rank=rank,
            manifest=DurableFileRecord(
                key=man_key, size_bytes=man_path.stat().st_size,
                checksum=man_local_csum),
            manifest_hash=man.manifest_hash,
            chunks=chunk_records,
        )

    # ── public: promote one generation ──────────────────────────────────

    def promote(self, generation: int) -> DurableCommitRecord | None:
        """Promote a locally committed generation to durable storage.

        Returns the sealed :class:`DurableCommitRecord` on every rank, or
        ``None`` if a rank failed to upload (no durable commit is written, so
        the partial upload is ignored by restore).
        """
        # Step 1: coordinator validates the local commit; abort all on failure.
        local_record: GlobalCommitRecord | None = None
        err: str | None = None
        if self.coord.is_coordinator:
            try:
                local_record = self._load_local_commit(generation)
            except DurablePromotionError as e:
                err = str(e)
        err = self.coord.broadcast(err)
        if err is not None:
            raise DurablePromotionError(err)
        local_record = self.coord.broadcast(local_record)

        # Step 2–4: every rank uploads + verifies its partition.
        entry: DurableRankEntry | None = None
        rank_err: str | None = None
        try:
            entry = self._promote_rank(generation, local_record)
        except (DurablePromotionError, OSError) as e:
            rank_err = f"rank {self.coord.rank}: {e}"

        gathered = self.coord.gather((entry, rank_err))

        record: DurableCommitRecord | None = None
        if self.coord.is_coordinator:
            record = self._coordinator_durable_commit(
                generation, local_record, gathered)
        record = self.coord.broadcast(record)
        self.coord.barrier()
        return record

    def _coordinator_durable_commit(
            self, generation: int, local_record: GlobalCommitRecord,
            gathered: list) -> DurableCommitRecord | None:
        failures = [e for (_entry, e) in gathered if e is not None]
        entries = [entry for (entry, e) in gathered if e is None and entry]
        if failures or len(entries) != self.coord.n_ranks:
            log.warning("durable promotion aborted for gen %d: %s",
                        generation, failures or "missing rank entries")
            return None

        record = DurableCommitRecord(
            generation=generation,
            n_ranks=self.coord.n_ranks,
            circuit_hash=local_record.circuit_hash,
            step_index=local_record.step_index,
            parent_generation=local_record.parent_generation,
            source_commit_hash=local_record.commit_hash,
            ranks={e.rank: e for e in entries},
            created=time.time(),
        )
        record.validate()
        record.seal()
        # Step 5: write the durable commit record LAST (the durability point).
        self.backend.put(dc.durable_commit_key(self.run_id, generation),
                         record.to_bytes())
        log.info("durable promotion: gen %d committed (%d ranks)",
                 generation, self.coord.n_ranks)
        return record

    # ── convenience used by the runner integration ──────────────────────

    def maybe_promote(self, generation: int, interval: int) -> bool:
        """Promote ``generation`` if it falls on the configured interval.

        Generation 0 (init) and any generation whose number is a multiple of
        ``interval`` are promoted, unless already durable.  Returns True iff a
        promotion was performed.  Safe to call on every rank every step.
        """
        if interval < 1:
            return False
        if generation != 0 and (generation % interval) != 0:
            return False
        latest = self.latest_durable_generation()
        if latest is not None and latest >= generation:
            return False
        rec = self.promote(generation)
        return rec is not None
