"""Durable restore protocol — durable store → rebuilt local work_dir.

Restore protocol:

  1. Local work_dir is lost/empty.
  2. Read durable commit records from the durable store.
  3. Pick the NEWEST VALID durable commit (newest → oldest; skip invalid).
  4. For that generation, each rank restores its own partition:
       - download every chunk into a *temp* rank dir,
       - validate each file's size + checksum against the durable commit
         record (reject partial/corrupt downloads),
       - validate the downloaded manifest's self-hash and that it equals the
         hash named in the durable commit record.
  5. Atomically publish the rank's restored generation dir (temp dir → rename).
  6. Re-publish the local records LAST (run.json, then the global commit
     record) so a concurrent recovery scan never sees a commit before its
     generation's chunks exist.
  7. The caller resumes generation execution from the restored generation
     (the normal recovery scanner now finds it locally).

Validity of a durable commit (mirrors recovery's generation validity):
  - the durable commit JSON self-hash matches its content;
  - it names every rank;
  - (when ``check_checksums``) every named file exists in the backend with the
    recorded size and checksum.

Hard invariant: **no durable commit record ⇒ no durable generation** — the
restore never reconstructs state from durable chunks that no record names.

No numpy, no kernel imports.
"""
from __future__ import annotations

import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path

from wenbo_engine.recovery.generation_manager import (
    Coordinator, gen_dir, run_json_path, commits_dir, write_json_atomic,
)
from wenbo_engine.recovery.rank_manifest import RankManifest
from wenbo_engine.recovery.global_commit import GlobalCommitRecord

from wenbo_engine.durable.backend import DurableBackend, sha256_bytes
from wenbo_engine.durable import durable_commit as dc
from wenbo_engine.durable.durable_commit import (
    DurableCommitRecord, list_durable_commit_keys,
)

log = logging.getLogger(__name__)


@dataclass
class RestoreResult:
    """Outcome of a restore attempt."""
    generation: int | None = None
    record: DurableCommitRecord | None = None
    inspected: int = 0                 # durable commit records examined
    rejected: list[int] = field(default_factory=list)

    @property
    def restored(self) -> bool:
        return self.generation is not None


class DurableRestoreManager:
    """Restores the newest valid durable generation into a local work_dir."""

    def __init__(self, work_dir: str | Path, run_id: str,
                 backend: DurableBackend, coordinator: Coordinator):
        self.work_dir = Path(work_dir)
        self.run_id = run_id
        self.backend = backend
        self.coord = coordinator

    # ── validity of a single durable commit record ──────────────────────

    def _durable_commit_valid(self, record: DurableCommitRecord, *,
                              check_checksums: bool) -> bool:
        if not record.verify_self_hash():
            log.warning("durable commit gen %d: self-hash mismatch",
                        record.generation)
            return False
        try:
            record.validate()
        except ValueError as e:
            log.warning("durable commit gen %d: %s", record.generation, e)
            return False
        if not check_checksums:
            return True
        # Verify every named file exists with the recorded size + checksum.
        for rank in range(record.n_ranks):
            entry = record.ranks[rank]
            files = [entry.manifest, *entry.chunks]
            for fr in files:
                if not self.backend.exists(fr.key):
                    log.warning("durable gen %d rank %d: missing %s",
                                record.generation, rank, fr.key)
                    return False
                if self.backend.size(fr.key) != fr.size_bytes:
                    log.warning("durable gen %d rank %d: size mismatch %s",
                                record.generation, rank, fr.key)
                    return False
                if self.backend.checksum(fr.key) != fr.checksum:
                    log.warning("durable gen %d rank %d: checksum mismatch %s",
                                record.generation, rank, fr.key)
                    return False
        return True

    def latest_valid_durable_commit(self, *, check_checksums: bool = True
                                    ) -> DurableCommitRecord | None:
        """Newest valid durable commit record, or None (read-only)."""
        keys = list_durable_commit_keys(self.backend, self.run_id)
        for key in reversed(keys):       # newest → oldest
            try:
                record = DurableCommitRecord.from_json(self.backend.get(key))
            except (ValueError, OSError, KeyError) as e:
                log.warning("unreadable durable commit %s: %s", key, e)
                continue
            if self._durable_commit_valid(record,
                                          check_checksums=check_checksums):
                return record
        return None

    # ── per-rank restore into a temp dir, then atomic publish ───────────

    def _restore_rank(self, record: DurableCommitRecord) -> None:
        rank = self.coord.rank
        entry = record.ranks[rank]
        generation = record.generation

        final_gdir = gen_dir(self.work_dir, rank, generation)
        tmp_gdir = final_gdir.with_name(final_gdir.name + ".restoring")
        if tmp_gdir.exists():
            shutil.rmtree(tmp_gdir, ignore_errors=True)
        tmp_chunks = tmp_gdir / "chunks"
        tmp_chunks.mkdir(parents=True, exist_ok=True)

        # Download + verify every chunk into the temp dir.
        for fr in entry.chunks:
            filename = fr.key.rsplit("/", 1)[-1]
            dst = tmp_chunks / filename
            data = self.backend.get(fr.key)
            if len(data) != fr.size_bytes:
                raise RuntimeError(
                    f"rank {rank} gen {generation}: chunk {filename} download "
                    f"size {len(data)} != {fr.size_bytes}")
            if sha256_bytes(data) != fr.checksum:
                raise RuntimeError(
                    f"rank {rank} gen {generation}: chunk {filename} download "
                    f"checksum mismatch")
            self.backend.get_to_file(fr.key, dst)

        # Download + verify the manifest, then cross-check its hash.
        man_data = self.backend.get(entry.manifest.key)
        if sha256_bytes(man_data) != entry.manifest.checksum:
            raise RuntimeError(
                f"rank {rank} gen {generation}: manifest download checksum mismatch")
        man = RankManifest.from_json(man_data)
        if not man.verify_self_hash() or man.manifest_hash != entry.manifest_hash:
            raise RuntimeError(
                f"rank {rank} gen {generation}: restored manifest hash mismatch")
        self.backend.get_to_file(entry.manifest.key,
                                 tmp_gdir / RankManifest.MANIFEST_NAME)

        # Atomic publish: replace any partial final dir with the temp dir.
        if final_gdir.exists():
            shutil.rmtree(final_gdir, ignore_errors=True)
        final_gdir.parent.mkdir(parents=True, exist_ok=True)
        import os
        os.replace(str(tmp_gdir), str(final_gdir))

    # ── re-publish local records LAST ───────────────────────────────────

    def _republish_records(self, record: DurableCommitRecord) -> None:
        """Coordinator rewrites run.json + the local global commit record.

        Done LAST so a concurrent recovery scan never sees a commit record
        before the generation's chunks have been published locally.
        """
        if not self.coord.is_coordinator:
            return
        # run.json: prefer the durable copy (it carries the real run metadata);
        # only write it if the local one is missing (lost work_dir).
        if not run_json_path(self.work_dir).exists():
            try:
                run_bytes = self.backend.get(dc.durable_run_json_key(self.run_id))
                rp = run_json_path(self.work_dir)
                rp.parent.mkdir(parents=True, exist_ok=True)
                import json
                write_json_atomic(rp, json.loads(run_bytes))
            except (OSError, ValueError, KeyError):
                pass  # run.json will be (re)written by the runner's init_run
        plan_key = dc.durable_plan_json_key(self.run_id)
        if self.backend.exists(plan_key) and \
                not (self.work_dir / "plan.json").exists():
            import json
            try:
                write_json_atomic(self.work_dir / "plan.json",
                                  json.loads(self.backend.get(plan_key)))
            except (OSError, ValueError):
                pass

        # The local global commit record for this generation (the local
        # durability point) — written last.
        local_commit = GlobalCommitRecord(
            generation=record.generation,
            n_ranks=record.n_ranks,
            circuit_hash=record.circuit_hash,
            step_index=record.step_index,
            parent_generation=record.parent_generation,
            rank_manifest_hashes={
                r: e.manifest_hash for r, e in record.ranks.items()
            },
            created=time.time(),
        )
        local_commit.write_atomic(commits_dir(self.work_dir))

    # ── public: restore the newest valid durable generation ─────────────

    def restore_latest(self, *, check_checksums: bool = True) -> RestoreResult:
        """Restore the newest valid durable generation into the local work_dir.

        Returns a :class:`RestoreResult`; ``generation is None`` means nothing
        durable was available (caller starts fresh).
        """
        # Step 2–3: coordinator picks the newest valid durable commit.
        record: DurableCommitRecord | None = None
        if self.coord.is_coordinator:
            record = self.latest_valid_durable_commit(
                check_checksums=check_checksums)
        record = self.coord.broadcast(record)
        if record is None:
            self.coord.barrier()
            return RestoreResult(generation=None)

        # Step 4–5: every rank restores + atomically publishes its partition.
        err: str | None = None
        try:
            self._restore_rank(record)
        except (RuntimeError, OSError, KeyError) as e:
            err = f"rank {self.coord.rank}: {e}"
        all_err = self.coord.gather(err)
        if self.coord.is_coordinator:
            failures = [e for e in all_err if e is not None]
        else:
            failures = []
        failures = self.coord.broadcast(failures)
        if failures:
            raise RuntimeError(
                f"durable restore of gen {record.generation} failed: {failures}")

        # Step 6: coordinator re-publishes run.json + the commit record LAST.
        self._republish_records(record)
        self.coord.barrier()

        log.info("durable restore: generation %d restored to %s",
                 record.generation, self.work_dir)
        return RestoreResult(generation=record.generation, record=record)
