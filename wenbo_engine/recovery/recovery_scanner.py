"""Recovery scanner — find the newest valid committed generation.

Scan algorithm:
  1. Read run.json (run metadata; used to cross-check circuit hash / n_ranks).
  2. List commit records newest → oldest.
  3. For each commit, validate the commit JSON (self-hash + structure).
  4. Validate every rank's manifest named by the commit.
  5. Validate manifest hashes against the commit's recorded hashes.
  6. Validate chunk file sizes.
  7. Validate checksums (if enabled).
  8. Return the newest fully-valid generation.
  9. Quarantine generation directories newer than the chosen one (incomplete
     or never-committed state left by a crash mid-commit).

Hard invariant: with no commit record, there is no committed generation —
the scanner returns ``generation=None`` (caller starts fresh).

Pure Python — no MPI, no numpy.
"""
from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

from wenbo_engine.recovery.global_commit import (
    GlobalCommitRecord, list_commit_files,
)
from wenbo_engine.recovery.generation_validator import (
    validate_generation, validate_commit_record, validate_rank_manifest,
)
from wenbo_engine.recovery.recovery_events import EventType, RecoveryEventLog
from wenbo_engine.recovery.generation_manager import (
    commits_dir, gen_dir, quarantine_dir,
    read_run_metadata, GENERATIONS_DIRNAME,
)


@dataclass
class ScanResult:
    """Outcome of a recovery scan."""

    generation: int | None                 # newest valid committed gen, or None
    record: GlobalCommitRecord | None = None
    commit_path: Path | None = None
    quarantined: list[Path] = field(default_factory=list)
    inspected: int = 0                     # commit records examined
    events: RecoveryEventLog | None = None  # the scan's event log

    @property
    def recovered(self) -> bool:
        return self.generation is not None


def _gen_num(gen_dirname: str) -> int | None:
    if not gen_dirname.startswith("gen_"):
        return None
    try:
        return int(gen_dirname[4:])
    except ValueError:
        return None


def _rank_num(rank_dirname: str) -> int | None:
    if not rank_dirname.startswith("rank_"):
        return None
    try:
        return int(rank_dirname[5:])
    except ValueError:
        return None


class RecoveryScanner:
    def __init__(self, work_dir: str | Path, *,
                 events: RecoveryEventLog | None = None):
        self.work_dir = Path(work_dir)
        self.events = events or RecoveryEventLog()

    # ── main entry ──────────────────────────────────────────────────────

    def scan(self, *, quarantine: bool = True,
             check_sizes: bool = True,
             check_checksums: bool = False) -> ScanResult:
        ev = self.events
        ev.emit(EventType.SCAN_STARTED, "scanning for committed generations")

        meta = read_run_metadata(self.work_dir)
        expected_circuit_hash = meta.circuit_hash if meta else None
        expected_n_ranks = meta.n_ranks if meta else None
        if meta:
            ev.emit(EventType.RUN_METADATA_READ,
                    f"run.json: {meta.n_ranks} ranks, circuit {meta.circuit_hash}")

        commit_files = list_commit_files(commits_dir(self.work_dir))
        if not commit_files:
            # No global commit record ⇒ no committed generation.
            ev.emit(EventType.NO_COMMITS,
                    "no commit records found — starting fresh")
            result = ScanResult(generation=None, events=ev)
            if quarantine:
                # Everything on disk is uncommitted; quarantine it all.
                result.quarantined = self._quarantine_newer_than(-1)
            ev.emit(EventType.FRESH_START, "no committed generation available")
            return result

        chosen: ScanResult | None = None
        inspected = 0
        for path in reversed(commit_files):     # newest → oldest
            inspected += 1
            try:
                record = GlobalCommitRecord.read_file(path)
            except (ValueError, OSError) as e:
                ev.emit(EventType.COMMIT_INVALID,
                        f"unreadable commit file {path.name}: {e}")
                continue
            ev.emit(EventType.COMMIT_FOUND, f"examining {path.name}",
                    generation=record.generation)

            res = validate_generation(
                self.work_dir, record,
                expected_circuit_hash=expected_circuit_hash,
                expected_n_ranks=expected_n_ranks,
                check_sizes=check_sizes,
                check_checksums=check_checksums,
                events=ev,
            )
            if res.valid:
                chosen = ScanResult(
                    generation=record.generation,
                    record=record,
                    commit_path=path,
                    inspected=inspected,
                    events=ev,
                )
                break
            # else: roll back further to an older commit
            ev.emit(EventType.ROLLBACK,
                    f"generation {record.generation} invalid, rolling back",
                    generation=record.generation)

        if chosen is None:
            ev.emit(EventType.FRESH_START,
                    "no valid committed generation — starting fresh")
            result = ScanResult(generation=None, inspected=inspected, events=ev)
            if quarantine:
                result.quarantined = self._quarantine_newer_than(-1)
            return result

        ev.emit(EventType.RECOVERED,
                f"recovered generation {chosen.generation}",
                generation=chosen.generation)
        if quarantine:
            chosen.quarantined = self._quarantine_newer_than(chosen.generation)
        return chosen

    # ── quarantine ──────────────────────────────────────────────────────

    def _quarantine_newer_than(self, keep_generation: int) -> list[Path]:
        """Move every gen dir with number > keep_generation into quarantine.

        These are generations a crash left behind: chunks/manifest may have
        been written, but no valid commit covers them, so they are not
        recoverable state.  Moving (not deleting) preserves them for forensics.
        """
        moved: list[Path] = []
        for rank_path in sorted(self.work_dir.glob("rank_*")):
            rank = _rank_num(rank_path.name)
            if rank is None:
                continue
            gens = rank_path / GENERATIONS_DIRNAME
            if not gens.exists():
                continue
            for gdir in sorted(gens.glob("gen_*")):
                gnum = _gen_num(gdir.name)
                if gnum is None or gnum <= keep_generation:
                    continue
                dest = self._quarantine_dest(rank, gdir.name)
                shutil.move(str(gdir), str(dest))
                moved.append(dest)
                self.events.emit(
                    EventType.QUARANTINED,
                    f"quarantined uncommitted {gdir.name}",
                    generation=gnum, rank=rank, dest=str(dest),
                )
        return moved

    def _quarantine_dest(self, rank: int, name: str) -> Path:
        qdir = quarantine_dir(self.work_dir, rank)
        qdir.mkdir(parents=True, exist_ok=True)
        dest = qdir / name
        # Avoid clobbering a previous quarantine of the same generation.
        suffix = 1
        while dest.exists():
            dest = qdir / f"{name}.{suffix}"
            suffix += 1
        return dest


# ── distributed (node-local, multi-node) recovery scan ──────────────────

class DistributedRecoveryScanner:
    """Recovery scan that works with a *node-local* work_dir per rank.

    The single-host :class:`RecoveryScanner` assumes one rank (rank 0) can read
    every rank's manifest + chunks from a shared work_dir.  That holds for
    ``mpirun -np N`` on one host (shared FS) but FAILS on true multi-node where
    each rank only sees its own ``rank_<r>/`` subtree on node-local NVMe.

    This scanner fixes that.  The protocol (rank 0 == coordinator):

      1. Only the coordinator lists + reads the global commit records
         (``commits/`` is coordinator-visible) and validates each at the
         commit level (self-hash, n_ranks, circuit hash).  It broadcasts the
         ordered candidate records (newest → oldest) to all ranks.
      2. For each candidate, **every rank validates ONLY its own partition**
         (``validate_rank_manifest`` reads just ``rank_<r>/generations/...``):
         manifest present, self-hash ok, manifest hash == the hash recorded in
         the global commit for this rank, lineage (parent_generation, stage_id)
         == the commit's, chunk sizes (+ checksums if enabled).
      3. Per-rank pass/fail is ``gather``-ed to the coordinator and the AND is
         ``broadcast`` back, so a generation is accepted only if **all** ranks
         validated their own slice — otherwise all ranks reject in lockstep and
         try the previous commit.

    Guarantees the required invariants:
      * no valid global commit record ⇒ no committed generation (rank 0 still
        owns the commit records);
      * every rank recovers the SAME generation id (the decision is a single
        broadcast value);
      * no mixed-generation state — a rank whose local manifest is a different
        generation/lineage fails the hash/lineage check, so the whole
        generation is rejected;
      * no shared work_dir assumption — each rank only ever reads its own
        rank dir.

    Degenerates correctly to the single-rank case under ``LocalCoordinator``.
    """

    def __init__(self, work_dir, coord, events: RecoveryEventLog | None = None):
        self.work_dir = Path(work_dir)
        self.coord = coord
        self.events = events or RecoveryEventLog(emit=False)

    def find_latest_valid_generation(
        self, *, check_sizes: bool = True, check_checksums: bool = False,
        quarantine: bool = True,
    ) -> ScanResult:
        coord = self.coord
        ev = self.events
        rank = coord.rank

        # ── 1. coordinator reads + commit-validates every record, broadcasts ──
        if coord.is_coordinator:
            ev.emit(EventType.SCAN_STARTED,
                    "distributed scan for committed generations")
            meta = read_run_metadata(self.work_dir)
            exp_circ = meta.circuit_hash if meta else None
            cands = []
            for p in reversed(list_commit_files(commits_dir(self.work_dir))):
                try:
                    rec = GlobalCommitRecord.read_file(p)
                except (ValueError, OSError) as e:
                    ev.emit(EventType.COMMIT_INVALID,
                            f"unreadable commit file {p.name}: {e}")
                    continue
                cres = validate_commit_record(
                    rec, expected_circuit_hash=exp_circ,
                    expected_n_ranks=coord.n_ranks, events=ev)
                cands.append({"rec": rec, "path": str(p),
                              "commit_ok": cres.valid})
            payload = {"exp_circ": exp_circ, "cands": cands}
        else:
            payload = None
        payload = coord.broadcast(payload)
        exp_circ = payload["exp_circ"]
        cands = payload["cands"]

        # ── 2-3. per-candidate: each rank validates its own slice; AND-combine ──
        chosen: ScanResult | None = None
        inspected = 0
        for c in cands:
            inspected += 1
            if not c["commit_ok"]:
                # commit-level invalid: all ranks skip in lockstep (no collective)
                continue
            rec = c["rec"]
            expected_hash = rec.rank_manifest_hashes.get(rank, "")
            local = validate_rank_manifest(
                self.work_dir, rank, rec.generation, expected_hash,
                expected_circuit_hash=exp_circ,
                expected_parent_generation=rec.parent_generation,
                expected_stage_id=rec.step_index,
                check_sizes=check_sizes, check_checksums=check_checksums,
                events=ev,
            )
            local_ok = bool(local.valid)
            gathered = coord.gather(local_ok)
            all_ok = all(gathered) if coord.is_coordinator else None
            all_ok = coord.broadcast(all_ok)
            if all_ok:
                chosen = ScanResult(
                    generation=rec.generation, record=rec,
                    commit_path=Path(c["path"]), inspected=inspected, events=ev)
                break
            ev.emit(EventType.ROLLBACK,
                    f"generation {rec.generation} invalid on >=1 rank, "
                    f"rolling back", generation=rec.generation)

        # ── result + per-rank quarantine of newer uncommitted gens ──
        if chosen is None:
            ev.emit(EventType.FRESH_START,
                    "no valid committed generation across all ranks")
            result = ScanResult(generation=None, inspected=inspected, events=ev)
            keep = -1
        else:
            ev.emit(EventType.RECOVERED,
                    f"recovered generation {chosen.generation} (all ranks)",
                    generation=chosen.generation)
            result = chosen
            keep = chosen.generation

        if quarantine:
            result.quarantined = self._quarantine_local_newer_than(rank, keep)
        coord.barrier()
        return result

    def _quarantine_local_newer_than(self, rank: int, keep_generation: int):
        """Quarantine THIS rank's gen dirs newer than ``keep_generation``.

        Scoped to the local rank only (each rank owns its node-local dir), so
        on a shared FS multiple ranks never fight over the same directories.
        """
        moved = []
        gens = gen_dir(self.work_dir, rank, 0).parent  # rank_<r>/generations
        if not gens.exists():
            return moved
        for gdir in sorted(gens.glob("gen_*")):
            gnum = _gen_num(gdir.name)
            if gnum is None or gnum <= keep_generation:
                continue
            qdir = quarantine_dir(self.work_dir, rank)
            qdir.mkdir(parents=True, exist_ok=True)
            dest = qdir / gdir.name
            suffix = 1
            while dest.exists():
                dest = qdir / f"{gdir.name}.{suffix}"
                suffix += 1
            shutil.move(str(gdir), str(dest))
            moved.append(dest)
            self.events.emit(
                EventType.QUARANTINED,
                f"quarantined uncommitted {gdir.name} (rank {rank})",
                generation=gnum, rank=rank, dest=str(dest))
        return moved


def find_latest_valid_generation_mpi(work_dir, comm, *,
                                     check_sizes: bool = True,
                                     check_checksums: bool = False,
                                     quarantine: bool = True,
                                     events: RecoveryEventLog | None = None) -> ScanResult:
    """Convenience wrapper: distributed scan over an mpi4py communicator."""
    from wenbo_engine.recovery.generation_manager import MPICoordinator
    scanner = DistributedRecoveryScanner(
        work_dir, MPICoordinator(comm), events=events)
    return scanner.find_latest_valid_generation(
        check_sizes=check_sizes, check_checksums=check_checksums,
        quarantine=quarantine)
