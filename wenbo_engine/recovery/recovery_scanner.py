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
from wenbo_engine.recovery.generation_validator import validate_generation
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
