"""Validation of a committed generation against what is actually on disk.

A generation is *valid* only if:
  - its global commit record is internally consistent (stored hash matches);
  - every rank named in the commit has a manifest.json on disk;
  - each manifest's recomputed content hash equals the hash recorded in the
    commit record (detects a tampered / torn / stale manifest);
  - every chunk file the manifest names exists and has the recorded size;
  - (optionally) every chunk file's sha256 matches the recorded checksum.

Any failure makes the generation invalid → recovery rolls back to an older
generation (or to a fresh start if none is valid).

Pure Python — no MPI, no numpy.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from wenbo_engine.recovery.global_commit import GlobalCommitRecord
from wenbo_engine.recovery.rank_manifest import RankManifest
from wenbo_engine.recovery.recovery_events import EventType, RecoveryEventLog
from wenbo_engine.recovery.generation_manager import (
    gen_dir, gen_chunks_dir, sha256_file,
)


@dataclass
class ValidationResult:
    valid: bool
    generation: int
    reason: str = ""
    failures: list[str] = field(default_factory=list)

    def fail(self, reason: str) -> "ValidationResult":
        self.valid = False
        self.failures.append(reason)
        if not self.reason:
            self.reason = reason
        return self


def validate_commit_record(
    record: GlobalCommitRecord,
    *,
    expected_circuit_hash: str | None = None,
    expected_n_ranks: int | None = None,
    events: RecoveryEventLog | None = None,
) -> ValidationResult:
    """Validate the commit record itself (no disk access beyond the record)."""
    ev = events or RecoveryEventLog(emit=False)
    res = ValidationResult(valid=True, generation=record.generation)

    if not record.verify_self_hash():
        res.fail("commit record self-hash mismatch (corrupted commit file)")
    try:
        record.validate()
    except ValueError as e:
        res.fail(str(e))
    if expected_n_ranks is not None and record.n_ranks != expected_n_ranks:
        res.fail(f"commit n_ranks={record.n_ranks} != run n_ranks={expected_n_ranks}")
    if expected_circuit_hash is not None and record.circuit_hash != expected_circuit_hash:
        res.fail(f"commit circuit_hash mismatch "
                 f"({record.circuit_hash} != {expected_circuit_hash})")

    if not res.valid:
        ev.emit(EventType.COMMIT_INVALID, res.reason, generation=record.generation)
    return res


def validate_rank_manifest(
    work_dir: str | Path,
    rank: int,
    generation: int,
    expected_hash: str,
    *,
    expected_circuit_hash: str | None = None,
    expected_parent_generation: int | None = None,
    expected_stage_id: int | None = None,
    check_sizes: bool = True,
    check_checksums: bool = False,
    events: RecoveryEventLog | None = None,
) -> ValidationResult:
    """Validate one rank's manifest + chunk files for a generation."""
    ev = events or RecoveryEventLog(emit=False)
    res = ValidationResult(valid=True, generation=generation)
    gdir = gen_dir(work_dir, rank, generation)

    if not RankManifest.exists(gdir):
        res.fail(f"rank {rank} gen {generation}: manifest missing")
        ev.emit(EventType.MANIFEST_MISSING, res.reason,
                generation=generation, rank=rank)
        return res

    try:
        man = RankManifest.read(gdir)
    except (ValueError, OSError) as e:
        res.fail(f"rank {rank} gen {generation}: manifest unreadable ({e})")
        ev.emit(EventType.MANIFEST_HASH_MISMATCH, res.reason,
                generation=generation, rank=rank)
        return res

    # Recomputed content hash must match both itself and the commit's record.
    if not man.verify_self_hash():
        res.fail(f"rank {rank} gen {generation}: manifest self-hash mismatch")
        ev.emit(EventType.MANIFEST_HASH_MISMATCH, res.reason,
                generation=generation, rank=rank)
        return res
    if man.manifest_hash != expected_hash:
        res.fail(f"rank {rank} gen {generation}: manifest hash "
                 f"{man.manifest_hash} != commit-recorded {expected_hash}")
        ev.emit(EventType.MANIFEST_HASH_MISMATCH, res.reason,
                generation=generation, rank=rank)
        return res

    if man.generation != generation or man.rank != rank:
        res.fail(f"rank {rank} gen {generation}: manifest identity mismatch "
                 f"(says rank={man.rank} gen={man.generation})")
        return res
    if expected_circuit_hash is not None and man.circuit_hash != expected_circuit_hash:
        res.fail(f"rank {rank} gen {generation}: circuit hash mismatch")
        return res

    # Lineage: every rank of a generation must share the same parent and stage
    # as the commit record.  Because each rank is checked against the SAME
    # record values, this also guarantees all ranks agree with each other —
    # the scanner never stitches together partitions with divergent lineage.
    if (expected_parent_generation is not None
            and man.parent_generation != expected_parent_generation):
        res.fail(f"rank {rank} gen {generation}: parent_generation "
                 f"{man.parent_generation} != commit {expected_parent_generation}")
        ev.emit(EventType.LINEAGE_MISMATCH, res.reason,
                generation=generation, rank=rank)
        return res
    if expected_stage_id is not None and man.stage_id != expected_stage_id:
        res.fail(f"rank {rank} gen {generation}: stage_id "
                 f"{man.stage_id} != commit {expected_stage_id}")
        ev.emit(EventType.LINEAGE_MISMATCH, res.reason,
                generation=generation, rank=rank)
        return res

    # Extent-backed generation: validate each chunk's extent slice instead of
    # a standalone chunk file (existence + size/offset coverage + checksum).
    if any(c.is_extent for c in man.chunks):
        from wenbo_engine.storage.extent_store import validate_extent_chunk
        from wenbo_engine.storage.extent_manifest import ExtentChunkRecord
        gdir = gen_dir(work_dir, rank, generation)
        for c in man.chunks:
            erec = ExtentChunkRecord(
                chunk_id=c.index, extent_id=c.extent_id,
                offset=c.extent_offset, length=c.size_bytes,
                checksum=c.checksum or "")
            reason = validate_extent_chunk(
                gdir, erec,
                check_checksum=check_checksums and c.checksum is not None)
            if reason:
                res.fail(f"rank {rank} gen {generation}: {reason}")
                ev.emit(EventType.CHUNK_MISSING, res.reason,
                        generation=generation, rank=rank)
        return res

    # Chunk files: existence + size (+ optional checksum).
    cdir = gen_chunks_dir(work_dir, rank, generation)
    for c in man.chunks:
        path = cdir / c.filename
        if not path.exists():
            res.fail(f"rank {rank} gen {generation}: chunk {c.filename} missing")
            ev.emit(EventType.CHUNK_MISSING, res.reason,
                    generation=generation, rank=rank, chunk=c.filename)
            continue
        if check_sizes:
            actual = path.stat().st_size
            if actual != c.size_bytes:
                res.fail(f"rank {rank} gen {generation}: chunk {c.filename} "
                         f"size {actual} != {c.size_bytes}")
                ev.emit(EventType.CHUNK_SIZE_MISMATCH, res.reason,
                        generation=generation, rank=rank, chunk=c.filename,
                        actual=actual, expected=c.size_bytes)
                continue
        if check_checksums and c.checksum is not None:
            digest = sha256_file(path)
            if digest != c.checksum:
                res.fail(f"rank {rank} gen {generation}: chunk {c.filename} "
                         f"checksum mismatch")
                ev.emit(EventType.CHUNK_CHECKSUM_MISMATCH, res.reason,
                        generation=generation, rank=rank, chunk=c.filename)
    return res


def validate_generation(
    work_dir: str | Path,
    record: GlobalCommitRecord,
    *,
    expected_circuit_hash: str | None = None,
    expected_n_ranks: int | None = None,
    check_sizes: bool = True,
    check_checksums: bool = False,
    events: RecoveryEventLog | None = None,
) -> ValidationResult:
    """Validate a full generation (commit record + every rank's manifest)."""
    ev = events or RecoveryEventLog(emit=False)
    gen = record.generation

    res = validate_commit_record(
        record,
        expected_circuit_hash=expected_circuit_hash,
        expected_n_ranks=expected_n_ranks,
        events=ev,
    )
    if not res.valid:
        ev.emit(EventType.GENERATION_REJECTED, res.reason, generation=gen)
        return res

    agg = ValidationResult(valid=True, generation=gen)
    for rank in range(record.n_ranks):
        expected_hash = record.rank_manifest_hashes.get(rank, "")
        r = validate_rank_manifest(
            work_dir, rank, gen, expected_hash,
            expected_circuit_hash=expected_circuit_hash,
            expected_parent_generation=record.parent_generation,
            expected_stage_id=record.step_index,
            check_sizes=check_sizes,
            check_checksums=check_checksums,
            events=ev,
        )
        if not r.valid:
            agg.valid = False
            agg.failures.extend(r.failures)
            if not agg.reason:
                agg.reason = r.reason

    if agg.valid:
        ev.emit(EventType.GENERATION_VALID, "generation validated",
                generation=gen)
    else:
        ev.emit(EventType.GENERATION_REJECTED, agg.reason, generation=gen)
    return agg
