"""Distributed generation-based recovery.

Defines committed progress by *rank manifests* plus a *global commit record*
rather than only by ``wal.json``.  A generation is recoverable only when a
global commit record names it and every rank's on-disk manifest + chunks
match what was committed.

This package is independent of the numerical kernels and of MPI: the protocol
is driven through a :class:`~wenbo_engine.recovery.generation_manager.Coordinator`
(``LocalCoordinator`` in-process, ``MPICoordinator`` under mpirun), so it is
fully unit-testable without a cluster.

The existing WAL recovery (``wenbo_engine.wal``) is unchanged; this is an
alternative ``--recovery=generation`` mode.
"""
from __future__ import annotations

from wenbo_engine.recovery.recovery_events import (
    EventType, RecoveryEvent, RecoveryEventLog,
)
from wenbo_engine.recovery.rank_manifest import ChunkRecord, RankManifest
from wenbo_engine.recovery.global_commit import (
    GlobalCommitRecord, commit_filename, list_commit_files,
)
from wenbo_engine.recovery.generation_validator import (
    ValidationResult, validate_commit_record, validate_rank_manifest,
    validate_generation,
)
from wenbo_engine.recovery.recovery_scanner import RecoveryScanner, ScanResult
from wenbo_engine.recovery.generation_manager import (
    GenerationManager, RunMetadata, Coordinator, LocalCoordinator,
    MPICoordinator, sha256_file, read_run_metadata,
    run_json_path, commits_dir, rank_dir, generations_dir, gen_dir,
    gen_chunks_dir, quarantine_dir,
)

# Recognized recovery modes for the --recovery CLI flag.
RECOVERY_MODES = ("none", "wal", "generation")

__all__ = [
    "RECOVERY_MODES",
    # events
    "EventType", "RecoveryEvent", "RecoveryEventLog",
    # dataclasses
    "ChunkRecord", "RankManifest", "GlobalCommitRecord",
    "commit_filename", "list_commit_files",
    # validation
    "ValidationResult", "validate_commit_record", "validate_rank_manifest",
    "validate_generation",
    # scanning
    "RecoveryScanner", "ScanResult",
    # manager + layout
    "GenerationManager", "RunMetadata", "Coordinator", "LocalCoordinator",
    "MPICoordinator", "sha256_file", "read_run_metadata",
    "run_json_path", "commits_dir", "rank_dir", "generations_dir", "gen_dir",
    "gen_chunks_dir", "quarantine_dir",
]
