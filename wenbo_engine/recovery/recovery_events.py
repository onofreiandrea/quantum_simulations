"""Structured recovery events.

Recovery decisions (which generation was chosen, why a generation was
rejected, what was quarantined) are recorded as structured events so they
can be asserted in tests and surfaced in logs/operator tooling.

This module is pure Python — no MPI, no numpy, no file I/O on import.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any

log = logging.getLogger(__name__)


class EventType(str, Enum):
    """Kinds of recovery events."""

    # Scan lifecycle
    SCAN_STARTED = "scan_started"
    RUN_METADATA_READ = "run_metadata_read"
    NO_COMMITS = "no_commits"

    # Per-commit decisions
    COMMIT_FOUND = "commit_found"
    COMMIT_INVALID = "commit_invalid"
    GENERATION_VALID = "generation_valid"
    GENERATION_REJECTED = "generation_rejected"

    # Per-rank / per-chunk validation failures
    MANIFEST_MISSING = "manifest_missing"
    MANIFEST_HASH_MISMATCH = "manifest_hash_mismatch"
    LINEAGE_MISMATCH = "lineage_mismatch"
    CHUNK_MISSING = "chunk_missing"
    CHUNK_SIZE_MISMATCH = "chunk_size_mismatch"
    CHUNK_CHECKSUM_MISMATCH = "chunk_checksum_mismatch"

    # Outcome
    RECOVERED = "recovered"
    ROLLBACK = "rollback"
    FRESH_START = "fresh_start"
    QUARANTINED = "quarantined"

    # Commit protocol (write path)
    GENERATION_PREPARED = "generation_prepared"
    GENERATION_COMMITTED = "generation_committed"
    GENERATION_INSTALLED = "generation_installed"
    COMMIT_ABORTED = "commit_aborted"

    # Fault injection (deterministic crash testing)
    FAULT_INJECTED = "fault_injected"


@dataclass
class RecoveryEvent:
    """A single recovery event with a type, message, and free-form details."""

    type: EventType
    message: str = ""
    generation: int | None = None
    rank: int | None = None
    details: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["type"] = self.type.value
        return d


class RecoveryEventLog:
    """Collects recovery events and mirrors them to the standard logger.

    Tests inspect ``events`` directly; production code reads the log lines.
    """

    # Event types that indicate a problem (logged at WARNING).
    _WARN_TYPES = frozenset({
        EventType.COMMIT_INVALID,
        EventType.GENERATION_REJECTED,
        EventType.MANIFEST_MISSING,
        EventType.MANIFEST_HASH_MISMATCH,
        EventType.LINEAGE_MISMATCH,
        EventType.CHUNK_MISSING,
        EventType.CHUNK_SIZE_MISMATCH,
        EventType.CHUNK_CHECKSUM_MISMATCH,
        EventType.ROLLBACK,
        EventType.QUARANTINED,
        EventType.COMMIT_ABORTED,
    })

    def __init__(self, *, logger: logging.Logger | None = None,
                 emit: bool = True):
        self.events: list[RecoveryEvent] = []
        self._log = logger or log
        self._emit = emit

    def emit(self, type: EventType, message: str = "", *,
             generation: int | None = None, rank: int | None = None,
             **details: Any) -> RecoveryEvent:
        ev = RecoveryEvent(
            type=type, message=message,
            generation=generation, rank=rank, details=details,
        )
        self.events.append(ev)
        if self._emit:
            level = logging.WARNING if type in self._WARN_TYPES else logging.INFO
            ctx = []
            if generation is not None:
                ctx.append(f"gen={generation}")
            if rank is not None:
                ctx.append(f"rank={rank}")
            prefix = f"[recovery:{type.value}]"
            suffix = f" ({', '.join(ctx)})" if ctx else ""
            self._log.log(level, "%s %s%s", prefix, message, suffix)
        return ev

    def of_type(self, type: EventType) -> list[RecoveryEvent]:
        return [e for e in self.events if e.type == type]

    def has(self, type: EventType) -> bool:
        return any(e.type == type for e in self.events)

    def __len__(self) -> int:
        return len(self.events)

    def __iter__(self):
        return iter(self.events)
