"""Named fault points in the generation-recovery commit protocol.

Each :class:`FaultPoint` names a precise position in the per-step commit
protocol (see :mod:`wenbo_engine.recovery.generation_manager` steps 1–13).
The :class:`~wenbo_engine.faults.fault_injector.FaultInjector` is invoked at
each named point and decides whether to fire a crash there.

These are *orchestration-layer* points only.  The numerical kernels never
reference them (rule: kernels do no I/O / MPI / WAL / commit), so no fault
point lives inside a kernel call.

Protocol-step mapping (commit producing generation g+1 from g)::

    FaultPoint                  protocol step           location
    ─────────────────────────── ─────────────────────── ──────────────────────
    BEFORE_STAGE                before step 1            top of per-step loop
    AFTER_READ                  after step 1 (read g)    after _apply_step reads
    AFTER_PARTIAL_WRITE         during step 2 (writing)  after the first chunk
    AFTER_ALL_WRITES            after step 2             all chunk tmp files
    AFTER_RENAME                after steps 3–4          chunks fsynced+renamed
    AFTER_MANIFEST_WRITE        after steps 5–6          manifest.tmp fsynced
    AFTER_MANIFEST_RENAME       after step 7             manifest.json in place
    AFTER_ALLGATHER_PREPARED    after step 8             coordinator gathered
    BEFORE_GLOBAL_COMMIT        before steps 9–11        commit record not yet
    AFTER_GLOBAL_COMMIT         after step 11            commit record on disk
    DURING_DURABLE_UPLOAD       step 12 (broadcast)      broadcast in flight
    BEFORE_DURABLE_COMMIT       before step 13 (install) about to install g+1

``DURING_DURABLE_UPLOAD`` / ``BEFORE_DURABLE_COMMIT`` are mapped onto the
broadcast / install phase of the *generation* commit (this task does NOT
implement durable/object-store checkpointing — those names are reused for the
final broadcast→install handoff so the full enum is defined and exercised).
"""
from __future__ import annotations

from enum import Enum


class FaultPoint(str, Enum):
    """A named, injectable position in the commit protocol."""

    BEFORE_STAGE = "BEFORE_STAGE"
    AFTER_READ = "AFTER_READ"
    AFTER_PARTIAL_WRITE = "AFTER_PARTIAL_WRITE"
    AFTER_ALL_WRITES = "AFTER_ALL_WRITES"
    AFTER_RENAME = "AFTER_RENAME"
    AFTER_MANIFEST_WRITE = "AFTER_MANIFEST_WRITE"
    AFTER_MANIFEST_RENAME = "AFTER_MANIFEST_RENAME"
    AFTER_ALLGATHER_PREPARED = "AFTER_ALLGATHER_PREPARED"
    BEFORE_GLOBAL_COMMIT = "BEFORE_GLOBAL_COMMIT"
    AFTER_GLOBAL_COMMIT = "AFTER_GLOBAL_COMMIT"
    DURING_DURABLE_UPLOAD = "DURING_DURABLE_UPLOAD"
    BEFORE_DURABLE_COMMIT = "BEFORE_DURABLE_COMMIT"

    def __str__(self) -> str:  # so f"{fp}" is the bare name, not FaultPoint.X
        return self.value


# Convenience: the full ordered list of every fault point (orchestration order).
ALL_FAULT_POINTS: tuple[FaultPoint, ...] = tuple(FaultPoint)

# The point at which a generation becomes globally durable.  Faults *before*
# this point must recover the previous generation; faults at/after it must
# recover the new generation.
GLOBAL_COMMIT_BOUNDARY = FaultPoint.AFTER_GLOBAL_COMMIT


def parse_fault_point(name: str | FaultPoint) -> FaultPoint:
    """Coerce a string/enum into a :class:`FaultPoint` (raises on unknown)."""
    if isinstance(name, FaultPoint):
        return name
    try:
        return FaultPoint(str(name))
    except ValueError as e:
        valid = ", ".join(fp.value for fp in FaultPoint)
        raise ValueError(
            f"unknown fault point {name!r} (valid: {valid})") from e
