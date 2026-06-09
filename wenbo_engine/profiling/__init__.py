"""Profiling for the out-of-core simulator.

Four small, additive measurement tools.  None of them touch the numerical
kernels, the WAL, or recovery logic — they only observe.

  * ``StageProfiler``  — one row per simulation step/stage with a fixed
    schema (read/write/kernel/mpi/commit/checksum seconds, bytes moved,
    op counts, recovery mode).  Its per-stage *handle* exposes thread-safe
    context-manager timers used by the runners, and can forward fine-grained
    events to an ``IOProfiler`` / ``MPIProfiler``.
  * ``IOProfiler``    — one row per disk read/write event.
  * ``MPIProfiler``   — one row per MPI exchange/collective (no-op when MPI
    is unavailable).
  * ``CalibrationRunner`` — micro-benchmarks the machine (NVMe bandwidth,
    fsync, rename, checksum, and — if MPI is present — Sendrecv / collective
    cost) and emits a ``cost_model`` dict.
"""
from __future__ import annotations

from wenbo_engine.profiling.stage_profiler import (
    StageProfiler, StageHandle, NullStageHandle, NULL_STAGE_HANDLE, STAGE_COLUMNS,
)
from wenbo_engine.profiling.io_profiler import IOProfiler, IO_COLUMNS
from wenbo_engine.profiling.mpi_profiler import MPIProfiler, MPI_COLUMNS
from wenbo_engine.profiling.calibration import CalibrationRunner

__all__ = [
    "StageProfiler",
    "StageHandle",
    "NullStageHandle",
    "NULL_STAGE_HANDLE",
    "STAGE_COLUMNS",
    "IOProfiler",
    "IO_COLUMNS",
    "MPIProfiler",
    "MPI_COLUMNS",
    "CalibrationRunner",
]
