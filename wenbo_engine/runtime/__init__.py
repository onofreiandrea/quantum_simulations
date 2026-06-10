"""Runtime memory-overlay + compute-unit execution (SnuQS/QDAO-style).

Instead of read-chunk → apply-one-gate → write-chunk per circuit step, the
compute-unit executor loads a chunk into a RAM *overlay*, applies several
compatible (local) gates while it is resident, and writes it back once —
reducing out-of-core read/write passes.  Opt-in via ``--execution-mode
compute_unit`` (default ``step``); kernels, recovery, MPI exchange, and both
storage layouts are unchanged.
"""
from __future__ import annotations

from wenbo_engine.runtime.memory_overlay import MemoryOverlay
from wenbo_engine.runtime.compute_unit import ComputeUnit, execute_local_unit
from wenbo_engine.runtime.overlay_scheduler import (
    build_compute_units, OverlayMetrics,
)

__all__ = [
    "MemoryOverlay",
    "ComputeUnit",
    "execute_local_unit",
    "build_compute_units",
    "OverlayMetrics",
]
