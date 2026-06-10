"""ComputeUnit: a group of compatible gates applied over a resident chunk set.

A *local* compute unit fuses the local-only gates of one or more consecutive
circuit steps so that each logical chunk is read once, has all the unit's gates
applied while resident in a :class:`MemoryOverlay`, and is written back once —
instead of one read+write per step.  Steps that need rank-nonlocal or MPI
exchange are NOT fused (they run via the existing per-step path); this keeps
correctness and the MPI/recovery semantics unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from wenbo_engine.runtime.memory_overlay import MemoryOverlay


@dataclass
class ComputeUnit:
    compute_unit_id: int
    kind: str                       # "local" (fused) | "step" (has nonlocal/MPI)
    src_generation: int
    dst_generation: int
    rank: int
    chunk_ids: list[int]
    active_qubits: list[int] = field(default_factory=list)
    local_ops: list = field(default_factory=list)   # fused (qs, U) for local units
    step: dict | None = None        # the raw compiled step for "step" units
    n_steps: int = 1                # how many circuit steps this unit covers
    step_range: tuple[int, int] = (0, 1)
    ram_budget_chunks: int = 0
    storage_layout: str = "chunks"
    recovery_boundary: bool = True  # every unit boundary is a commit/recovery point

    @property
    def gate_count(self) -> int:
        if self.kind == "local":
            return len(self.local_ops)
        s = self.step or {}
        return (len(s.get("local_ops", [])) + len(s.get("rank_nonlocal_ops", []))
                + len(s.get("mpi_nonlocal_ops", [])))


def execute_local_unit(unit: ComputeUnit, src_chunks_dir, dst_chunks_dir,
                       apply_local_ops) -> MemoryOverlay:
    """Run a local compute unit through a memory overlay.

    For each chunk: load once → apply all the unit's fused local gates →
    write back once.  ``apply_local_ops(array, ops)`` is the runner's existing
    kernel dispatcher (kernels are not modified).  Returns the overlay so the
    caller can read its load/writeback counters.
    """
    overlay = MemoryOverlay(src_chunks_dir, dst_chunks_dir,
                            ram_budget_chunks=unit.ram_budget_chunks)
    for cid in unit.chunk_ids:
        arr = overlay.get(cid)
        apply_local_ops(arr, unit.local_ops)
        overlay.mark_dirty(cid)
        overlay.writeback(cid)          # one write per chunk for the whole unit
    return overlay
