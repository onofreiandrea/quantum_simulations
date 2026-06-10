"""Group compiled steps into compute units.

Consecutive **local-only** steps (no rank-nonlocal, no MPI-nonlocal gates) are
fused into one ``local`` :class:`ComputeUnit` — they all act within-chunk, so a
chunk can be loaded once, have every fused gate applied, and be written once.
A step that needs rank-nonlocal or MPI exchange becomes its own ``step`` unit
(run via the existing per-step path) — never fused, so MPI traffic is not
increased and exchange semantics are untouched.

Pure, deterministic, side-effect-free: every rank builds the identical unit
list from the identical compiled steps, so the collective commit cadence stays
aligned across ranks.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from wenbo_engine.runtime.compute_unit import ComputeUnit


def _is_local_only(step: dict) -> bool:
    return (not step.get("rank_nonlocal_ops") and not step.get("mpi_nonlocal_ops"))


def build_compute_units(steps, *, rank: int, n_chunks_per_rank: int,
                        start_gen: int, ram_budget_chunks: int = 0,
                        storage_layout: str = "chunks") -> list[ComputeUnit]:
    """Partition ``steps[start_index:]`` into compute units.

    ``start_gen`` is the generation already committed (units are numbered from
    ``start_gen + 1``).  Returns the units to execute, in order.
    """
    chunk_ids = list(range(n_chunks_per_rank))
    units: list[ComputeUnit] = []
    gen = start_gen
    i = 0
    while i < len(steps):
        if _is_local_only(steps[i]):
            j = i
            fused = []
            while j < len(steps) and _is_local_only(steps[j]):
                fused.extend(steps[j]["local_ops"])
                j += 1
            gen += 1
            active = sorted({q for qs, _ in fused for q in qs})
            units.append(ComputeUnit(
                compute_unit_id=len(units), kind="local",
                src_generation=gen - 1, dst_generation=gen, rank=rank,
                chunk_ids=chunk_ids, active_qubits=active, local_ops=fused,
                n_steps=j - i, step_range=(i, j),
                ram_budget_chunks=ram_budget_chunks, storage_layout=storage_layout))
            i = j
        else:
            gen += 1
            s = steps[i]
            active = sorted({q for grp in ("local_ops", "rank_nonlocal_ops",
                                           "mpi_nonlocal_ops")
                             for qs, _ in s[grp] for q in qs})
            units.append(ComputeUnit(
                compute_unit_id=len(units), kind="step",
                src_generation=gen - 1, dst_generation=gen, rank=rank,
                chunk_ids=chunk_ids, active_qubits=active, step=s,
                n_steps=1, step_range=(i, i + 1),
                ram_budget_chunks=ram_budget_chunks, storage_layout=storage_layout))
            i += 1
    return units


@dataclass
class OverlayMetrics:
    """Accumulated compute-unit / overlay profiling counters."""
    compute_units_executed: int = 0
    local_units: int = 0
    step_units: int = 0
    overlay_load_count: int = 0
    overlay_writeback_count: int = 0
    total_gates: int = 0

    def record_unit(self, unit: ComputeUnit, overlay=None) -> None:
        self.compute_units_executed += 1
        self.total_gates += unit.gate_count
        if unit.kind == "local":
            self.local_units += 1
        else:
            self.step_units += 1
        if overlay is not None:
            self.overlay_load_count += overlay.load_count
            self.overlay_writeback_count += overlay.writeback_count

    @property
    def gates_per_compute_unit(self) -> float:
        if self.compute_units_executed == 0:
            return 0.0
        return self.total_gates / self.compute_units_executed

    def to_dict(self) -> dict:
        return {
            "compute_units_executed": self.compute_units_executed,
            "local_units": self.local_units,
            "step_units": self.step_units,
            "overlay_load_count": self.overlay_load_count,
            "overlay_writeback_count": self.overlay_writeback_count,
            "gates_per_compute_unit": round(self.gates_per_compute_unit, 3),
        }
