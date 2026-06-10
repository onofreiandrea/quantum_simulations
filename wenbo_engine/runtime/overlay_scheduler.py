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


def _step_unit(steps, idx, rank, chunk_ids, gen, ram_budget_chunks,
               storage_layout, unit_id, fallback=False) -> ComputeUnit:
    s = steps[idx]
    active = sorted({q for grp in ("local_ops", "rank_nonlocal_ops",
                                   "mpi_nonlocal_ops")
                     for qs, _ in s[grp] for q in qs})
    return ComputeUnit(
        compute_unit_id=unit_id, kind="step", src_generation=gen - 1,
        dst_generation=gen, rank=rank, chunk_ids=chunk_ids, active_qubits=active,
        step=s, n_steps=1, step_range=(idx, idx + 1),
        ram_budget_chunks=ram_budget_chunks, storage_layout=storage_layout,
        fallback=fallback)


def build_compute_units(steps, *, rank: int, n_chunks_per_rank: int,
                        start_gen: int, ram_budget_chunks: int = 0,
                        storage_layout: str = "chunks",
                        min_gates: int = 4) -> list[ComputeUnit]:
    """Partition ``steps[start_index:]`` into compute units (adaptive).

    A run of consecutive local-only steps is fused into ONE ``local`` unit only
    if it has at least ``min_gates`` local gates; a shorter run falls back to
    one ``step`` unit per step (the existing pipelined per-step path), avoiding
    overlay overhead on fragments too small to benefit.  ``start_gen`` is the
    already-committed generation; units are numbered from ``start_gen + 1``.
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
            if len(fused) >= min_gates:
                gen += 1
                active = sorted({q for qs, _ in fused for q in qs})
                units.append(ComputeUnit(
                    compute_unit_id=len(units), kind="local",
                    src_generation=gen - 1, dst_generation=gen, rank=rank,
                    chunk_ids=chunk_ids, active_qubits=active, local_ops=fused,
                    n_steps=j - i, step_range=(i, j),
                    ram_budget_chunks=ram_budget_chunks,
                    storage_layout=storage_layout))
            else:
                # too short: fall back to the step path, one unit per step
                for s_idx in range(i, j):
                    gen += 1
                    units.append(_step_unit(
                        steps, s_idx, rank, chunk_ids, gen, ram_budget_chunks,
                        storage_layout, len(units), fallback=True))
            i = j
        else:
            gen += 1
            units.append(_step_unit(steps, i, rank, chunk_ids, gen,
                                    ram_budget_chunks, storage_layout, len(units)))
            i += 1
    return units


@dataclass
class OverlayMetrics:
    """Accumulated compute-unit / overlay profiling counters."""
    min_gates: int = 4
    compute_units_executed: int = 0    # fused local units only
    local_units: int = 0
    step_units: int = 0
    compute_unit_fallbacks: int = 0    # short local steps run via the step path
    overlay_load_count: int = 0
    overlay_writeback_count: int = 0
    overlay_bytes_read: int = 0
    overlay_bytes_written: int = 0
    total_local_gates: int = 0         # gates inside fused local units

    def record_unit(self, unit: ComputeUnit, overlay=None) -> None:
        if unit.kind == "local":
            self.local_units += 1
            self.compute_units_executed += 1
            self.total_local_gates += unit.gate_count
        else:
            self.step_units += 1
            if unit.fallback:
                self.compute_unit_fallbacks += 1
        if overlay is not None:
            self.overlay_load_count += overlay.load_count
            self.overlay_writeback_count += overlay.writeback_count
            self.overlay_bytes_read += overlay.bytes_read
            self.overlay_bytes_written += overlay.bytes_written

    @property
    def avg_gates_per_compute_unit(self) -> float:
        if self.compute_units_executed == 0:
            return 0.0
        return self.total_local_gates / self.compute_units_executed

    def to_dict(self) -> dict:
        return {
            "compute_unit_min_gates": self.min_gates,
            "compute_units_executed": self.compute_units_executed,
            "compute_unit_fallbacks": self.compute_unit_fallbacks,
            "local_units": self.local_units,
            "step_units": self.step_units,
            "overlay_load_count": self.overlay_load_count,
            "overlay_writeback_count": self.overlay_writeback_count,
            "overlay_bytes_read": self.overlay_bytes_read,
            "overlay_bytes_written": self.overlay_bytes_written,
            "avg_gates_per_compute_unit": round(self.avg_gates_per_compute_unit, 3),
            # back-compat alias
            "gates_per_compute_unit": round(self.avg_gates_per_compute_unit, 3),
        }
