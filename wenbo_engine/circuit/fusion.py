"""Gate fusion and level batching for I/O reduction.

Core optimisation: matmul is far faster than I/O (especially on cluster
storage), so maximise compute per chunk read-write cycle.

Two techniques:
  1. Level batching  – consecutive all-local levels are applied in one
     chunk pass (read once → apply all gates → write once).
  2. 1Q gate fusion  – consecutive 1-qubit gates on the same qubit
     (with no intervening 2Q gate on that qubit) are pre-multiplied
     into a single 2×2 matrix.
"""
from __future__ import annotations

import numpy as np

from wenbo_engine.kernel import gates as gmod


# ── helpers ──────────────────────────────────────────────────────────

def _compile_ops(
    level_gates: list[dict], k: int,
) -> tuple[list[tuple[list[int], np.ndarray]],
           list[tuple[list[int], np.ndarray]]]:
    """Convert gate dicts → (qubits, U) tuples, split local / non-local."""
    local: list[tuple[list[int], np.ndarray]] = []
    nonlocal_: list[tuple[list[int], np.ndarray]] = []
    for g in level_gates:
        U = gmod.gate_matrix(g["gate"], g["params"])
        qs = g["qubits"]
        if all(q < k for q in qs):
            local.append((qs, U))
        else:
            nonlocal_.append((qs, U))
    return local, nonlocal_


# ── 1Q fusion ────────────────────────────────────────────────────────

def fuse_1q_ops(
    ops: list[tuple[list[int], np.ndarray]],
) -> list[tuple[list[int], np.ndarray]]:
    """Fuse consecutive 1Q gates on the same qubit into one 2×2 matrix.

    If qubit q sees 1Q gates U0, U1, U2 with no 2Q gate on q in between,
    they are replaced by a single (q, U2 @ U1 @ U0).

    2Q gates flush any pending 1Q fusion on their qubits to preserve
    correctness of the application order.
    """
    if not ops:
        return ops

    pending: dict[int, np.ndarray] = {}   # qubit → accumulated 2×2
    result: list[tuple[list[int], np.ndarray]] = []

    def _flush_qubit(q: int) -> None:
        if q in pending:
            result.append(([q], pending.pop(q)))

    def _flush_all() -> None:
        for q in sorted(pending):
            result.append(([q], pending[q]))
        pending.clear()

    for qubits, U in ops:
        if len(qubits) == 1:
            q = qubits[0]
            if q in pending:
                pending[q] = U @ pending[q]          # compose: new @ old
            else:
                pending[q] = U.copy()
        else:
            # 2Q gate: flush pending 1Q ops on involved qubits first
            for q in qubits:
                _flush_qubit(q)
            result.append((qubits, U))

    _flush_all()
    return result


# ── level batching ───────────────────────────────────────────────────

def batch_levels(
    levels: list[list[dict]], k: int,
) -> list[dict]:
    """Batch consecutive all-local levels into fused passes.

    Returns a list of *pass* dicts:
        {
          "local_ops":    [(qubits, U), ...],   # fused local ops
          "nonlocal_ops": [(qubits, U), ...],   # non-local ops (unfused)
          "level_indices": [int, ...],           # original level numbers
        }

    A pass with only local_ops needs **one** chunk read-write cycle
    instead of one per original level.

    A pass with nonlocal_ops is always a single original level (no batching).
    """
    passes: list[dict] = []
    pending_local: list[tuple[list[int], np.ndarray]] = []
    pending_indices: list[int] = []

    for lv_idx, level_gates in enumerate(levels):
        if not level_gates:
            continue

        local, nonlocal_ = _compile_ops(level_gates, k)

        if nonlocal_:
            # Flush any accumulated local-only batch first
            if pending_local:
                passes.append({
                    "local_ops": fuse_1q_ops(pending_local),
                    "nonlocal_ops": [],
                    "level_indices": list(pending_indices),
                })
                pending_local = []
                pending_indices = []
            # Emit this level as a single pass (it has non-local gates)
            passes.append({
                "local_ops": local,
                "nonlocal_ops": nonlocal_,
                "level_indices": [lv_idx],
            })
        else:
            # All-local level → accumulate into current batch
            pending_local.extend(local)
            pending_indices.append(lv_idx)

    # Flush remaining
    if pending_local:
        passes.append({
            "local_ops": fuse_1q_ops(pending_local),
            "nonlocal_ops": [],
            "level_indices": list(pending_indices),
        })

    return passes


def fusion_stats(levels: list[list[dict]], k: int) -> dict:
    """Return stats about what fusion achieves (for benchmarking)."""
    passes = batch_levels(levels, k)
    n_original_levels = sum(1 for lv in levels if lv)
    n_passes = len(passes)
    n_local_only = sum(1 for p in passes if not p["nonlocal_ops"])
    total_ops_before = sum(len(lv) for lv in levels)
    total_ops_after = sum(
        len(p["local_ops"]) + len(p["nonlocal_ops"]) for p in passes
    )
    return {
        "original_levels": n_original_levels,
        "fused_passes": n_passes,
        "local_only_passes": n_local_only,
        "io_reduction": (
            f"{n_original_levels}→{n_passes} "
            f"({(1 - n_passes / max(n_original_levels, 1)) * 100:.0f}% fewer)"
        ),
        "ops_before": total_ops_before,
        "ops_after": total_ops_after,
    }
