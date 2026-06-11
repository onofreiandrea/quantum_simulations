"""Cost model for multi-step MPI-exchange *windows* (analysis only).

This module predicts the cost of fusing several consecutive MPI-nonlocal steps
into a single **gather / apply / scatter** window, versus the current per-step
exchange path.  It is **pure** (no MPI, no I/O, no state) and **deterministic**,
so the window planner can score candidate windows offline before any executor
exists.

Nothing here changes runtime execution.  The point is to answer, honestly and
separately:

  * how many **MPI bytes** a window would save,
  * how many **Sendrecv calls** it would save,
  * how many **commits** it would save,
  * how much **extra RAM** it would cost (co-resident gathered region), and
  * how much **extra recomputation** a crash would cost (a window has one
    commit boundary, so a crash inside it re-does more work than a per-step
    run that commits after every step).

The "blended" baseline/window costs at the bottom exist only to rank
candidates; the *separated* quantities above are the truthful output and are
what the report surfaces.  See :mod:`wenbo_engine.mpi.window_planner`.

Bit/byte conventions match the runner:

* ``chunk_bytes = (1 << chunk_bits) * itemsize`` (complex64 → itemsize 8).
* The baseline gate-aware path re-fetches each remote chunk **once per step**
  (the remote-buffer cache is cleared at every step boundary), so the same
  remote chunk fetched across ``W`` adjacent steps costs ``W`` fetches.
* A window fetches the distinct remote chunks **once** (gather) and writes them
  back **once** (scatter): ``2 * distinct_remote_chunks`` chunk transfers.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

ITEMSIZE = 8  # complex64
_GIB = 1 << 30


def chunk_bytes(chunk_bits: int, itemsize: int = ITEMSIZE) -> int:
    """Bytes in one chunk: ``(1 << chunk_bits) * itemsize``."""
    return (1 << chunk_bits) * itemsize


@dataclass(frozen=True)
class CostWeights:
    """Transparent per-unit weights for the *blended* ranking score only.

    These do NOT affect the separated byte/sendrecv/commit/RAM/recompute
    figures the report relies on — they only combine them into a single number
    so candidate windows can be ordered.  Defaults are deliberately simple and
    documented; change them without changing any correctness claim.
    """
    byte_cost_per_gib: float = 1.0        # cost per GiB moved over MPI
    sendrecv_latency_cost: float = 0.05   # fixed cost per Sendrecv call
    commit_cost: float = 0.10             # cost per commit boundary
    ram_penalty_per_gib: float = 0.02     # cost per GiB of extra resident RAM
    recompute_cost_per_gate: float = 0.01  # cost per gate of expected redo work


# ── separated quantities ────────────────────────────────────────────────

def baseline_fetches(remote_gate_fetches: int) -> int:
    """Remote-chunk fetches the per-step path performs across the window.

    ``remote_gate_fetches`` is the sum, over every step in the window, of the
    distinct ``(partner, remote_chunk)`` pairs that step must fetch — i.e. what
    the per-step gate-aware path actually exchanges (cache cleared each step).
    """
    return int(remote_gate_fetches)


def window_fetches(distinct_remote_chunks: int) -> int:
    """Remote-chunk transfers a window performs: gather + scatter (×2)."""
    return 2 * int(distinct_remote_chunks)


def bytes_for(n_chunk_transfers: int, cbytes: int) -> int:
    return int(n_chunk_transfers) * int(cbytes)


def repeated_fetches_avoided(remote_gate_fetches: int,
                             distinct_remote_chunks: int) -> int:
    """Fetches the window removes by gathering each distinct chunk once.

    ``= max(0, baseline_fetches - distinct_remote_chunks)``.  This is the
    cross-step redundancy a window collapses (the gather replaces every repeat;
    the single scatter is counted separately as window overhead).
    """
    return max(0, int(remote_gate_fetches) - int(distinct_remote_chunks))


def sendrecv_reduction(baseline_sendrecv: int, window_sendrecv: int) -> int:
    """Sendrecv calls saved (never negative in a beneficial window)."""
    return int(baseline_sendrecv) - int(window_sendrecv)


def estimate_window_ram_gib(co_resident_chunks: int, cbytes: int,
                            overhead_factor: float = 1.0) -> float:
    """Peak extra RAM to hold the gathered region co-resident, in GiB.

    A window keeps ``distinct_remote_chunks`` remote chunks plus the matching
    local chunks resident at once (gather → apply all gates → scatter), so
    ``co_resident_chunks`` is typically ``2 * distinct_remote_chunks``.
    ``overhead_factor`` (>=1) covers send/recv staging buffers.
    """
    return co_resident_chunks * cbytes * overhead_factor / _GIB


def ram_feasible(estimated_ram_gib: float, ram_budget_gib: float | None) -> bool:
    """Rule 1: feasible only if estimated RAM <= budget (no budget → unknown)."""
    if ram_budget_gib is None:
        return False
    return estimated_ram_gib <= ram_budget_gib


def expected_recomputation_units(gates: int, commit_count: int) -> float:
    """Expected gate-work redone after a single crash, given commit cadence.

    Generation recovery replays from the last commit.  With ``commit_count``
    evenly-spaced commits over ``gates`` gates, a uniformly-timed crash loses,
    in expectation, half of one commit segment::

        segment = gates / commit_count
        expected_redo = segment / 2

    Fewer commits → larger segments → more expected redo.  Returns 0 when there
    is no work or no commit.
    """
    if gates <= 0 or commit_count <= 0:
        return 0.0
    segment = gates / commit_count
    return segment / 2.0


def recomputation_cost_increase(gates: int, commit_count_baseline: int,
                                commit_count_window: int) -> float:
    """Extra expected redo a window introduces by committing less often.

    ``>= 0`` whenever ``commit_count_window < commit_count_baseline`` (the
    window's larger segments cost more redo on crash) — this is the recovery
    risk the window trades for fewer messages.
    """
    base = expected_recomputation_units(gates, commit_count_baseline)
    win = expected_recomputation_units(gates, commit_count_window)
    return win - base


# ── blended scores (ranking only) ───────────────────────────────────────

def baseline_cost(*, mpi_bytes: int, sendrecv_count: int, commit_count: int,
                  recompute_units: float, weights: CostWeights) -> float:
    """per_step_mpi_exchange + per_step_io(commit) + expected_recomputation."""
    return (
        weights.byte_cost_per_gib * (mpi_bytes / _GIB)
        + weights.sendrecv_latency_cost * sendrecv_count
        + weights.commit_cost * commit_count
        + weights.recompute_cost_per_gate * recompute_units
    )


def window_cost(*, gather_bytes: int, scatter_bytes: int, sendrecv_count: int,
                commit_count: int, recompute_units: float,
                extra_ram_gib: float, weights: CostWeights) -> float:
    """gather + scatter + one_window_commit + recompute + extra_ram_penalty."""
    return (
        weights.byte_cost_per_gib * ((gather_bytes + scatter_bytes) / _GIB)
        + weights.sendrecv_latency_cost * sendrecv_count
        + weights.commit_cost * commit_count
        + weights.recompute_cost_per_gate * recompute_units
        + weights.ram_penalty_per_gib * extra_ram_gib
    )


def weights_to_dict(w: CostWeights) -> dict:
    return asdict(w)
