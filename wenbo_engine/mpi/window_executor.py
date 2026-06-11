"""MPI true-mixing window executor (gather / apply / scatter).

The window feasibility planner predicts that fusing a run of consecutive
true-mixing MPI steps into one *gather → apply → scatter* window saves Sendrecv
calls, MPI bytes and commits.  This module *executes* such a window — narrowly,
safely, and only when it is provably correct and RAM-feasible.

**Correctness model (NOT a cross-step cache).**  A window is executed by making
the needed amplitude region *co-resident*, applying all the window's gates to
it, then scattering the updated chunks back.  No remote chunk is ever cached
across steps; each window does a fresh gather and a fresh scatter, then commits
exactly one generation.  This is the only correct way to fuse the steps — the
amplitudes a later gate needs have been mutated by earlier gates, so stale
remote copies would be wrong.

**Scope (kept deliberately narrow).**  A window is executable only when every
step it spans contains *only* single-qubit, non-diagonal (true-mixing) gates on
**rank** bits — no local ops, no rank-nonlocal ops, no permutation gates, no
2-qubit gates, no diagonal MPI gates.  ``mpi_nonlocal_mixing_heavy`` satisfies
this; ``mpi_nonlocal_phase_heavy`` (all diagonal → no candidate windows) and
``mpi_nonlocal_heavy`` (permutation / mixed) do not, and fall back to the
existing gate-aware per-step path.

Why this is exact: the window's gates act only on the ``R`` rank qubits.  The
amplitudes that differ only in the ``R`` bits — at a fixed (local chunk index,
offset) — form an invariant 2^|R| subspace distributed one-per-rank across the
*rank group* (ranks sharing all non-``R`` bits).  Gathering that group's
matching chunk onto a leader, applying the window's 1q gates along the ``R``
axes, and scattering the rows back reproduces, bit-for-bit, the result of
applying the same gates step-by-step with per-step exchange.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from wenbo_engine.kernel import gates as _gates
from wenbo_engine.mpi.diagonal_nonlocal import classify_nonlocal_gate

_GIB = 1 << 30
_ITEMSIZE = 8  # complex64


@dataclass
class ExecutableWindow:
    """A window the executor will run as one gather/apply/scatter + commit."""
    start_step: int
    end_step: int
    rank_bits: list           # sorted rank-bit positions (q-k-n_local_bits)
    rank_qubits: list         # sorted global rank-qubit indices (parallel to rank_bits)
    gates: list               # [(rank_bit_position, U), ...] in execution order
    group_size: int           # 2^len(rank_bits)
    estimated_ram_gib: float

    @property
    def n_steps(self) -> int:
        return self.end_step - self.start_step + 1

    @property
    def n_gates(self) -> int:
        return len(self.gates)


@dataclass
class WindowExecMetrics:
    """Per-rank window-executor counters (aggregated across ranks at the end)."""
    windows_executed: int = 0
    window_steps_executed: int = 0
    window_gates_executed: int = 0
    gather_bytes: int = 0
    scatter_bytes: int = 0
    window_sendrecv_count: int = 0
    commits_saved: int = 0
    estimated_ram_gib: float = 0.0
    fallbacks: int = 0
    fallback_reasons: list = field(default_factory=list)


def _is_single_qubit_true_mixing(qs, U) -> bool:
    if len(qs) != 1:
        return False
    kind, _ = classify_nonlocal_gate(U)
    return kind == "true_mixing"


def _step_is_pure_mixing_mpi(step, k: int, n_local_bits: int) -> tuple[bool, str]:
    """Whether a compiled step is a pure single-qubit true-mixing MPI step.

    Returns (ok, reason).  Rejects steps with local ops, rank-nonlocal ops, or
    any MPI op that is not a 1q true-mixing gate on a rank bit.
    """
    if step["local_ops"]:
        return False, "step has local ops"
    if step["rank_nonlocal_ops"]:
        return False, "step has rank-nonlocal ops"
    if not step["mpi_nonlocal_ops"]:
        return False, "step has no MPI gates"
    for qs, U in step["mpi_nonlocal_ops"]:
        if len(qs) != 1:
            return False, "multi-qubit MPI gate"
        q = qs[0]
        if (q - k) < n_local_bits:
            return False, "MPI gate qubit is not a rank bit"
        kind, _ = classify_nonlocal_gate(U)
        if kind == "diagonal":
            return False, "diagonal MPI gate (handled by fast path)"
        if kind == "permutation":
            return False, "permutation MPI gate (unsupported)"
        if kind != "true_mixing":
            return False, f"unsupported gate kind {kind!r}"
    return True, ""


def plan_executable_windows(steps, k: int, n_local_bits: int, num_ranks: int,
                            chunk_size: int, ram_budget_gib: float | None,
                            *, min_window_steps: int = 2,
                            ram_overhead_factor: float = 2.0):
    """Deterministically select windows the executor can run safely.

    Walks the compiled steps in order, grouping maximal runs of consecutive
    pure single-qubit true-mixing MPI steps (length >= ``min_window_steps``).
    Each run is accepted only if its rank group fits the RAM budget; otherwise
    it is returned as a rejected window (so the caller records a fallback).

    Returns ``(windows, rejections)`` where ``windows`` is a list of
    :class:`ExecutableWindow` and ``rejections`` is a list of
    ``(start_step, end_step, reason)``.
    """
    chunk_bytes = chunk_size * _ITEMSIZE
    windows: list[ExecutableWindow] = []
    rejections: list[tuple[int, int, str]] = []

    # maximal runs of pure-mixing steps
    runs: list[tuple[int, int]] = []
    start = None
    for i, s in enumerate(steps):
        ok, _ = _step_is_pure_mixing_mpi(s, k, n_local_bits)
        if ok:
            if start is None:
                start = i
        else:
            if start is not None:
                runs.append((start, i - 1))
                start = None
    if start is not None:
        runs.append((start, len(steps) - 1))

    for (s0, s1) in runs:
        if (s1 - s0 + 1) < min_window_steps:
            continue                       # single step: nothing to fuse
        rank_bits: set[int] = set()
        gates: list[tuple[int, np.ndarray]] = []
        for si in range(s0, s1 + 1):
            for qs, U in steps[si]["mpi_nonlocal_ops"]:
                rb = (qs[0] - k) - n_local_bits
                rank_bits.add(rb)
                gates.append((rb, U))
        sorted_bits = sorted(rank_bits)
        group_size = 1 << len(sorted_bits)
        # leader holds recv (group_size chunks) + send (group_size chunks)
        est_ram = ram_overhead_factor * group_size * chunk_bytes / _GIB
        if ram_budget_gib is not None and est_ram > ram_budget_gib:
            rejections.append((s0, s1,
                               f"estimated_ram_gib={est_ram:.4f} exceeds "
                               f"ram_budget_gib={ram_budget_gib}"))
            continue
        if group_size > num_ranks:
            rejections.append((s0, s1, "rank group larger than communicator"))
            continue
        rank_qubits = [b + n_local_bits + k for b in sorted_bits]
        windows.append(ExecutableWindow(
            start_step=s0, end_step=s1, rank_bits=sorted_bits,
            rank_qubits=rank_qubits, gates=gates, group_size=group_size,
            estimated_ram_gib=est_ram))
    return windows, rejections


def _rank_pattern(rank: int, sorted_bits: list) -> int:
    """Compose the rank's R-bit pattern, MSB-first by sorted bit position."""
    m = len(sorted_bits)
    pat = 0
    for idx, b in enumerate(sorted_bits):
        bit = (rank >> b) & 1
        pat |= bit << (m - 1 - idx)
    return pat


def _apply_1q_on_axis(arr: np.ndarray, axis: int, U: np.ndarray) -> np.ndarray:
    """Apply a 2x2 gate along ``axis`` (size 2): new[i] = sum_j U[i,j] old[j]."""
    moved = np.tensordot(U, arr, axes=([1], [axis]))   # result axis 0 is new index
    return np.moveaxis(moved, 0, axis)


def apply_window_to_group_buffer(buf_by_pattern: np.ndarray,
                                 sorted_bits: list, gates) -> np.ndarray:
    """Apply the window's 1q gates to a (2^m, chunk_size) group buffer.

    ``buf_by_pattern`` is indexed by R-pattern integer on axis 0.  Pure: returns
    a new array.  Used both by the MPI executor and by unit tests (no MPI).
    """
    m = len(sorted_bits)
    cs = buf_by_pattern.shape[-1]
    arr = buf_by_pattern.reshape((2,) * m + (cs,))
    bit_axis = {b: idx for idx, b in enumerate(sorted_bits)}
    for rb, U in gates:
        arr = _apply_1q_on_axis(arr, bit_axis[rb], np.asarray(U))
    return arr.reshape((1 << m, cs))


def execute_window(comm, rank: int, win: ExecutableWindow, src_chunks_dir,
                   dst_chunks_dir, n_chunks_per_rank: int, chunk_size: int,
                   metrics: WindowExecMetrics) -> None:
    """Run one window: gather group region, apply gates, scatter, write dst.

    Leader-based per local chunk index: the group's leader gathers the matching
    chunk from every group member, applies the window gates on the co-resident
    2^m amplitudes, and scatters the updated rows back.  Each rank writes its
    own updated chunk into ``dst_chunks_dir``.  No cross-step cache; the gather
    is fresh for this window only.
    """
    from wenbo_engine.storage.block_store import (
        read_chunk, write_chunk_atomic, chunk_filename, DTYPE,
    )
    sorted_bits = win.rank_bits
    m = len(sorted_bits)
    # color = rank with R-bits cleared → members share all non-R bits.
    color = rank
    for b in sorted_bits:
        color &= ~(1 << b)
    sub = comm.Split(color, rank)
    try:
        G = sub.Get_size()
        sub_rank = sub.Get_rank()
        assert G == win.group_size, (G, win.group_size)
        patterns = sub.allgather(_rank_pattern(rank, sorted_bits))
        # subrank for each pattern (inverse map), to reorder gather/scatter rows.
        pat_to_subrank = {p: i for i, p in enumerate(patterns)}
        my_pattern = _rank_pattern(rank, sorted_bits)
        chunk_bytes = chunk_size * np.dtype(DTYPE).itemsize

        for ci in range(n_chunks_per_rank):
            my_chunk = read_chunk(src_chunks_dir / chunk_filename(ci))
            recvbuf = (np.empty(G * chunk_size, dtype=DTYPE)
                       if sub_rank == 0 else None)
            sub.Gather(np.ascontiguousarray(my_chunk), recvbuf, root=0)
            if sub_rank == 0:
                rows = recvbuf.reshape(G, chunk_size)
                # reorder subrank-order rows into pattern-order
                by_pattern = np.empty((G, chunk_size), dtype=DTYPE)
                for sr in range(G):
                    by_pattern[patterns[sr]] = rows[sr]
                out_by_pattern = apply_window_to_group_buffer(
                    by_pattern, sorted_bits, win.gates)
                # back to subrank order for Scatter
                sendbuf = np.empty(G * chunk_size, dtype=DTYPE)
                send_rows = sendbuf.reshape(G, chunk_size)
                for sr in range(G):
                    send_rows[sr] = out_by_pattern[patterns[sr]]
                metrics.gather_bytes += (G - 1) * chunk_bytes
                metrics.scatter_bytes += (G - 1) * chunk_bytes
                metrics.window_sendrecv_count += 2   # one Gather + one Scatter
            else:
                sendbuf = None
            my_new = np.empty(chunk_size, dtype=DTYPE)
            sub.Scatter(sendbuf, my_new, root=0)
            write_chunk_atomic(dst_chunks_dir / chunk_filename(ci), my_new)
    finally:
        sub.Free()
