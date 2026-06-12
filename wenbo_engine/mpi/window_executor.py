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
from time import perf_counter as _perf

import numpy as np

from wenbo_engine.kernel import gates as _gates
from wenbo_engine.mpi.diagonal_nonlocal import classify_nonlocal_gate
# Module-level so the benchmark can instrument window disk I/O (mirrors how the
# runner/overlay expose these names); the executor logic is unchanged.
from wenbo_engine.storage.block_store import (
    read_chunk, write_chunk_atomic, chunk_filename, DTYPE,
)

_GIB = 1 << 30
_ITEMSIZE = 8  # complex64

# The leader co-resides the rank group's data while applying the window gates.
# To stay within budget at any scale we GATHER IN SEGMENTS along the chunk (the
# 1q rank-bit gates act independently per amplitude offset, so segmenting is
# exact).  The leader holds ~``_LEADER_COPY_FACTOR`` arrays of
# ``group_size * seg_len`` (recv buffer, pattern-ordered buffer, send buffer,
# and one tensordot temporary), plus the rank's own src+dst chunk.
_LEADER_COPY_FACTOR = 4
_WINDOW_BUDGET_FRACTION = 0.5   # cap the leader's window working set at half budget


def plan_segment(chunk_size: int, group_size: int,
                 ram_budget_gib: float | None) -> tuple[int | None, float]:
    """Choose a gather segment length that keeps the leader within budget.

    Returns ``(seg_len, estimated_peak_gib)``.  ``seg_len`` is ``None`` when not
    even ``src + dst`` chunks fit the budget fraction (caller must fall back).
    With no budget, the whole chunk is one segment (legacy/unbounded).
    """
    chunk_bytes = chunk_size * _ITEMSIZE
    if ram_budget_gib is None or ram_budget_gib <= 0:
        peak = (2 * chunk_bytes + _LEADER_COPY_FACTOR * group_size
                * chunk_bytes) / _GIB
        return chunk_size, peak
    budget = _WINDOW_BUDGET_FRACTION * ram_budget_gib * _GIB
    base = 2 * chunk_bytes                       # rank's own src + dst chunk
    avail = budget - base
    if avail <= _LEADER_COPY_FACTOR * group_size * _ITEMSIZE:
        return None, base / _GIB                 # cannot fit even one segment
    seg_len = int(avail // (_LEADER_COPY_FACTOR * group_size * _ITEMSIZE))
    seg_len = max(1, min(chunk_size, seg_len))
    peak = (base + _LEADER_COPY_FACTOR * group_size * seg_len * _ITEMSIZE) / _GIB
    return seg_len, peak


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
    seg_len: int = 0          # gather segment length (amplitudes) keeping leader in budget
    chunk_size: int = 0

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
    # measured timing (calibrated-cost-model telemetry)
    gather_time: float = 0.0
    scatter_time: float = 0.0
    leader_compute_time: float = 0.0
    segment_time: float = 0.0


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
        if group_size > num_ranks:
            rejections.append((s0, s1, "rank group larger than communicator"))
            continue
        # Bound the leader's resident memory by gathering in segments.  Reject
        # (fall back) only if not even src+dst chunks fit the budget fraction.
        seg_len, est_ram = plan_segment(chunk_size, group_size, ram_budget_gib)
        if seg_len is None:
            rejections.append((s0, s1,
                               f"src+dst chunk ({2 * chunk_bytes / _GIB:.4f} GiB) "
                               f"exceeds window RAM budget fraction of "
                               f"ram_budget_gib={ram_budget_gib}"))
            continue
        rank_qubits = [b + n_local_bits + k for b in sorted_bits]
        windows.append(ExecutableWindow(
            start_step=s0, end_step=s1, rank_bits=sorted_bits,
            rank_qubits=rank_qubits, gates=gates, group_size=group_size,
            estimated_ram_gib=est_ram, seg_len=seg_len, chunk_size=chunk_size))
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

        seg_len = win.seg_len or chunk_size
        seg_bytes = seg_len * np.dtype(DTYPE).itemsize
        for ci in range(n_chunks_per_rank):
            my_chunk = read_chunk(src_chunks_dir / chunk_filename(ci))
            my_new = np.empty(chunk_size, dtype=DTYPE)
            # Gather/apply/scatter in segments so the leader's resident memory
            # stays within budget regardless of chunk size (gates act per
            # offset, so segmenting is exact).
            for a in range(0, chunk_size, seg_len):
                _seg_t0 = _perf()
                b = min(a + seg_len, chunk_size)
                slen = b - a
                seg = np.ascontiguousarray(my_chunk[a:b])
                recvbuf = (np.empty(G * slen, dtype=DTYPE)
                           if sub_rank == 0 else None)
                _g0 = _perf()
                sub.Gather(seg, recvbuf, root=0)
                _g1 = _perf()
                metrics.gather_time += _g1 - _g0
                if sub_rank == 0:
                    _c0 = _perf()
                    rows = recvbuf.reshape(G, slen)
                    by_pattern = np.empty((G, slen), dtype=DTYPE)
                    for sr in range(G):
                        by_pattern[patterns[sr]] = rows[sr]
                    out_by_pattern = apply_window_to_group_buffer(
                        by_pattern, sorted_bits, win.gates)
                    sendbuf = np.empty(G * slen, dtype=DTYPE)
                    send_rows = sendbuf.reshape(G, slen)
                    for sr in range(G):
                        send_rows[sr] = out_by_pattern[patterns[sr]]
                    metrics.leader_compute_time += _perf() - _c0
                    metrics.gather_bytes += (G - 1) * slen * np.dtype(DTYPE).itemsize
                    metrics.scatter_bytes += (G - 1) * slen * np.dtype(DTYPE).itemsize
                    metrics.window_sendrecv_count += 2   # one Gather + one Scatter
                else:
                    sendbuf = None
                seg_new = np.empty(slen, dtype=DTYPE)
                _s0 = _perf()
                sub.Scatter(sendbuf, seg_new, root=0)
                metrics.scatter_time += _perf() - _s0
                my_new[a:b] = seg_new
                metrics.segment_time += _perf() - _seg_t0
            write_chunk_atomic(dst_chunks_dir / chunk_filename(ci), my_new)
    finally:
        sub.Free()
