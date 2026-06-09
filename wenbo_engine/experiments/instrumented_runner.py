"""Instrumented single-node out-of-core runner (for experiments).

This is the same double-buffer algorithm as
``wenbo_engine.runner.single_node`` — two state directories alternate as
source/destination, the source is never mutated, writes are atomic, and the
**existing** :class:`wenbo_engine.wal.wal.WAL` provides crash recovery — but
every read / kernel / write / commit / checksum region is wrapped in a
:class:`~wenbo_engine.profiling.stage_profiler.StageProfiler` timer so the
experiment harness can emit stage / IO profiles.

It is a *separate* module on purpose: the canonical runner stays untouched
(its recovery logic is reused verbatim, not modified), and instrumentation
overhead only exists on this experiment path.
"""
from __future__ import annotations

import math
import zlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from wenbo_engine.circuit.io import validate_circuit_dict, levelize
from wenbo_engine.circuit.fusion import batch_levels
from wenbo_engine.kernel import gates as gmod
from wenbo_engine.kernel.cpu_nonlocal import (
    apply_1q_pair, apply_2q_pair_qa_local,
    apply_2q_pair_qb_local, apply_2q_quad,
)
from wenbo_engine.storage.block_store import (
    DTYPE, chunk_filename, init_zero_state, read_chunk, write_chunk_atomic,
)
from wenbo_engine.storage.manifest import Manifest, write_manifest_atomic
from wenbo_engine.wal.wal import WAL
from wenbo_engine.profiling.stage_profiler import NULL_STAGE_HANDLE

_ITEMSIZE = np.dtype(DTYPE).itemsize


@dataclass
class RunResult:
    final_dir: Path
    n_steps: int
    start_step: int          # first step executed this invocation
    resumed: bool            # True if a WAL recovery skipped earlier steps
    n_chunks: int
    chunk_size: int


# ── helpers (mirrors of single_node internals, kept local for decoupling) ──

def _buf_dir(work: Path, buf: str) -> Path:
    return work / f"state_{buf}"


def _other(buf: str) -> str:
    return "b" if buf == "a" else "a"


def _classify_ops(gates, k):
    local, nonlocal_ = [], []
    for g in gates:
        U = gmod.gate_matrix(g["gate"], g["params"])
        qs = g["qubits"]
        if all(q < k for q in qs):
            local.append((qs, U))
        else:
            nonlocal_.append((qs, U))
    return local, nonlocal_


def _apply_nonlocal(data: dict, qs, U, k: int) -> None:
    """Apply one non-local gate across the chunks loaded in `data`."""
    if len(qs) == 1:
        pbit = qs[0] - k
        done = set()
        for ci in data:
            c0 = ci & ~(1 << pbit)
            if c0 in done:
                continue
            done.add(c0)
            apply_1q_pair(data[c0], data[c0 | (1 << pbit)], U)
        return
    qa, qb = qs
    if qa < k:
        pbit = qb - k
        done = set()
        for ci in data:
            c0 = ci & ~(1 << pbit)
            if c0 in done:
                continue
            done.add(c0)
            apply_2q_pair_qa_local(data[c0], data[c0 | (1 << pbit)], qa, U)
    elif qb < k:
        pbit = qa - k
        done = set()
        for ci in data:
            c0 = ci & ~(1 << pbit)
            if c0 in done:
                continue
            done.add(c0)
            apply_2q_pair_qb_local(data[c0], data[c0 | (1 << pbit)], qb, U)
    else:
        pa, pb = qa - k, qb - k
        done = set()
        for ci in data:
            cb = ci & ~(1 << pa) & ~(1 << pb)
            if cb in done:
                continue
            done.add(cb)
            apply_2q_quad(
                data[cb], data[cb | (1 << pb)],
                data[cb | (1 << pa)], data[cb | (1 << pa) | (1 << pb)], U)


def _nonlocal_groups(n_chunks, nonlocal_ops, k):
    nl_bits = sorted({q - k for qs, _ in nonlocal_ops for q in qs if q >= k})
    mask = sum(1 << b for b in nl_bits)
    processed, groups = set(), []
    for c in range(n_chunks):
        base = c & ~mask
        if base in processed:
            continue
        processed.add(base)
        group = []
        for combo in range(1 << len(nl_bits)):
            idx = base
            for i, b in enumerate(nl_bits):
                if combo & (1 << i):
                    idx |= (1 << b)
            group.append(idx)
        groups.append(group)
    return groups


# ── public API ───────────────────────────────────────────────────────────

def run(circuit_dict: dict, work_dir, chunk_size: int = 1 << 20,
        kernel: str = "scalar", use_wal: bool = True,
        use_fusion: bool = False, profiler=None, checksum: bool = False
        ) -> RunResult:
    """Run a circuit out-of-core on one node, populating `profiler` if given."""
    cd = validate_circuit_dict(circuit_dict)
    n = cd["number_of_qubits"]
    N = 1 << n
    if chunk_size > N:
        chunk_size = N
    if N % chunk_size != 0:
        raise ValueError("2^n must be divisible by chunk_size")
    k = int(math.log2(chunk_size))
    chunk_bytes = chunk_size * _ITEMSIZE
    work = Path(work_dir)

    if kernel == "batched":
        from wenbo_engine.kernel.cpu_batched import apply_1q as a1, apply_2q as a2
    else:
        from wenbo_engine.kernel.cpu_scalar import apply_1q as a1, apply_2q as a2

    levels = levelize(cd)
    if use_fusion:
        steps = batch_levels(levels, k)
    else:
        steps = []
        for lv in levels:
            if not lv:
                continue
            lo, nlo = _classify_ops(lv, k)
            steps.append({"local_ops": lo, "nonlocal_ops": nlo})

    wal = WAL(work / "wal.json", circuit_dict=cd) if use_wal else None
    start_step = wal.done_steps if wal else 0
    current_buf = wal.committed_buf if wal else "a"
    resumed = start_step > 0

    a_dir = _buf_dir(work, "a")
    if start_step == 0 and not (a_dir / "manifest.json").exists():
        init_zero_state(str(a_dir), n, chunk_size)

    n_chunks = N // chunk_size
    man = Manifest(n_qubits=n, chunk_size=chunk_size, n_chunks=n_chunks,
                   chunks=[chunk_filename(i) for i in range(n_chunks)])

    for step_idx in range(start_step, len(steps)):
        src_dir = _buf_dir(work, current_buf)
        dst_dir = _buf_dir(work, _other(current_buf))
        _wipe(dst_dir)
        (dst_dir / "chunks").mkdir(parents=True, exist_ok=True)

        step = steps[step_idx]
        local_ops = step["local_ops"]
        nonlocal_ops = step["nonlocal_ops"]
        mode = "resume" if (step_idx == start_step and resumed) else "normal"

        stage_cm = (profiler.stage(
            step_idx, local_ops=len(local_ops),
            rank_nonlocal_ops=len(nonlocal_ops), mpi_nonlocal_ops=0,
            recovery_mode=mode)
            if profiler is not None else _null_stage())

        with stage_cm as h:
            affected = set()
            if nonlocal_ops:
                affected = _process_nonlocal(
                    src_dir, dst_dir, man, local_ops, nonlocal_ops, k,
                    a1, a2, h, chunk_bytes, checksum)
            for ci in range(n_chunks):
                if ci in affected:
                    continue
                _process_local(src_dir, dst_dir, man, ci, local_ops,
                               a1, a2, h, chunk_bytes, checksum)

            with h.commit():
                write_manifest_atomic(dst_dir, man)
                if wal:
                    wal.commit_step(step_idx, _other(current_buf))

        current_buf = _other(current_buf)

    if wal:
        wal.close()
    return RunResult(
        final_dir=_buf_dir(work, current_buf), n_steps=len(steps),
        start_step=start_step, resumed=resumed,
        n_chunks=n_chunks, chunk_size=chunk_size)


# ── per-chunk processing with profiler hooks ───────────────────────────────

def _process_local(src_dir, dst_dir, man, ci, local_ops, a1, a2,
                   h, chunk_bytes, checksum):
    with h.read(chunk_bytes):
        data = read_chunk(src_dir / "chunks" / man.chunks[ci])
    with h.kernel():
        for qs, U in local_ops:
            (a1(data, qs[0], U) if len(qs) == 1
             else a2(data, qs[0], qs[1], U))
    if checksum:
        with h.checksum():
            zlib.crc32(data.tobytes())
    with h.write(chunk_bytes):
        write_chunk_atomic(dst_dir / "chunks" / man.chunks[ci], data)


def _process_nonlocal(src_dir, dst_dir, man, local_ops, nonlocal_ops, k,
                      a1, a2, h, chunk_bytes, checksum):
    groups = _nonlocal_groups(man.n_chunks, nonlocal_ops, k)
    affected = set()
    for group in groups:
        affected.update(group)
        data = {}
        for ci in group:
            with h.read(chunk_bytes):
                data[ci] = read_chunk(src_dir / "chunks" / man.chunks[ci])
        with h.kernel():
            for ci in group:
                for qs, U in local_ops:
                    (a1(data[ci], qs[0], U) if len(qs) == 1
                     else a2(data[ci], qs[0], qs[1], U))
            for qs, U in nonlocal_ops:
                _apply_nonlocal(data, qs, U, k)
        for ci in group:
            if checksum:
                with h.checksum():
                    zlib.crc32(data[ci].tobytes())
            with h.write(chunk_bytes):
                write_chunk_atomic(dst_dir / "chunks" / man.chunks[ci], data[ci])
    return affected


# ── tiny utilities ─────────────────────────────────────────────────────────

def _wipe(buf_dir: Path) -> None:
    import shutil
    chunks = buf_dir / "chunks"
    if chunks.exists():
        shutil.rmtree(chunks)
    mani = buf_dir / "manifest.json"
    if mani.exists():
        mani.unlink()


class _null_stage:
    """nullcontext yielding the shared no-op handle (no profiler attached)."""
    def __enter__(self):
        return NULL_STAGE_HANDLE

    def __exit__(self, *exc):
        return False
