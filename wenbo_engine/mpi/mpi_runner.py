"""MPI-based distributed out-of-core quantum state-vector simulator.

Distributes the state vector across MPI ranks.  Each rank stores and
processes its own partition of chunks on local NVMe, using the same
double-buffering and WAL crash-recovery as the single-node runner.

Gate classification (n qubits, chunk has k local qubits, p = log2(ranks)):
  - Local        (q < k)         : within-chunk, no cross-chunk I/O
  - Rank-nonlocal (k <= q < n-p) : partner chunks on same rank
  - MPI-nonlocal  (q >= n-p)     : partner chunks on different rank

Within a compiled step, all gates touch disjoint qubits (from
levelization), so local/rank-nonlocal/MPI-nonlocal gates commute and
can be applied in any order.

Launch:
    mpirun -np 4 python -m wenbo_engine.mpi.mpi_benchmark ...
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import shutil
import time
from pathlib import Path
from queue import Queue
from threading import Thread

import numpy as np
from mpi4py import MPI

from wenbo_engine.circuit.io import validate_circuit_dict, levelize
from wenbo_engine.kernel import gates as gmod
from wenbo_engine.kernel.cpu_batched import apply_1q, apply_2q
from wenbo_engine.kernel.cpu_nonlocal import (
    apply_1q_pair,
    apply_2q_pair_qa_local,
    apply_2q_pair_qb_local,
    apply_2q_quad,
)
from wenbo_engine.storage.block_store import (
    DTYPE, chunk_filename, read_chunk, write_chunk_atomic,
)

log = logging.getLogger(__name__)


# ── helpers ────────────────────────────────────────────────────────────

def _circuit_hash(circuit_dict: dict) -> str:
    raw = json.dumps(circuit_dict, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _write_wal(wal_path: Path, circ_hash: str,
               committed_buf: str, done_steps: int) -> None:
    wal_path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps({
        "circuit_hash": circ_hash,
        "committed_buf": committed_buf,
        "done_steps": done_steps,
    })
    tmp = wal_path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(str(tmp), str(wal_path))


def _read_wal(wal_path: Path) -> dict | None:
    if not wal_path.exists():
        return None
    with open(wal_path) as f:
        return json.loads(f.read())


def _wipe_buf(buf_dir: Path) -> None:
    chunks = buf_dir / "chunks"
    if chunks.exists():
        shutil.rmtree(chunks)


def _crash_after_step() -> int | None:
    val = os.environ.get("WE_CRASH_AFTER_STEP")
    return int(val) if val is not None else None


# ── pipelined I/O + compute ───────────────────────────────────────────

_SENTINEL = None


def _chunk_reader(src_dir: Path, n_chunks_per_rank: int,
                  skip_set: set, q_out: Queue,
                  errors: list | None = None) -> None:
    """Read individual chunks, skipping those already processed."""
    try:
        for local_ci in range(n_chunks_per_rank):
            if local_ci in skip_set:
                continue
            data = read_chunk(src_dir / "chunks" / chunk_filename(local_ci))
            q_out.put((local_ci, data))
    except Exception as e:
        log.error("Reader thread failed: %s", e)
        if errors is not None:
            errors.append(e)
    q_out.put(_SENTINEL)


def _chunk_worker(local_ops: list, q_in: Queue, q_out: Queue,
                  errors: list | None = None) -> None:
    """Apply local gates to individual chunks."""
    try:
        while True:
            item = q_in.get()
            if item is _SENTINEL:
                break
            local_ci, data = item
            _apply_local_ops(data, local_ops)
            q_out.put((local_ci, data))
    except Exception as e:
        log.error("Worker thread failed: %s", e)
        if errors is not None:
            errors.append(e)
    q_out.put(_SENTINEL)


def _chunk_writer(dst_dir: Path, q_in: Queue,
                  errors: list | None = None) -> None:
    """Write individual chunks to destination."""
    chunks_dst = dst_dir / "chunks"
    try:
        while True:
            item = q_in.get()
            if item is _SENTINEL:
                return
            local_ci, data = item
            write_chunk_atomic(chunks_dst / chunk_filename(local_ci), data)
    except Exception as e:
        log.error("Writer thread failed: %s", e)
        if errors is not None:
            errors.append(e)


def _pipeline_local_chunks(src_dir: Path, dst_dir: Path,
                           n_chunks_per_rank: int,
                           local_ops: list, buffer_depth: int,
                           skip_set: set) -> None:
    """Pipeline reader → worker → writer for local-only chunks."""
    (dst_dir / "chunks").mkdir(parents=True, exist_ok=True)
    errors: list = []
    rq: Queue = Queue(maxsize=buffer_depth)
    wq: Queue = Queue(maxsize=buffer_depth)
    rt = Thread(target=_chunk_reader,
                args=(src_dir, n_chunks_per_rank, skip_set, rq, errors))
    wt = Thread(target=_chunk_worker, args=(local_ops, rq, wq, errors))
    wr = Thread(target=_chunk_writer, args=(dst_dir, wq, errors))
    rt.start(); wt.start(); wr.start()
    rt.join(); wt.join(); wr.join()
    if errors:
        raise RuntimeError(f"Pipeline failed: {errors[0]}") from errors[0]


def _group_reader(src_dir: Path, groups: list[list[int]],
                  q_out: Queue, errors: list | None = None) -> None:
    """Read chunk groups (for nonlocal gates) from source."""
    try:
        for group in groups:
            data = {ci: read_chunk(src_dir / "chunks" / chunk_filename(ci))
                    for ci in group}
            q_out.put((group, data))
    except Exception as e:
        log.error("Group reader failed: %s", e)
        if errors is not None:
            errors.append(e)
    q_out.put(_SENTINEL)


def _group_worker(local_ops: list, nonlocal_ops: list, k: int,
                  q_in: Queue, q_out: Queue,
                  errors: list | None = None) -> None:
    """Apply local + nonlocal gates to chunk groups."""
    try:
        while True:
            item = q_in.get()
            if item is _SENTINEL:
                break
            group, data = item
            for ci in group:
                _apply_local_ops(data[ci], local_ops)
            for qs, U in nonlocal_ops:
                _apply_nonlocal_within_rank(data, qs, U, k)
            q_out.put((group, data))
    except Exception as e:
        log.error("Group worker failed: %s", e)
        if errors is not None:
            errors.append(e)
    q_out.put(_SENTINEL)


def _group_writer(dst_dir: Path, q_in: Queue,
                  errors: list | None = None) -> None:
    """Write chunk groups to destination."""
    chunks_dst = dst_dir / "chunks"
    try:
        while True:
            item = q_in.get()
            if item is _SENTINEL:
                return
            group, data = item
            for ci in group:
                write_chunk_atomic(chunks_dst / chunk_filename(ci), data[ci])
    except Exception as e:
        log.error("Group writer failed: %s", e)
        if errors is not None:
            errors.append(e)


# ── rank-nonlocal sub-pass splitting ──────────────────────────────────

def _split_nonlocal_batches(nonlocal_ops: list, k: int, chunk_size: int,
                            max_group_mem: int = 4 * 1024**3) -> list[list]:
    """Split rank-nonlocal ops into memory-safe batches.

    When all nonlocal gates are grouped together, the group contains
    2^(total_nonlocal_bits) chunks.  If that exceeds max_group_mem,
    we find independent connected components of nonlocal bits and
    batch them so each batch's group fits in memory.

    Returns list of op-batches (each batch is a list of (qs, U) tuples).
    """
    if not nonlocal_ops:
        return []

    # Nonlocal bits per gate
    gate_bits = []
    for qs, _U in nonlocal_ops:
        bits = frozenset(q - k for q in qs if q >= k)
        gate_bits.append(bits)

    all_bits = set()
    for bits in gate_bits:
        all_bits.update(bits)

    chunk_bytes = chunk_size * np.dtype(DTYPE).itemsize
    total_group_mem = (1 << len(all_bits)) * chunk_bytes

    if total_group_mem <= max_group_mem:
        return [nonlocal_ops]

    # Union-find for connected components
    parent = {b: b for b in all_bits}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for bits in gate_bits:
        bits_list = list(bits)
        for i in range(1, len(bits_list)):
            union(bits_list[0], bits_list[i])

    # Group gates by connected component
    from collections import defaultdict
    comp_gates: dict[int, list] = defaultdict(list)
    comp_bits: dict[int, set] = defaultdict(set)
    for (qs, U), bits in zip(nonlocal_ops, gate_bits):
        root = find(next(iter(bits)))
        comp_gates[root].append((qs, U))
        comp_bits[root].update(bits)

    # Sort components smallest-first for greedy packing
    sorted_comps = sorted(comp_bits.keys(), key=lambda r: len(comp_bits[r]))

    max_bits = int(math.log2(max_group_mem / chunk_bytes))
    batches: list[list] = []
    cur_ops: list = []
    cur_bits: set = set()

    for root in sorted_comps:
        merged = cur_bits | comp_bits[root]
        if len(merged) > max_bits and cur_ops:
            batches.append(cur_ops)
            cur_ops = list(comp_gates[root])
            cur_bits = set(comp_bits[root])
        else:
            cur_ops.extend(comp_gates[root])
            cur_bits = merged

    if cur_ops:
        batches.append(cur_ops)

    return batches


# ── step compilation ───────────────────────────────────────────────────

def _classify_ops(gates: list[dict], k: int):
    """Split gate dicts into (qubits, U) tuples classified as local / nonlocal."""
    local, nonlocal_ = [], []
    for g in gates:
        U = gmod.gate_matrix(g["gate"], g["params"]).astype(DTYPE)
        qs = g["qubits"]
        if all(q < k for q in qs):
            local.append((qs, U))
        else:
            nonlocal_.append((qs, U))
    return local, nonlocal_


def _compile_steps(levels: list[list[dict]], k: int, n_local_bits: int):
    """Compile levelized circuit into steps with 3-way op classification.

    Each step contains:
      local_ops          : all qubits < k
      rank_nonlocal_ops  : some qubit >= k, but partner bit < n_local_bits
      mpi_nonlocal_ops   : some qubit has partner bit >= n_local_bits
    """
    steps = []
    for lv in levels:
        if not lv:
            continue
        local_ops, nonlocal_ops = _classify_ops(lv, k)

        rank_nl, mpi_nl = [], []
        for qs, U in nonlocal_ops:
            # Check if any nonlocal qubit's partner bit is in the rank portion
            has_mpi = any((q - k) >= n_local_bits for q in qs if q >= k)
            if has_mpi:
                mpi_nl.append((qs, U))
            else:
                rank_nl.append((qs, U))

        steps.append({
            "local_ops": local_ops,
            "rank_nonlocal_ops": rank_nl,
            "mpi_nonlocal_ops": mpi_nl,
        })
    return steps


# ── state init ─────────────────────────────────────────────────────────

def _init_rank_state(rank: int, chunks_dir: Path,
                     n_chunks_per_rank: int, chunk_size: int) -> None:
    """Initialize |0...0> state for this rank's chunk partition.

    Uses sparse files for zero chunks — the OS returns zeros on read
    without writing any data to disk.  Only chunk 0 of rank 0 gets
    a real write (it holds state[0] = 1.0).
    """
    chunks_dir.mkdir(parents=True, exist_ok=True)
    sparse_bytes = chunk_size * np.dtype(DTYPE).itemsize
    for local_ci in range(n_chunks_per_rank):
        path = chunks_dir / chunk_filename(local_ci)
        if rank == 0 and local_ci == 0:
            data = np.zeros(chunk_size, dtype=DTYPE)
            data[0] = 1.0 + 0j
            write_chunk_atomic(path, data)
        else:
            with open(path, "wb") as f:
                f.truncate(sparse_bytes)


# ── local gate application ─────────────────────────────────────────────

def _apply_local_ops(chunk_data: np.ndarray, local_ops: list) -> None:
    for qs, U in local_ops:
        if len(qs) == 1:
            apply_1q(chunk_data, qs[0], U)
        else:
            apply_2q(chunk_data, qs[0], qs[1], U)


# ── rank-nonlocal processing (same as single_node) ────────────────────

def _apply_nonlocal_within_rank(data: dict, qs: list[int],
                                U: np.ndarray, k: int) -> None:
    """Apply one nonlocal gate across chunks loaded in `data` dict."""
    if len(qs) == 1:
        q = qs[0]
        pbit = q - k
        done = set()
        for ci in data:
            c0 = ci & ~(1 << pbit)
            if c0 in done:
                continue
            done.add(c0)
            c1 = c0 | (1 << pbit)
            apply_1q_pair(data[c0], data[c1], U)
    else:
        qa, qb = qs
        qa_local, qb_local = qa < k, qb < k
        if qa_local:
            pbit = qb - k
            done = set()
            for ci in data:
                c0 = ci & ~(1 << pbit)
                if c0 in done:
                    continue
                done.add(c0)
                c1 = c0 | (1 << pbit)
                apply_2q_pair_qa_local(data[c0], data[c1], qa, U)
        elif qb_local:
            pbit = qa - k
            done = set()
            for ci in data:
                c0 = ci & ~(1 << pbit)
                if c0 in done:
                    continue
                done.add(c0)
                c1 = c0 | (1 << pbit)
                apply_2q_pair_qb_local(data[c0], data[c1], qb, U)
        else:
            pa, pb = qa - k, qb - k
            done = set()
            for ci in data:
                cb = ci & ~(1 << pa) & ~(1 << pb)
                if cb in done:
                    continue
                done.add(cb)
                apply_2q_quad(
                    data[cb],
                    data[cb | (1 << pb)],
                    data[cb | (1 << pa)],
                    data[cb | (1 << pa) | (1 << pb)],
                    U,
                )


def _process_rank_nonlocal(src_dir: Path, dst_dir: Path,
                           n_chunks_per_rank: int,
                           local_ops: list, nonlocal_ops: list,
                           k: int, buffer_depth: int) -> set[int]:
    """Process rank-nonlocal groups with pipelined I/O.

    Uses reader→worker→writer threads at the GROUP level: each pipeline
    item is a group of 2^b chunks (b = number of distinct nonlocal bits).
    This overlaps disk reads of group N+1 with compute on group N and
    writes of group N-1.

    Returns set of affected local chunk indices.
    """
    (dst_dir / "chunks").mkdir(parents=True, exist_ok=True)

    nl_bits = set()
    for qs, U in nonlocal_ops:
        for q in qs:
            if q >= k:
                nl_bits.add(q - k)
    nl_bits_sorted = sorted(nl_bits)
    mask = sum(1 << b for b in nl_bits_sorted)

    # Pre-compute all groups
    processed_bases = set()
    groups = []
    affected = set()

    for c in range(n_chunks_per_rank):
        base = c & ~mask
        if base in processed_bases:
            continue
        processed_bases.add(base)

        group = []
        for combo in range(1 << len(nl_bits_sorted)):
            idx = base
            for i, b in enumerate(nl_bits_sorted):
                if combo & (1 << i):
                    idx |= (1 << b)
            if idx < n_chunks_per_rank:
                group.append(idx)
        groups.append(group)
        affected.update(group)

    # Pipeline: group_reader → group_worker → group_writer
    errors: list = []
    rq: Queue = Queue(maxsize=buffer_depth)
    wq: Queue = Queue(maxsize=buffer_depth)
    rt = Thread(target=_group_reader, args=(src_dir, groups, rq, errors))
    wt = Thread(target=_group_worker,
                args=(local_ops, nonlocal_ops, k, rq, wq, errors))
    wr = Thread(target=_group_writer, args=(dst_dir, wq, errors))
    rt.start(); wt.start(); wr.start()
    rt.join(); wt.join(); wr.join()
    if errors:
        raise RuntimeError(f"Group pipeline failed: {errors[0]}") from errors[0]

    return affected


# ── MPI-nonlocal gate application ──────────────────────────────────────

def _apply_mpi_gate(comm: MPI.Comm, rank: int, n_local_bits: int,
                    buf_dir: Path, n_chunks_per_rank: int,
                    chunk_size: int, k: int,
                    qs: list[int], U: np.ndarray) -> None:
    """Apply one MPI-nonlocal gate by exchanging chunks with partner rank(s)."""
    qs_local = [q for q in qs if q < k]
    qs_rank_nl = [q for q in qs if k <= q < k + n_local_bits]
    qs_mpi = [q for q in qs if (q - k) >= n_local_bits]

    if len(qs) == 1:
        _mpi_1q(comm, rank, n_local_bits, buf_dir,
                n_chunks_per_rank, chunk_size, k, qs[0], U)
    elif len(qs) == 2:
        if len(qs_mpi) == 1:
            mpi_q = qs_mpi[0]
            other_q = qs_local[0] if qs_local else qs_rank_nl[0]
            _mpi_2q_one(comm, rank, n_local_bits, buf_dir,
                        n_chunks_per_rank, chunk_size, k,
                        qs, U, mpi_q, other_q)
        else:
            _mpi_2q_both(comm, rank, n_local_bits, buf_dir,
                         n_chunks_per_rank, chunk_size, k, qs, U)


def _mpi_1q(comm, rank, n_local_bits, buf_dir,
            n_chunks_per_rank, chunk_size, k, q, U):
    """1-qubit gate with 3-stage pipelining.

    Reader thread:  pre-fetches next chunk from NVMe
    Main thread:    Sendrecv + compute (must be on main thread for MPI)
    Writer thread:  writes result back to NVMe

    Overlap: while main does sendrecv(N)+compute(N),
             reader loads chunk N+1, writer writes chunk N-1.
    """
    rank_bit = (q - k) - n_local_bits
    partner_rank = rank ^ (1 << rank_bit)
    i_am_low = (rank & (1 << rank_bit)) == 0

    chunks_dir = buf_dir / "chunks"

    # Reader thread: pre-fetch chunks from disk
    rq: Queue = Queue(maxsize=4)
    rt = Thread(target=_chunk_reader,
                args=(buf_dir, n_chunks_per_rank, set(), rq))
    rt.start()

    # Writer thread: write results to disk
    wq: Queue = Queue(maxsize=4)
    wt = Thread(target=_chunk_writer, args=(buf_dir, wq))
    wt.start()

    recv_buf = np.empty(chunk_size, dtype=DTYPE)

    while True:
        item = rq.get()
        if item is _SENTINEL:
            break
        local_ci, my_data = item

        comm.Sendrecv(sendbuf=my_data, dest=partner_rank,
                      recvbuf=recv_buf, source=partner_rank)

        if i_am_low:
            apply_1q_pair(my_data, recv_buf, U)
        else:
            apply_1q_pair(recv_buf, my_data, U)

        wq.put((local_ci, my_data))

    wq.put(_SENTINEL)
    rt.join()
    wt.join()


def _mpi_2q_one(comm, rank, n_local_bits, buf_dir,
                n_chunks_per_rank, chunk_size, k,
                qs, U, mpi_q, other_q):
    """2-qubit gate: one MPI-nonlocal qubit, one local or rank-nonlocal."""
    rank_bit = (mpi_q - k) - n_local_bits
    partner_rank = rank ^ (1 << rank_bit)
    i_am_low = (rank & (1 << rank_bit)) == 0
    mpi_is_first = (qs.index(mpi_q) == 0)

    recv_buf = np.empty(chunk_size, dtype=DTYPE)

    chunks_dir = buf_dir / "chunks"
    wq: Queue = Queue(maxsize=4)
    wt = Thread(target=_chunk_writer, args=(buf_dir, wq))
    wt.start()

    if other_q < k:
        # Other qubit is local (within chunk)
        for local_ci in range(n_chunks_per_rank):
            my_data = read_chunk(chunks_dir / chunk_filename(local_ci))

            comm.Sendrecv(sendbuf=my_data, dest=partner_rank,
                          recvbuf=recv_buf, source=partner_rank)

            c0 = my_data if i_am_low else recv_buf
            c1 = recv_buf if i_am_low else my_data

            if mpi_is_first:
                apply_2q_pair_qb_local(c0, c1, other_q, U)
            else:
                apply_2q_pair_qa_local(c0, c1, other_q, U)

            wq.put((local_ci, my_data))
    else:
        # Other qubit is rank-nonlocal: need to pair local chunks too
        other_pbit = other_q - k

        for local_ci in range(n_chunks_per_rank):
            if local_ci & (1 << other_pbit):
                continue  # process pairs from the "low" side only

            local_ci_hi = local_ci | (1 << other_pbit)

            d_lo = read_chunk(chunks_dir / chunk_filename(local_ci))
            d_hi = read_chunk(chunks_dir / chunk_filename(local_ci_hi))

            r_lo = np.empty(chunk_size, dtype=DTYPE)
            r_hi = np.empty(chunk_size, dtype=DTYPE)

            comm.Sendrecv(sendbuf=d_lo, dest=partner_rank,
                          recvbuf=r_lo, source=partner_rank)
            comm.Sendrecv(sendbuf=d_hi, dest=partner_rank,
                          recvbuf=r_hi, source=partner_rank)

            if i_am_low:
                c00, c01, c10, c11 = d_lo, d_hi, r_lo, r_hi
            else:
                c00, c01, c10, c11 = r_lo, r_hi, d_lo, d_hi

            if mpi_is_first:
                apply_2q_quad(c00, c01, c10, c11, U)
            else:
                apply_2q_quad(c00, c10, c01, c11, U)

            wq.put((local_ci, d_lo))
            wq.put((local_ci_hi, d_hi))

    wq.put(_SENTINEL)
    wt.join()


def _mpi_2q_both(comm, rank, n_local_bits, buf_dir,
                 n_chunks_per_rank, chunk_size, k, qs, U):
    """2-qubit gate: both qubits MPI-nonlocal."""
    qa, qb = qs
    rank_bit_a = (qa - k) - n_local_bits
    rank_bit_b = (qb - k) - n_local_bits

    # Four partner ranks
    r00 = rank & ~(1 << rank_bit_a) & ~(1 << rank_bit_b)
    r01 = r00 | (1 << rank_bit_b)
    r10 = r00 | (1 << rank_bit_a)
    r11 = r00 | (1 << rank_bit_a) | (1 << rank_bit_b)
    partners = [r00, r01, r10, r11]

    recv = {r: np.empty(chunk_size, dtype=DTYPE) for r in partners if r != rank}

    chunks_dir = buf_dir / "chunks"
    wq: Queue = Queue(maxsize=4)
    wt = Thread(target=_chunk_writer, args=(buf_dir, wq))
    wt.start()

    for local_ci in range(n_chunks_per_rank):
        my_data = read_chunk(chunks_dir / chunk_filename(local_ci))

        for pr in partners:
            if pr == rank:
                continue
            comm.Sendrecv(sendbuf=my_data, dest=pr,
                          recvbuf=recv[pr], source=pr)

        # Build quad: (qa_bit, qb_bit) -> data
        quad = {}
        for r in partners:
            bits = ((r >> rank_bit_a) & 1, (r >> rank_bit_b) & 1)
            quad[bits] = my_data if r == rank else recv[r]

        apply_2q_quad(quad[(0, 0)], quad[(0, 1)],
                      quad[(1, 0)], quad[(1, 1)], U)

        wq.put((local_ci, my_data))

    wq.put(_SENTINEL)
    wt.join()


# ── step execution ─────────────────────────────────────────────────────

def _apply_step(comm: MPI.Comm, rank: int, n_local_bits: int,
                src_dir: Path, dst_dir: Path,
                n_chunks_per_rank: int, chunk_size: int, k: int,
                local_ops: list, rank_nonlocal_ops: list,
                mpi_nonlocal_ops: list,
                buffer_depth: int = 4) -> None:
    """Apply one simulation step (one levelized layer).

    When rank-nonlocal gates touch many distinct nonlocal bits, the
    chunk groups would be too large for memory.  In that case, gates
    are split into memory-safe batches processed in separate sub-passes.
    """
    (dst_dir / "chunks").mkdir(parents=True, exist_ok=True)

    # Phase 1: rank-local processing (local + rank-nonlocal ops), src -> dst
    affected = set()
    if rank_nonlocal_ops:
        try:
            import psutil
            total_ram = psutil.virtual_memory().total
        except ImportError:
            total_ram = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
        ram_avail = max(total_ram - 8 * 1024**3, 16 * 1024**3)
        # Largest group that allows pipeline depth >= 1: (2*1+3)*group <= avail
        max_group = int(ram_avail // 5)
        batches = _split_nonlocal_batches(
            rank_nonlocal_ops, k, chunk_size, max_group_mem=max_group)

        chunk_bytes = chunk_size * np.dtype(DTYPE).itemsize

        def _safe_depth(batch_ops):
            """Compute buffer_depth that keeps pipeline memory safe.

            Worst case groups alive = 2*depth + 3:
              reader(1) + rq(depth) + worker(1) + wq(depth) + writer(1)
            """
            nl = set()
            for qs, _U in batch_ops:
                for q in qs:
                    if q >= k:
                        nl.add(q - k)
            group_mem = (1 << len(nl)) * chunk_bytes
            try:
                import psutil
                total_ram = psutil.virtual_memory().total
            except ImportError:
                total_ram = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
            max_avail = max(total_ram - 8 * 1024**3, 16 * 1024**3)
            # Solve: (2*d + 3) * group_mem <= max_avail
            safe = max(1, int((max_avail / group_mem - 3) / 2))
            return min(safe, buffer_depth)

        # First batch: local ops + nonlocal batch, src -> dst
        affected = _process_rank_nonlocal(
            src_dir, dst_dir, n_chunks_per_rank,
            local_ops, batches[0], k, _safe_depth(batches[0]))

        # Subsequent batches: nonlocal ops only, dst -> dst (in-place)
        for batch_ops in batches[1:]:
            _process_rank_nonlocal(
                dst_dir, dst_dir, n_chunks_per_rank,
                [], batch_ops, k, _safe_depth(batch_ops))

    # Process remaining local-only chunks with pipelined I/O
    _pipeline_local_chunks(src_dir, dst_dir, n_chunks_per_rank,
                           local_ops, buffer_depth, affected)

    # Phase 2: MPI-nonlocal gates (update dst in-place)
    for qs, U in mpi_nonlocal_ops:
        _apply_mpi_gate(comm, rank, n_local_bits,
                        dst_dir, n_chunks_per_rank, chunk_size, k,
                        qs, U)

    comm.Barrier()


# ── public API ─────────────────────────────────────────────────────────

def run(
    circuit_dict: dict,
    work_dir: str | Path,
    chunk_size: int = 1 << 24,
    use_wal: bool = True,
    comm: MPI.Comm | None = None,
    buffer_depth: int = 4,
) -> Path:
    """Run a quantum circuit distributed across MPI ranks.

    Each rank stores its chunk partition under work_dir/rank_N/.
    Uses double-buffering + WAL for crash recovery.

    Returns:
        Path to this rank's committed state directory (state_a or state_b).
    """
    if comm is None:
        comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_ranks = comm.Get_size()

    assert n_ranks > 0 and (n_ranks & (n_ranks - 1)) == 0, \
        f"Number of MPI ranks must be power of 2, got {n_ranks}"
    p = int(math.log2(n_ranks))

    circuit_dict = validate_circuit_dict(circuit_dict)
    n = circuit_dict["number_of_qubits"]
    k = int(math.log2(chunk_size))
    n_chunks_total = 1 << (n - k)
    n_chunks_per_rank = n_chunks_total // n_ranks
    n_local_bits = n - k - p  # bits in local chunk index

    assert n_chunks_per_rank >= 1, \
        f"Not enough chunks ({n_chunks_total}) for {n_ranks} ranks"

    work_dir = Path(work_dir)
    rank_dir = work_dir / f"rank_{rank}"

    state_a = rank_dir / "state_a"
    state_b = rank_dir / "state_b"

    # Compile circuit
    levels = levelize(circuit_dict)
    steps = _compile_steps(levels, k, n_local_bits)
    circ_hash = _circuit_hash(circuit_dict)

    crash_after = _crash_after_step()

    # WAL recovery
    wal_path = rank_dir / "wal.json"
    done_steps = 0
    committed_buf = "a"

    if use_wal:
        wal_data = _read_wal(wal_path)
        if wal_data is not None:
            if wal_data.get("circuit_hash") != circ_hash:
                if rank == 0:
                    log.warning("WAL circuit hash mismatch — starting fresh")
            else:
                done_steps = wal_data["done_steps"]
                committed_buf = wal_data["committed_buf"]

    # Sync recovery state across ranks: if one rank committed its WAL
    # but another crashed before committing, they'll disagree on done_steps.
    # Take the minimum so all ranks restart from the same safe point.
    all_done = comm.allgather(done_steps)
    min_done = min(all_done)
    if min_done != done_steps:
        if rank == 0:
            log.info(f"WAL sync: rank done_steps={all_done}, "
                     f"restarting from min={min_done}")
        done_steps = min_done
        # Recompute committed_buf: starts at "a", alternates each step
        committed_buf = "a" if min_done % 2 == 0 else "b"
        if use_wal:
            _write_wal(wal_path, circ_hash, committed_buf, min_done)
    elif done_steps > 0 and rank == 0:
        log.info(f"WAL recovery: resuming from step {done_steps}")

    # Init |0...0> if fresh
    if done_steps == 0:
        _wipe_buf(state_a)
        _wipe_buf(state_b)
        _init_rank_state(rank, state_a / "chunks",
                         n_chunks_per_rank, chunk_size)
        committed_buf = "a"
        if use_wal:
            _write_wal(wal_path, circ_hash, committed_buf, 0)

    comm.Barrier()

    # Count op types for logging
    total_local = sum(len(s["local_ops"]) for s in steps)
    total_rank_nl = sum(len(s["rank_nonlocal_ops"]) for s in steps)
    total_mpi_nl = sum(len(s["mpi_nonlocal_ops"]) for s in steps)

    if rank == 0:
        state_gb = (1 << n) * 8 / (1024 ** 3)
        log.info(f"MPI runner: {n}q, {state_gb:.1f} GB state, "
                 f"{n_chunks_total} chunks ({n_chunks_per_rank}/rank), "
                 f"chunk=2^{k}, {n_ranks} ranks")
        log.info(f"  {len(steps)} steps: {total_local} local, "
                 f"{total_rank_nl} rank-nonlocal, {total_mpi_nl} MPI-nonlocal gates")

    # Determine src/dst
    if committed_buf == "a":
        src_dir, dst_dir = state_a, state_b
    else:
        src_dir, dst_dir = state_b, state_a

    for step_idx in range(done_steps, len(steps)):
        t0 = time.time()
        step = steps[step_idx]

        # Ensure destination chunk dir exists (no wipe needed —
        # write_chunk_atomic uses os.replace for atomic overwrite)
        (dst_dir / "chunks").mkdir(parents=True, exist_ok=True)

        _apply_step(
            comm, rank, n_local_bits,
            src_dir, dst_dir,
            n_chunks_per_rank, chunk_size, k,
            step["local_ops"],
            step["rank_nonlocal_ops"],
            step["mpi_nonlocal_ops"],
            buffer_depth,
        )

        # Swap buffers
        committed_buf = "b" if committed_buf == "a" else "a"
        src_dir, dst_dir = dst_dir, src_dir

        # All ranks finished the step — commit WAL together
        comm.Barrier()
        if use_wal:
            _write_wal(wal_path, circ_hash, committed_buf, step_idx + 1)

        dt = time.time() - t0
        if rank == 0:
            log.info(f"  step {step_idx + 1}/{len(steps)} done in {dt:.1f}s "
                     f"(local={len(step['local_ops'])}, "
                     f"rank_nl={len(step['rank_nonlocal_ops'])}, "
                     f"mpi_nl={len(step['mpi_nonlocal_ops'])})")

        # Crash injection for testing
        if crash_after is not None and (step_idx + 1) >= crash_after:
            if rank == 0:
                log.info(f"Crash injection after step {step_idx + 1}")
            comm.Barrier()
            os._exit(1)

    return src_dir  # committed state directory


# ── utilities ──────────────────────────────────────────────────────────

def compute_norm(work_dir: str | Path,
                 comm: MPI.Comm | None = None) -> float:
    """Compute global state-vector norm across all ranks (distributed)."""
    if comm is None:
        comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    work_dir = Path(work_dir)
    rank_dir = work_dir / f"rank_{rank}"

    wal_data = _read_wal(rank_dir / "wal.json")
    buf = wal_data["committed_buf"] if wal_data else "a"
    chunks_dir = rank_dir / f"state_{buf}" / "chunks"

    local_norm_sq = 0.0
    for chunk_file in sorted(chunks_dir.glob("chunk_*.bin")):
        data = read_chunk(chunk_file)
        local_norm_sq += float(np.sum(np.abs(data.astype(np.complex128)) ** 2))

    global_norm_sq = np.array(0.0)
    comm.Allreduce(np.array(local_norm_sq), global_norm_sq, op=MPI.SUM)

    return float(np.sqrt(global_norm_sq))


def collect_state(work_dir: str | Path,
                  comm: MPI.Comm | None = None) -> np.ndarray | None:
    """Gather full state vector on rank 0.  Only for small states (testing)."""
    if comm is None:
        comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    work_dir = Path(work_dir)
    rank_dir = work_dir / f"rank_{rank}"

    wal_data = _read_wal(rank_dir / "wal.json")
    buf = wal_data["committed_buf"] if wal_data else "a"
    chunks_dir = rank_dir / f"state_{buf}" / "chunks"

    local_chunks = sorted(chunks_dir.glob("chunk_*.bin"))
    local_state = np.concatenate(
        [read_chunk(f) for f in local_chunks]
    ).astype(np.complex128)

    all_states = comm.gather(local_state, root=0)

    if rank == 0:
        return np.concatenate(all_states)
    return None
