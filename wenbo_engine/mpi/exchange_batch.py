"""Gate-aware batched MPI exchange executor.

Replaces the naive "one blocking ``Sendrecv`` per chunk per gate" path with a
batched one: for each MPI gate, all the chunks it exchanges with its partner
are sent/received in a few large ``Sendrecv`` calls (bounded by a memory
budget) instead of one call per chunk, and any remote chunk already received
this stage is reused from :class:`RemoteBufferCache` instead of re-exchanged.

The per-pair kernel math is IDENTICAL to the naive runner (it calls the same
``cpu_nonlocal`` kernels), so the final state is bit-for-bit unchanged — only
the *number* and *size* of MPI messages differ.  Gates that don't fit the
batched fast path (``kind == "fallback"``) are returned to the caller, which
applies them with the existing naive per-gate path.

No kernel/recovery/WAL code is touched; this only orchestrates exchange + kernel
calls on already-loaded chunks.
"""
from __future__ import annotations

import numpy as np

from wenbo_engine.storage.block_store import (
    read_chunk, write_chunk_atomic, chunk_filename, DTYPE,
)
from wenbo_engine.kernel.cpu_nonlocal import (
    apply_1q_pair, apply_2q_pair_qa_local, apply_2q_pair_qb_local,
)


def _apply_pair(ge, my: np.ndarray, rem: np.ndarray) -> None:
    """Apply one gate to a (local, remote) chunk pair — same math as naive."""
    if ge.kind == "1q":
        if ge.i_am_low:
            apply_1q_pair(my, rem, ge.U)
        else:
            apply_1q_pair(rem, my, ge.U)
    elif ge.kind == "2q_one_local":
        c0 = my if ge.i_am_low else rem
        c1 = rem if ge.i_am_low else my
        if ge.mpi_is_first:
            apply_2q_pair_qb_local(c0, c1, ge.other_q, ge.U)
        else:
            apply_2q_pair_qa_local(c0, c1, ge.other_q, ge.U)
    else:  # pragma: no cover - planner never marks these batchable
        raise ValueError(f"non-batchable gate kind {ge.kind!r}")


def apply_stage_gate_aware(comm, plan, buf_dir, chunk_size: int,
                           batch_chunks: int, cache) -> list:
    """Execute the batchable gates in ``plan``; return the fallback gates.

    ``batch_chunks`` bounds how many chunks are exchanged per ``Sendrecv``
    (memory bound: ~``2 * batch_chunks * chunk_bytes`` resident).
    """
    chunks_dir = buf_dir / "chunks"
    fallback = []
    for ge in plan:
        if not ge.batchable:
            fallback.append(ge)
            continue
        pairs = ge.chunk_pairs
        for start in range(0, len(pairs), batch_chunks):
            batch = pairs[start:start + batch_chunks]
            local = {}
            remotes = {}
            need = []                       # chunks not already cached
            for (lci, rci) in batch:
                local[lci] = read_chunk(chunks_dir / chunk_filename(lci))
                cached = cache.get(ge.partner_rank, rci)
                if cached is not None:
                    remotes[lci] = cached
                    cache.hits += 1
                else:
                    need.append((lci, rci))
            if need:
                # ONE batched Sendrecv for every uncached chunk in this batch.
                send = np.concatenate([local[lci] for (lci, _) in need])
                recv = np.empty(len(need) * chunk_size, dtype=DTYPE)
                comm.Sendrecv(sendbuf=send, dest=ge.partner_rank,
                              recvbuf=recv, source=ge.partner_rank)
                cache.misses += 1
                for j, (lci, rci) in enumerate(need):
                    rem = recv[j * chunk_size:(j + 1) * chunk_size].copy()
                    remotes[lci] = rem
                    cache.put(ge.partner_rank, rci, rem)
            for (lci, rci) in batch:
                _apply_pair(ge, local[lci], remotes[lci])
                write_chunk_atomic(chunks_dir / chunk_filename(lci), local[lci])
    return fallback


def default_batch_chunks(chunk_size: int,
                         budget_bytes: int = 256 * 1024 * 1024) -> int:
    """How many chunks fit a per-exchange memory budget (>=1)."""
    chunk_bytes = chunk_size * np.dtype(DTYPE).itemsize
    return max(1, budget_bytes // max(chunk_bytes, 1))
