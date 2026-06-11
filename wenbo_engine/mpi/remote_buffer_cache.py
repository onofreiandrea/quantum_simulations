"""Per-stage cache of received remote chunks.

When more than one gate (or kernel) in a stage needs the SAME remote chunk
from the SAME partner rank, the chunk should be fetched over MPI once and
reused, not re-exchanged.  This cache holds received remote buffers keyed by
``(partner_rank, chunk_index)`` for the lifetime of a stage and is cleared at
the stage boundary.

It is also the single place that issues the real ``Sendrecv`` for a cache
miss, so ``hits`` / ``misses`` are exact counts of avoided vs. issued
exchanges.  Correctness note: within a levelized stage a given local chunk is
touched by at most one batchable gate (gates are on disjoint qubits), so a
cached remote chunk never goes stale before the stage ends.

RAM safety: ``max_bytes`` caps the total cached bytes.  When caching another
remote chunk would exceed the cap, least-recently-used entries are evicted —
a later use simply re-issues a Sendrecv (correctness preserved, only the reuse
optimization is dropped).  This is what keeps gate-aware exchange from OOMing
at large n (the unbounded cache grew to the whole remote partition).
``max_bytes <= 0`` is unbounded (previous behaviour).
"""
from __future__ import annotations

from collections import OrderedDict

import numpy as np

from wenbo_engine.storage.block_store import DTYPE


class RemoteBufferCache:
    def __init__(self, max_bytes: int = 0):
        self._buf: "OrderedDict[tuple[int, int], np.ndarray]" = OrderedDict()
        self.max_bytes = int(max_bytes)
        self._bytes = 0
        self.peak_bytes = 0
        self.hits = 0
        self.misses = 0
        self.evictions = 0

    def get(self, partner_rank: int, chunk_index: int):
        key = (partner_rank, chunk_index)
        buf = self._buf.get(key)
        if buf is not None:
            self._buf.move_to_end(key)
        return buf

    def _evict_to_fit(self, incoming: int) -> None:
        if self.max_bytes <= 0:
            return
        # keep at least the incoming buffer itself possible; evict LRU others
        while self._buf and self._bytes + incoming > self.max_bytes:
            _k, old = self._buf.popitem(last=False)   # LRU
            self._bytes -= int(old.nbytes)
            self.evictions += 1

    def put(self, partner_rank: int, chunk_index: int, buf: np.ndarray) -> None:
        key = (partner_rank, chunk_index)
        if key in self._buf:
            self._bytes -= int(self._buf.pop(key).nbytes)
        self._evict_to_fit(int(buf.nbytes))
        self._buf[key] = buf
        self._buf.move_to_end(key)
        self._bytes += int(buf.nbytes)
        if self._bytes > self.peak_bytes:
            self.peak_bytes = self._bytes

    def exchange(self, comm, partner_rank: int, chunk_index: int,
                 send_buf: np.ndarray, chunk_size: int) -> np.ndarray:
        """Return the partner's chunk, exchanging over MPI only on a miss.

        On a hit, returns the cached remote buffer (no MPI call).  On a miss,
        issues one ``Sendrecv`` and caches a copy of the received data (subject
        to the ``max_bytes`` budget — LRU-evicting older entries if needed).
        """
        cached = self.get(partner_rank, chunk_index)
        if cached is not None:
            self.hits += 1
            return cached
        self.misses += 1
        recv = np.empty(chunk_size, dtype=DTYPE)
        comm.Sendrecv(sendbuf=send_buf, dest=partner_rank,
                      recvbuf=recv, source=partner_rank)
        self.put(partner_rank, chunk_index, recv.copy())
        return recv

    def clear(self) -> None:
        self._buf.clear()
        self._bytes = 0
