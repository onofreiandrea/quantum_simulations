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
"""
from __future__ import annotations

import numpy as np

from wenbo_engine.storage.block_store import DTYPE


class RemoteBufferCache:
    def __init__(self):
        self._buf: dict[tuple[int, int], np.ndarray] = {}
        self.hits = 0
        self.misses = 0

    def get(self, partner_rank: int, chunk_index: int):
        return self._buf.get((partner_rank, chunk_index))

    def put(self, partner_rank: int, chunk_index: int, buf: np.ndarray) -> None:
        self._buf[(partner_rank, chunk_index)] = buf

    def exchange(self, comm, partner_rank: int, chunk_index: int,
                 send_buf: np.ndarray, chunk_size: int) -> np.ndarray:
        """Return the partner's chunk, exchanging over MPI only on a miss.

        On a hit, returns the cached remote buffer (no MPI call).  On a miss,
        issues one ``Sendrecv`` and caches a copy of the received data.
        """
        key = (partner_rank, chunk_index)
        cached = self._buf.get(key)
        if cached is not None:
            self.hits += 1
            return cached
        self.misses += 1
        recv = np.empty(chunk_size, dtype=DTYPE)
        comm.Sendrecv(sendbuf=send_buf, dest=partner_rank,
                      recvbuf=recv, source=partner_rank)
        self._buf[key] = recv.copy()
        return recv

    def clear(self) -> None:
        self._buf.clear()
