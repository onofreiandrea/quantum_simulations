"""RAM overlay over a rank's logical chunks.

Loads logical chunks (chunk files) into memory, lets kernels operate on the
in-RAM arrays, tracks which are dirty, and writes each dirty chunk back exactly
once.  Honours a RAM budget (max resident chunks): a clean chunk may be evicted
to make room, a dirty one is written back first.

Layout-agnostic: it reads from a *source* chunks directory and writes to a
*destination* chunks directory.  Extent-backed source/destination generations
are handled by the runner's existing materialize/pack wrapper, so the overlay
always sees plain chunk files (keeps it simple and correct for both layouts).

Uses the project's instrumented ``read_chunk`` / ``write_chunk_atomic`` so the
benchmark measures the reduced read/write traffic for free.
"""
from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import numpy as np

from wenbo_engine.storage.block_store import (
    read_chunk, write_chunk_atomic, chunk_filename,
)


class MemoryOverlay:
    def __init__(self, src_chunks_dir, dst_chunks_dir,
                 ram_budget_chunks: int = 0):
        """``ram_budget_chunks`` <= 0 means unbounded."""
        self.src = Path(src_chunks_dir)
        self.dst = Path(dst_chunks_dir)
        self.dst.mkdir(parents=True, exist_ok=True)
        self.budget = int(ram_budget_chunks)
        self._resident: "OrderedDict[int, np.ndarray]" = OrderedDict()
        self._dirty: set[int] = set()
        # profiling
        self.load_count = 0
        self.writeback_count = 0

    # ── load / access ────────────────────────────────────────────────────
    def get(self, chunk_id: int) -> np.ndarray:
        """Return the resident array for ``chunk_id``, loading it if needed."""
        if chunk_id in self._resident:
            self._resident.move_to_end(chunk_id)
            return self._resident[chunk_id]
        self._evict_if_needed()
        arr = read_chunk(self.src / chunk_filename(chunk_id))
        self._resident[chunk_id] = arr
        self.load_count += 1
        return arr

    def mark_dirty(self, chunk_id: int) -> None:
        self._dirty.add(chunk_id)

    @property
    def resident_count(self) -> int:
        return len(self._resident)

    @property
    def dirty_count(self) -> int:
        return len(self._dirty)

    # ── writeback / eviction ─────────────────────────────────────────────
    def writeback(self, chunk_id: int) -> None:
        """Write one dirty chunk back to the destination, once."""
        if chunk_id not in self._dirty:
            return
        write_chunk_atomic(self.dst / chunk_filename(chunk_id),
                           self._resident[chunk_id])
        self._dirty.discard(chunk_id)
        self.writeback_count += 1

    def flush(self) -> None:
        """Write back every dirty chunk (once each)."""
        for cid in list(self._dirty):
            self.writeback(cid)

    def _evict_if_needed(self) -> None:
        if self.budget <= 0:
            return
        # evict least-recently-used CLEAN chunks until under budget
        while len(self._resident) >= self.budget:
            for cid in list(self._resident):
                if cid not in self._dirty:
                    self._resident.pop(cid)
                    break
            else:
                # all resident chunks are dirty — flush the LRU one first
                cid = next(iter(self._resident))
                self.writeback(cid)
                self._resident.pop(cid)
