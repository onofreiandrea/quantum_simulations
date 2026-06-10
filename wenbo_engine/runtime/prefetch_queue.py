"""Synchronous prefetch queue (simple by design).

A thin, ordered loader for the chunks a compute unit will touch.  This is the
*synchronous* implementation (no threads) requested as the first version: it
simply loads chunks through a :class:`MemoryOverlay` in order.  The interface is
deliberately small so an async (threaded) prefetch can be slotted in later
without changing call sites — but no async is implemented here (kept simple).
"""
from __future__ import annotations


class PrefetchQueue:
    def __init__(self, overlay):
        self._overlay = overlay

    def prefetch(self, chunk_ids) -> None:
        """Load each chunk into the overlay (synchronous)."""
        for cid in chunk_ids:
            self._overlay.get(cid)

    def get(self, chunk_id):
        return self._overlay.get(chunk_id)
