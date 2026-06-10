"""Synchronous writeback queue (simple by design).

Collects dirty chunk ids and flushes them through a :class:`MemoryOverlay`
exactly once each.  This is the synchronous first version (no threads); the
small interface leaves room for an async writeback later without changing call
sites, but none is implemented here (kept simple).
"""
from __future__ import annotations


class WritebackQueue:
    def __init__(self, overlay):
        self._overlay = overlay
        self._pending: list[int] = []

    def enqueue(self, chunk_id: int) -> None:
        self._overlay.mark_dirty(chunk_id)
        self._pending.append(chunk_id)

    def flush(self) -> None:
        for cid in self._pending:
            self._overlay.writeback(cid)   # no-op if already clean
        self._pending.clear()
