"""Unit tests for the RAM memory overlay."""
import numpy as np

from wenbo_engine.storage.block_store import (
    write_chunk_atomic, read_chunk, chunk_filename, DTYPE,
)
from wenbo_engine.runtime.memory_overlay import MemoryOverlay

CS = 4


def _src(tmp_path, n):
    src = tmp_path / "src"
    src.mkdir()
    for c in range(n):
        write_chunk_atomic(src / chunk_filename(c), np.full(CS, c + 1, dtype=DTYPE))
    return src


# ── 1. load + write one chunk ───────────────────────────────────────────

def test_load_and_write_one_chunk(tmp_path):
    src = _src(tmp_path, 1)
    ov = MemoryOverlay(src, tmp_path / "dst")
    arr = ov.get(0)
    assert np.array_equal(arr, np.full(CS, 1, dtype=DTYPE))
    arr[:] = 7.0
    ov.mark_dirty(0)
    ov.writeback(0)
    assert np.array_equal(read_chunk(tmp_path / "dst" / chunk_filename(0)),
                          np.full(CS, 7, dtype=DTYPE))
    assert ov.load_count == 1 and ov.writeback_count == 1


# ── 2. load multiple chunks ─────────────────────────────────────────────

def test_load_multiple_chunks(tmp_path):
    src = _src(tmp_path, 4)
    ov = MemoryOverlay(src, tmp_path / "dst")
    for c in range(4):
        assert np.array_equal(ov.get(c), np.full(CS, c + 1, dtype=DTYPE))
    assert ov.load_count == 4
    # re-get is cached (no extra load)
    ov.get(0)
    assert ov.load_count == 4


# ── 3. dirty chunk written once ─────────────────────────────────────────

def test_dirty_written_once(tmp_path):
    src = _src(tmp_path, 1)
    ov = MemoryOverlay(src, tmp_path / "dst")
    ov.get(0)
    ov.mark_dirty(0)
    ov.writeback(0)
    ov.writeback(0)            # already clean -> no second write
    ov.flush()                 # nothing dirty
    assert ov.writeback_count == 1


# ── 4. clean chunk is not rewritten ─────────────────────────────────────

def test_clean_not_rewritten(tmp_path):
    src = _src(tmp_path, 2)
    ov = MemoryOverlay(src, tmp_path / "dst")
    ov.get(0)                  # read, never marked dirty
    ov.flush()
    assert ov.writeback_count == 0
    assert not (tmp_path / "dst" / chunk_filename(0)).exists()


# ── RAM budget: evicts clean, flushes dirty before evict ────────────────

def test_ram_budget_evicts_clean(tmp_path):
    src = _src(tmp_path, 4)
    ov = MemoryOverlay(src, tmp_path / "dst", ram_budget_chunks=2)
    ov.get(0); ov.get(1)
    assert ov.resident_count <= 2
    ov.get(2); ov.get(3)       # forces eviction of clean chunks
    assert ov.resident_count <= 2
    assert ov.writeback_count == 0   # all were clean


def test_ram_budget_flushes_dirty_before_evict(tmp_path):
    src = _src(tmp_path, 3)
    ov = MemoryOverlay(src, tmp_path / "dst", ram_budget_chunks=1)
    a = ov.get(0); a[:] = 9.0; ov.mark_dirty(0)
    ov.get(1)                  # must flush dirty chunk 0 before evicting it
    assert ov.writeback_count == 1
    assert np.array_equal(read_chunk(tmp_path / "dst" / chunk_filename(0)),
                          np.full(CS, 9, dtype=DTYPE))


# ── 10 (overlay scope): only the dst dir is written, src untouched ──────

def test_overlay_only_writes_dst(tmp_path):
    src = _src(tmp_path, 2)
    before = {p.name: p.read_bytes() for p in src.iterdir()}
    ov = MemoryOverlay(src, tmp_path / "dst")
    a = ov.get(0); a[:] = 5.0; ov.mark_dirty(0); ov.writeback(0)
    # src files are unchanged (overlay never writes into the source)
    after = {p.name: p.read_bytes() for p in src.iterdir()}
    assert before == after
    assert (tmp_path / "dst" / chunk_filename(0)).exists()
