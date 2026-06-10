"""Unit tests for the extent storage layout (store + manifest)."""
import numpy as np
import pytest

from wenbo_engine.storage.block_store import DTYPE
from wenbo_engine.storage.extent_manifest import (
    ExtentManifest, extent_filename, EXTENTS_DIRNAME,
)
from wenbo_engine.storage.extent_store import (
    write_generation_extents, read_logical_chunk, validate_extent_chunk,
    validate_extent_generation,
)

CS = 4  # logical chunk size (elements)


def _mk_chunks(n):
    return {c: np.full(CS, c + 1, dtype=DTYPE) for c in range(n)}


# ── 1. write/read one logical chunk ─────────────────────────────────────

def test_write_read_one_chunk(tmp_path):
    chunks = {0: np.array([1, 2, 3, 4], dtype=DTYPE)}
    man = write_generation_extents(tmp_path, chunks, chunk_size=CS)
    out = read_logical_chunk(tmp_path, man, 0)
    assert np.array_equal(out, chunks[0])
    assert man.n_chunks == 1 and man.n_extents == 1


# ── 2. many logical chunks packed into one extent ───────────────────────

def test_many_chunks_one_extent(tmp_path):
    chunks = _mk_chunks(8)
    man = write_generation_extents(tmp_path, chunks, chunk_size=CS,
                                   extent_bytes=10 ** 9)   # huge -> one extent
    assert man.n_extents == 1                              # all packed together
    # exactly ONE physical extent file for 8 logical chunks
    files = list((tmp_path / EXTENTS_DIRNAME).glob("extent_*.dat"))
    assert len(files) == 1
    for c in range(8):
        assert np.array_equal(read_logical_chunk(tmp_path, man, c), chunks[c])


def test_rolls_to_multiple_extents_under_budget(tmp_path):
    chunks = _mk_chunks(8)
    chunk_bytes = CS * np.dtype(DTYPE).itemsize
    man = write_generation_extents(tmp_path, chunks, chunk_size=CS,
                                   extent_bytes=chunk_bytes * 3)  # ~3 per extent
    assert man.n_extents >= 3
    files = list((tmp_path / EXTENTS_DIRNAME).glob("extent_*.dat"))
    assert len(files) == man.n_extents < 8       # fewer files than chunks
    for c in range(8):
        assert np.array_equal(read_logical_chunk(tmp_path, man, c), chunks[c])


# ── 3. manifest maps chunk_id -> correct extent/offset/length ───────────

def test_manifest_mapping_correct(tmp_path):
    chunks = _mk_chunks(4)
    man = write_generation_extents(tmp_path, chunks, chunk_size=CS,
                                   extent_bytes=10 ** 9)
    cb = CS * np.dtype(DTYPE).itemsize
    for c in range(4):
        rec = man.record(c)
        assert rec.extent_id == 0
        assert rec.offset == c * cb            # packed contiguously in order
        assert rec.length == cb


def test_manifest_roundtrip_atomic(tmp_path):
    man = write_generation_extents(tmp_path, _mk_chunks(5), chunk_size=CS)
    man.write_atomic(tmp_path)
    assert ExtentManifest.exists(tmp_path)
    re = ExtentManifest.read(tmp_path)
    assert re.verify_self_hash()
    assert re.n_chunks == 5
    assert re.to_dict() == man.to_dict()


# ── 4 + checksum corruption ─────────────────────────────────────────────

def test_checksum_catches_corruption(tmp_path):
    man = write_generation_extents(tmp_path, _mk_chunks(4), chunk_size=CS,
                                   extent_bytes=10 ** 9)
    # flip a byte inside extent 0 (same length) -> only checksum detects it
    ext = tmp_path / EXTENTS_DIRNAME / extent_filename(0)
    raw = bytearray(ext.read_bytes())
    raw[0] ^= 0xFF
    ext.write_bytes(bytes(raw))
    # size check passes, checksum check fails
    assert validate_extent_generation(tmp_path, man, check_checksums=False) == []
    fails = validate_extent_generation(tmp_path, man, check_checksums=True)
    assert any("checksum mismatch" in f for f in fails)


# ── 6. wrong extent length rejected ─────────────────────────────────────

def test_wrong_extent_length_rejected(tmp_path):
    man = write_generation_extents(tmp_path, _mk_chunks(4), chunk_size=CS,
                                   extent_bytes=10 ** 9)
    ext = tmp_path / EXTENTS_DIRNAME / extent_filename(0)
    ext.write_bytes(ext.read_bytes()[:8])     # truncate
    fails = validate_extent_generation(tmp_path, man, check_checksums=False)
    assert any("too short" in f for f in fails)


# ── 7. partial / missing extent rejected ────────────────────────────────

def test_missing_extent_rejected(tmp_path):
    man = write_generation_extents(tmp_path, _mk_chunks(4), chunk_size=CS,
                                   extent_bytes=10 ** 9)
    (tmp_path / EXTENTS_DIRNAME / extent_filename(0)).unlink()
    fails = validate_extent_generation(tmp_path, man, check_checksums=False)
    assert any("missing" in f for f in fails)


# ── 10 (core): extent bytes equal the original chunk-file bytes ──────────

def test_extent_payload_matches_raw_chunk_bytes(tmp_path):
    chunks = {0: np.random.randn(CS).astype(np.float32).view(DTYPE),
              1: np.random.randn(CS).astype(np.float32).view(DTYPE)}
    man = write_generation_extents(tmp_path, chunks, chunk_size=CS,
                                   extent_bytes=10 ** 9)
    for c in (0, 1):
        # what a chunk-file layout would have stored == what the extent yields
        assert read_logical_chunk(tmp_path, man, c).tobytes() == \
            np.ascontiguousarray(chunks[c], dtype=DTYPE).tobytes()
