"""Tests for the generation recovery dataclasses.

Covers ChunkRecord / RankManifest / GlobalCommitRecord:
  - JSON serialization + deserialization round-trips
  - stable content hashing (deterministic, excludes volatile fields)
  - tamper detection via self-hash
  - atomic write + read from disk
  - structural validation
"""
import tempfile
from pathlib import Path

import pytest

from wenbo_engine.recovery import (
    ChunkRecord, RankManifest, GlobalCommitRecord, commit_filename,
)


# ── ChunkRecord ────────────────────────────────────────────────────────

def test_chunk_record_roundtrip():
    c = ChunkRecord(index=3, filename="chunk_000003.bin",
                    size_bytes=8192, checksum="abc123")
    c2 = ChunkRecord.from_dict(c.to_dict())
    assert c2 == c


def test_chunk_record_optional_checksum():
    c = ChunkRecord(index=0, filename="chunk_000000.bin", size_bytes=64)
    assert c.checksum is None
    assert ChunkRecord.from_dict(c.to_dict()).checksum is None


# ── RankManifest ───────────────────────────────────────────────────────

def _manifest(generation=1, created=100.0):
    return RankManifest(
        rank=0, generation=generation, n_chunks=2, chunk_size=128,
        dtype="complex64", circuit_hash="cafef00d",
        chunks=[
            ChunkRecord(0, "chunk_000000.bin", 1024, "aa"),
            ChunkRecord(1, "chunk_000001.bin", 1024, "bb"),
        ],
        created=created,
    )


def test_rank_manifest_json_roundtrip():
    m = _manifest().seal()
    m2 = RankManifest.from_json(m.to_json())
    assert m2.to_dict() == m.to_dict()
    assert m2.manifest_hash == m.manifest_hash


def test_rank_manifest_hash_is_deterministic():
    assert _manifest().compute_hash() == _manifest().compute_hash()


def test_rank_manifest_hash_excludes_created():
    # Same content, different wall-clock times → same hash.
    assert _manifest(created=1.0).compute_hash() == \
        _manifest(created=999.0).compute_hash()


def test_rank_manifest_hash_changes_with_content():
    a = _manifest()
    b = _manifest()
    b.chunks[0].size_bytes = 2048
    assert a.compute_hash() != b.compute_hash()


def test_rank_manifest_verify_self_hash():
    m = _manifest().seal()
    assert m.verify_self_hash()
    # Tamper after sealing → detected.
    m.chunks[0].size_bytes = 4096
    assert not m.verify_self_hash()


def test_rank_manifest_validate_chunk_count():
    m = _manifest()
    m.n_chunks = 5
    with pytest.raises(ValueError, match="n_chunks"):
        m.validate()


def test_rank_manifest_validate_duplicate_index():
    m = _manifest()
    m.chunks[1].index = 0
    with pytest.raises(ValueError, match="duplicate"):
        m.validate()


def test_rank_manifest_write_and_read(tmp_path):
    m = _manifest()
    gen_dir = tmp_path / "gen_000001"
    m.write_atomic(gen_dir)
    assert (gen_dir / "manifest.json").exists()
    loaded = RankManifest.read(gen_dir)
    assert loaded.manifest_hash == m.manifest_hash
    assert loaded.verify_self_hash()
    assert RankManifest.exists(gen_dir)


def test_rank_manifest_write_seals_hash(tmp_path):
    m = _manifest()
    m.manifest_hash = ""  # not sealed
    m.write_atomic(tmp_path / "gen_000001")
    assert m.manifest_hash  # write_atomic seals


# ── GlobalCommitRecord ─────────────────────────────────────────────────

def _commit(generation=1, n_ranks=2):
    return GlobalCommitRecord(
        generation=generation, n_ranks=n_ranks, circuit_hash="cafef00d",
        step_index=generation,
        rank_manifest_hashes={r: f"hash{r}" for r in range(n_ranks)},
        created=100.0,
    )


def test_commit_json_roundtrip():
    c = _commit().seal()
    c2 = GlobalCommitRecord.from_json(c.to_json())
    assert c2.to_dict() == c.to_dict()
    assert c2.verify_self_hash()


def test_commit_hash_excludes_created():
    a = _commit()
    a.created = 1.0
    b = _commit()
    b.created = 2.0
    assert a.compute_hash() == b.compute_hash()


def test_commit_self_hash_detects_tamper():
    c = _commit().seal()
    assert c.verify_self_hash()
    c.rank_manifest_hashes[0] = "evil"
    assert not c.verify_self_hash()


def test_commit_validate_requires_all_ranks():
    c = _commit(n_ranks=3)
    del c.rank_manifest_hashes[2]
    with pytest.raises(ValueError, match="missing manifest hashes"):
        c.validate()


def test_commit_write_and_read(tmp_path):
    c = _commit()
    c.write_atomic(tmp_path)
    path = tmp_path / commit_filename(c.generation)
    assert path.exists()
    loaded = GlobalCommitRecord.read_file(path)
    assert loaded.verify_self_hash()
    assert loaded.commit_hash == c.commit_hash


def test_commit_filename_format():
    assert commit_filename(0) == "commit_000000.json"
    assert commit_filename(42) == "commit_000042.json"
