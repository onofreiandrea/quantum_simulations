"""Per-rank generation manifest.

Each rank, when it produces generation ``g``, writes a ``manifest.json``
inside ``rank_XXXX/generations/gen_XXXXXX/`` describing exactly which chunk
files it wrote, their sizes, and (optionally) their checksums.

The manifest carries a content hash (``manifest_hash``) so a coordinator can
record it in the global commit record and recovery can later verify the
on-disk manifest is byte-for-byte the one that was committed.

Pure Python — no MPI, no numpy.  File I/O is local and atomic
(tmp + fsync + rename), matching the rest of wenbo_engine.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from wenbo_engine.recovery.global_commit import _fsync_dir


def _stable_hash(payload: dict) -> str:
    """Deterministic sha256 (hex, 32 chars) of a JSON-able payload."""
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                     default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


@dataclass
class ChunkRecord:
    """One chunk file written by a rank for a generation."""

    index: int          # local chunk index within the rank
    filename: str       # e.g. "chunk_000000.bin"
    size_bytes: int     # expected on-disk size
    checksum: str | None = None   # sha256 hex of file bytes, if computed

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "filename": self.filename,
            "size_bytes": self.size_bytes,
            "checksum": self.checksum,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ChunkRecord":
        return cls(
            index=int(d["index"]),
            filename=str(d["filename"]),
            size_bytes=int(d["size_bytes"]),
            checksum=d.get("checksum"),
        )


@dataclass
class RankManifest:
    """Manifest for one rank's output of a single generation."""

    rank: int
    generation: int
    n_chunks: int
    chunk_size: int           # amplitudes per chunk
    dtype: str
    circuit_hash: str
    # lineage: which generation this was derived from and which circuit step
    # produced it.  -1 means "initial state" (generation 0, no step applied).
    # Both are part of the hashed content so a tampered lineage is detected,
    # and recovery cross-checks that all ranks of a generation agree on them.
    parent_generation: int = -1
    stage_id: int = -1
    chunks: list[ChunkRecord] = field(default_factory=list)
    created: float = 0.0
    manifest_hash: str = ""

    MANIFEST_NAME = "manifest.json"

    # ── hashing ───────────────────────────────────────────────────────

    def _hash_payload(self) -> dict:
        """Content used for the stable hash (excludes volatile/derived fields).

        ``created`` and ``manifest_hash`` are deliberately omitted so the
        hash is a pure content identity: two ranks producing identical chunk
        layouts at different wall-clock times hash equal.
        """
        return {
            "rank": self.rank,
            "generation": self.generation,
            "parent_generation": self.parent_generation,
            "stage_id": self.stage_id,
            "n_chunks": self.n_chunks,
            "chunk_size": self.chunk_size,
            "dtype": self.dtype,
            "circuit_hash": self.circuit_hash,
            "chunks": [c.to_dict() for c in self.chunks],
        }

    def compute_hash(self) -> str:
        return _stable_hash(self._hash_payload())

    def seal(self) -> "RankManifest":
        """Populate ``manifest_hash`` from current content.  Returns self."""
        self.manifest_hash = self.compute_hash()
        return self

    def verify_self_hash(self) -> bool:
        """True if the stored hash matches the recomputed content hash."""
        return bool(self.manifest_hash) and \
            self.manifest_hash == self.compute_hash()

    # ── validation of the structure itself ────────────────────────────

    def validate(self) -> None:
        if self.n_chunks != len(self.chunks):
            raise ValueError(
                f"n_chunks={self.n_chunks} != len(chunks)={len(self.chunks)}")
        seen = set()
        for c in self.chunks:
            if c.index in seen:
                raise ValueError(f"duplicate chunk index {c.index}")
            seen.add(c.index)

    # ── (de)serialization ─────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "rank": self.rank,
            "generation": self.generation,
            "parent_generation": self.parent_generation,
            "stage_id": self.stage_id,
            "n_chunks": self.n_chunks,
            "chunk_size": self.chunk_size,
            "dtype": self.dtype,
            "circuit_hash": self.circuit_hash,
            "chunks": [c.to_dict() for c in self.chunks],
            "created": self.created,
            "manifest_hash": self.manifest_hash,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RankManifest":
        return cls(
            rank=int(d["rank"]),
            generation=int(d["generation"]),
            parent_generation=int(d.get("parent_generation", -1)),
            stage_id=int(d.get("stage_id", -1)),
            n_chunks=int(d["n_chunks"]),
            chunk_size=int(d["chunk_size"]),
            dtype=str(d["dtype"]),
            circuit_hash=str(d["circuit_hash"]),
            chunks=[ChunkRecord.from_dict(c) for c in d.get("chunks", [])],
            created=float(d.get("created", 0.0)),
            manifest_hash=str(d.get("manifest_hash", "")),
        )

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_json(cls, raw: str | bytes) -> "RankManifest":
        if isinstance(raw, bytes):
            raw = raw.decode()
        return cls.from_dict(json.loads(raw))

    # ── atomic file I/O ───────────────────────────────────────────────

    def write_atomic(self, gen_dir: str | Path, *,
                     after_tmp_fsync=None) -> Path:
        """Write manifest.json into ``gen_dir`` (tmp + fsync + rename).

        Implements commit-protocol steps 5–7.  Seals the hash first so the
        persisted file always carries a hash consistent with its content.

        ``after_tmp_fsync`` is an optional hook called *after* manifest.tmp is
        written and fsynced (step 6) but *before* it is renamed into place
        (step 7).  It exists solely so the fault injector can crash between
        those two steps; it is ``None`` (no-op) in normal operation.
        """
        self.validate()
        self.seal()
        d = Path(gen_dir)
        d.mkdir(parents=True, exist_ok=True)
        final = d / self.MANIFEST_NAME
        tmp = d / (self.MANIFEST_NAME + ".tmp")
        raw = self.to_json()
        with open(tmp, "w") as f:
            f.write(raw)
            f.flush()
            os.fsync(f.fileno())
        if after_tmp_fsync is not None:
            after_tmp_fsync()
        os.replace(str(tmp), str(final))
        _fsync_dir(d)          # make the manifest.json rename durable
        return final

    @classmethod
    def read(cls, gen_dir: str | Path) -> "RankManifest":
        p = Path(gen_dir) / cls.MANIFEST_NAME
        with open(p) as f:
            return cls.from_json(f.read())

    @staticmethod
    def exists(gen_dir: str | Path) -> bool:
        return (Path(gen_dir) / RankManifest.MANIFEST_NAME).exists()
