"""Extent manifest: maps logical chunk ids to (extent, offset, length, checksum).

In the *extent* storage layout many logical chunks are packed into a few
physical ``extent_NNNN.dat`` files instead of one file per chunk.  This manifest
is the index that makes a logical chunk addressable again:

    chunk_id -> (extent_id, offset_bytes, length_bytes, sha256)

It is pure data + (de)serialization; no I/O of extent payloads happens here
(see :mod:`wenbo_engine.storage.extent_store`).  The manifest itself is written
atomically (tmp + fsync + rename) and carries a self-hash so a torn or tampered
manifest is detected — mirroring :class:`wenbo_engine.recovery.rank_manifest`.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

EXTENTS_DIRNAME = "extents"


def extent_filename(extent_id: int) -> str:
    return f"extent_{extent_id:04d}.dat"


@dataclass
class ExtentChunkRecord:
    chunk_id: int
    extent_id: int
    offset: int          # byte offset within the extent file
    length: int          # byte length of this chunk's payload
    checksum: str        # sha256 hex of the chunk bytes

    def to_dict(self) -> dict:
        return {"chunk_id": self.chunk_id, "extent_id": self.extent_id,
                "offset": self.offset, "length": self.length,
                "checksum": self.checksum}

    @classmethod
    def from_dict(cls, d: dict) -> "ExtentChunkRecord":
        return cls(int(d["chunk_id"]), int(d["extent_id"]), int(d["offset"]),
                   int(d["length"]), str(d["checksum"]))


@dataclass
class ExtentManifest:
    n_chunks: int
    n_extents: int
    chunk_size: int                     # logical chunk length in elements
    dtype: str = "complex64"
    records: dict[int, ExtentChunkRecord] = field(default_factory=dict)
    manifest_hash: str = ""

    # ── content hash (excludes the hash field + volatile data) ──────────
    def _payload(self) -> dict:
        return {
            "n_chunks": self.n_chunks,
            "n_extents": self.n_extents,
            "chunk_size": self.chunk_size,
            "dtype": self.dtype,
            "records": [self.records[c].to_dict()
                        for c in sorted(self.records)],
        }

    def compute_hash(self) -> str:
        raw = json.dumps(self._payload(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode()).hexdigest()[:32]

    def seal(self) -> "ExtentManifest":
        self.manifest_hash = self.compute_hash()
        return self

    def verify_self_hash(self) -> bool:
        return bool(self.manifest_hash) and self.manifest_hash == self.compute_hash()

    def record(self, chunk_id: int) -> ExtentChunkRecord:
        return self.records[chunk_id]

    # ── (de)serialization ──────────────────────────────────────────────
    def to_dict(self) -> dict:
        d = self._payload()
        d["manifest_hash"] = self.manifest_hash
        d["layout"] = "extents"
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "ExtentManifest":
        recs = {int(r["chunk_id"]): ExtentChunkRecord.from_dict(r)
                for r in d.get("records", [])}
        m = cls(n_chunks=int(d["n_chunks"]), n_extents=int(d["n_extents"]),
                chunk_size=int(d["chunk_size"]), dtype=str(d.get("dtype", "complex64")),
                records=recs, manifest_hash=str(d.get("manifest_hash", "")))
        return m

    def write_atomic(self, gen_dir: str | Path,
                     filename: str = "extent_manifest.json") -> Path:
        if not self.manifest_hash:
            self.seal()
        d = Path(gen_dir)
        d.mkdir(parents=True, exist_ok=True)
        path = d / filename
        tmp = path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            json.dump(self.to_dict(), f, sort_keys=True)
            f.flush()
            os.fsync(f.fileno())
        os.replace(str(tmp), str(path))
        return path

    @classmethod
    def read(cls, gen_dir: str | Path,
             filename: str = "extent_manifest.json") -> "ExtentManifest":
        path = Path(gen_dir) / filename
        with open(path) as f:
            return cls.from_dict(json.load(f))

    @staticmethod
    def exists(gen_dir: str | Path,
               filename: str = "extent_manifest.json") -> bool:
        return (Path(gen_dir) / filename).exists()
