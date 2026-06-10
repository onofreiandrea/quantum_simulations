"""Extent store: pack many logical chunks into few physical extent files.

The *extent* layout writes a generation's logical chunks into a handful of
``extents/extent_NNNN.dat`` files (each a concatenation of chunk payloads)
instead of one ``chunk_NNNNNN.bin`` per chunk, plus an
:class:`~wenbo_engine.storage.extent_manifest.ExtentManifest` mapping every
``chunk_id -> (extent_id, offset, length, checksum)``.

Logical chunk semantics are unchanged: a caller still reads/writes a numpy
``complex64`` array per ``chunk_id``.  This module only changes the physical
packing, so kernels and the MPI exchange are untouched.

Atomicity: extents are written to ``*.dat.tmp``, fsync'd, then atomically
renamed; the manifest is written atomically last.  Partial/incomplete extent
data is never visible as committed progress — commitment is still owned solely
by the generation's global commit record (this module writes the per-rank
generation payload; the recovery layer decides if the generation is committed).
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np

from wenbo_engine.storage.block_store import DTYPE
from wenbo_engine.storage.extent_manifest import (
    ExtentManifest, ExtentChunkRecord, extent_filename, EXTENTS_DIRNAME,
)

_ITEMSIZE = np.dtype(DTYPE).itemsize
# Default max bytes per extent file before rolling to a new one (128 MiB).
DEFAULT_EXTENT_BYTES = 128 * 1024 * 1024


def _sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def write_generation_extents(gen_dir, chunks: dict[int, np.ndarray], *,
                             chunk_size: int,
                             extent_bytes: int = DEFAULT_EXTENT_BYTES
                             ) -> ExtentManifest:
    """Pack ``{chunk_id: array}`` into extent files under ``gen_dir/extents``.

    Returns a sealed :class:`ExtentManifest`.  Writes each extent atomically
    (tmp + fsync + rename); the manifest is sealed but NOT written to disk here
    (the caller writes it atomically as the last step, so a crash mid-write
    leaves no manifest -> the generation is not recoverable).
    """
    gdir = Path(gen_dir)
    edir = gdir / EXTENTS_DIRNAME
    edir.mkdir(parents=True, exist_ok=True)

    records: dict[int, ExtentChunkRecord] = {}
    extent_id = 0
    cur_tmp = None
    cur_f = None
    cur_off = 0

    def _open_new():
        nonlocal cur_tmp, cur_f, cur_off
        cur_tmp = edir / (extent_filename(extent_id) + ".tmp")
        cur_f = open(cur_tmp, "wb")
        cur_off = 0

    def _close_and_rename():
        nonlocal cur_f
        if cur_f is None:
            return
        cur_f.flush()
        os.fsync(cur_f.fileno())
        cur_f.close()
        os.replace(str(cur_tmp), str(edir / extent_filename(extent_id)))
        cur_f = None

    _open_new()
    for cid in sorted(chunks):
        arr = np.ascontiguousarray(chunks[cid], dtype=DTYPE)
        payload = arr.tobytes()
        # roll to a new extent if this chunk would overflow the budget
        # (but never split a single chunk across extents).
        if cur_off > 0 and cur_off + len(payload) > extent_bytes:
            _close_and_rename()
            extent_id += 1
            _open_new()
        cur_f.write(payload)
        records[cid] = ExtentChunkRecord(
            chunk_id=cid, extent_id=extent_id, offset=cur_off,
            length=len(payload), checksum=_sha256_bytes(payload))
        cur_off += len(payload)
    _close_and_rename()

    man = ExtentManifest(n_chunks=len(chunks), n_extents=extent_id + 1,
                         chunk_size=chunk_size, dtype=str(np.dtype(DTYPE)),
                         records=records)
    return man.seal()


def read_logical_chunk(gen_dir, manifest: ExtentManifest,
                       chunk_id: int) -> np.ndarray:
    """Read one logical chunk by id from its extent (seek + read length)."""
    rec = manifest.record(chunk_id)
    path = Path(gen_dir) / EXTENTS_DIRNAME / extent_filename(rec.extent_id)
    with open(path, "rb") as f:
        f.seek(rec.offset)
        raw = f.read(rec.length)
    if len(raw) != rec.length:
        raise ValueError(
            f"chunk {chunk_id}: extent {rec.extent_id} short read "
            f"({len(raw)} != {rec.length})")
    return np.frombuffer(raw, dtype=DTYPE).copy()


def validate_extent_chunk(gen_dir, rec: ExtentChunkRecord, *,
                          check_checksum: bool = False) -> str | None:
    """Validate one chunk's extent slice.  Returns None if ok, else a reason.

    Checks: extent file exists; its size covers ``offset + length`` (catches a
    missing/short/partial extent and wrong offset); optionally the slice's
    sha256 matches the recorded checksum (catches same-size corruption).
    """
    path = Path(gen_dir) / EXTENTS_DIRNAME / extent_filename(rec.extent_id)
    if not path.exists():
        return f"chunk {rec.chunk_id}: extent {rec.extent_id} missing"
    size = path.stat().st_size
    if size < rec.offset + rec.length:
        return (f"chunk {rec.chunk_id}: extent {rec.extent_id} too short "
                f"(size {size} < {rec.offset + rec.length})")
    if check_checksum:
        with open(path, "rb") as f:
            f.seek(rec.offset)
            raw = f.read(rec.length)
        if _sha256_bytes(raw) != rec.checksum:
            return f"chunk {rec.chunk_id}: extent {rec.extent_id} checksum mismatch"
    return None


def validate_extent_generation(gen_dir, manifest: ExtentManifest, *,
                               check_checksums: bool = False) -> list[str]:
    """Validate every chunk in the manifest; return a list of failure reasons."""
    failures = []
    if not manifest.verify_self_hash():
        failures.append("extent manifest self-hash mismatch")
    for cid in sorted(manifest.records):
        reason = validate_extent_chunk(gen_dir, manifest.records[cid],
                                       check_checksum=check_checksums)
        if reason:
            failures.append(reason)
    return failures
