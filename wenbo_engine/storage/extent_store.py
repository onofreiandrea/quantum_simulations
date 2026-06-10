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


def pack_chunk_files(gen_dir, n_chunks: int, *, chunk_size: int,
                     extent_bytes: int = DEFAULT_EXTENT_BYTES,
                     delete_chunks: bool = True) -> ExtentManifest:
    """Pack an existing ``gen_dir/chunks/`` directory into extent files.

    Reads ``chunk_000000.bin .. chunk_{n-1}.bin``, writes them into extents
    (atomically), and (default) removes the chunk files + the now-empty chunks
    dir.  Returns the sealed :class:`ExtentManifest`.  Used to convert a
    freshly-computed generation (written by the unchanged compute path) into the
    at-rest extent layout before commit.
    """
    from wenbo_engine.storage.block_store import read_chunk, chunk_filename
    gdir = Path(gen_dir)
    cdir = gdir / "chunks"
    chunks = {i: read_chunk(cdir / chunk_filename(i)) for i in range(n_chunks)}
    man = write_generation_extents(gdir, chunks, chunk_size=chunk_size,
                                   extent_bytes=extent_bytes)
    if delete_chunks:
        for i in range(n_chunks):
            (cdir / chunk_filename(i)).unlink(missing_ok=True)
        try:
            cdir.rmdir()
        except OSError:
            pass
    return man


def materialize_to_chunk_files(gen_dir, records, *, force: bool = False) -> None:
    """Unpack extent slices back into ``gen_dir/chunks/chunk_NNNNNN.bin``.

    ``records`` is an iterable of objects with ``index``/``extent_id``/
    ``extent_offset``/``size_bytes`` (rank-manifest ChunkRecords) OR
    ExtentChunkRecords.  No-op for an already-materialized chunks dir unless
    ``force``.  Lets the unchanged compute path read a committed (extent-backed)
    generation as plain chunk files.
    """
    from wenbo_engine.storage.block_store import write_chunk_atomic, chunk_filename
    gdir = Path(gen_dir)
    cdir = gdir / "chunks"
    for r in records:
        idx = getattr(r, "index", None)
        if idx is None:
            idx = r.chunk_id
        eid = getattr(r, "extent_id")
        off = getattr(r, "extent_offset", None)
        if off is None:
            off = r.offset
        length = getattr(r, "size_bytes", None)
        if length is None:
            length = r.length
        dst = cdir / chunk_filename(idx)
        if dst.exists() and not force:
            continue
        path = gdir / EXTENTS_DIRNAME / extent_filename(eid)
        with open(path, "rb") as f:
            f.seek(off)
            raw = f.read(length)
        write_chunk_atomic(dst, np.frombuffer(raw, dtype=DTYPE).copy())


def reconstruct_extents_from_chunks(gen_dir, records, *,
                                    delete_chunks: bool = True) -> None:
    """Rebuild extent files from chunk files using recorded offsets.

    ``records`` = rank-manifest ChunkRecords (index/extent_id/extent_offset/
    size_bytes).  Used by durable restore: chunks are downloaded as files, then
    re-packed into the exact extent layout the manifest describes, so the
    restored generation is extent-backed and validates against its manifest.
    """
    from collections import defaultdict
    from wenbo_engine.storage.block_store import read_chunk, chunk_filename
    gdir = Path(gen_dir)
    cdir = gdir / "chunks"
    edir = gdir / EXTENTS_DIRNAME
    edir.mkdir(parents=True, exist_ok=True)
    by_ext = defaultdict(list)
    for r in records:
        by_ext[r.extent_id].append(r)
    for eid, recs in by_ext.items():
        recs.sort(key=lambda r: r.extent_offset)
        tmp = edir / (extent_filename(eid) + ".tmp")
        with open(tmp, "wb") as f:
            for r in recs:
                data = read_chunk(cdir / chunk_filename(r.index))
                f.write(np.ascontiguousarray(data, dtype=DTYPE).tobytes())
            f.flush()
            os.fsync(f.fileno())
        os.replace(str(tmp), str(edir / extent_filename(eid)))
    if delete_chunks:
        for r in records:
            (cdir / chunk_filename(r.index)).unlink(missing_ok=True)
        try:
            cdir.rmdir()
        except OSError:
            pass


def read_chunk_from_extent(gen_dir, extent_id: int, offset: int,
                           length: int) -> np.ndarray:
    """Read one logical chunk directly from its extent slice (seek + read).

    Direct overlay read path: no materialize-to-chunk-file step.  ``offset`` /
    ``length`` come from the rank manifest's extent record for the chunk.
    """
    path = Path(gen_dir) / EXTENTS_DIRNAME / extent_filename(extent_id)
    with open(path, "rb") as f:
        f.seek(offset)
        raw = f.read(length)
    if len(raw) != length:
        raise ValueError(f"extent {extent_id}: short read ({len(raw)} != {length})")
    return np.frombuffer(raw, dtype=DTYPE).copy()


class ExtentWriter:
    """Streaming, atomic writer for a generation's destination extents.

    Direct overlay write path: chunks are appended (any order) to extent files;
    each ``append`` records the exact ``(extent_id, offset, length, checksum)``.
    Extents roll past ``extent_bytes``.  ``finalize`` fsyncs every extent and
    atomically renames it into place, then returns the sealed ExtentManifest —
    partial extent data is never published; the generation becomes committed
    only via the global commit record.
    """

    def __init__(self, gen_dir, chunk_size: int,
                 extent_bytes: int = DEFAULT_EXTENT_BYTES):
        self.edir = Path(gen_dir) / EXTENTS_DIRNAME
        self.edir.mkdir(parents=True, exist_ok=True)
        self.chunk_size = chunk_size
        self.extent_bytes = extent_bytes
        self.records: dict[int, ExtentChunkRecord] = {}
        self.extent_id = 0
        self._f = None
        self._tmp = None
        self._off = 0
        self._open()

    def _open(self):
        self._tmp = self.edir / (extent_filename(self.extent_id) + ".tmp")
        self._f = open(self._tmp, "wb")
        self._off = 0

    def _close(self):
        if self._f is None:
            return
        self._f.flush()
        os.fsync(self._f.fileno())          # req 4: fsync extent output
        self._f.close()
        os.replace(str(self._tmp),
                   str(self.edir / extent_filename(self.extent_id)))
        self._f = None

    def append(self, chunk_id: int, arr: np.ndarray) -> ExtentChunkRecord:
        payload = np.ascontiguousarray(arr, dtype=DTYPE).tobytes()
        if self._off > 0 and self._off + len(payload) > self.extent_bytes:
            self._close()
            self.extent_id += 1
            self._open()
        self._f.write(payload)
        rec = ExtentChunkRecord(chunk_id=chunk_id, extent_id=self.extent_id,
                                offset=self._off, length=len(payload),
                                checksum=_sha256_bytes(payload))
        self.records[chunk_id] = rec
        self._off += len(payload)
        return rec

    def finalize(self) -> ExtentManifest:
        self._close()
        return ExtentManifest(
            n_chunks=len(self.records), n_extents=self.extent_id + 1,
            chunk_size=self.chunk_size, dtype=str(np.dtype(DTYPE)),
            records=self.records).seal()


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
