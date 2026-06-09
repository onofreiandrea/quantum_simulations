"""Per-stage profiler → stage_profile.csv.

One row per simulation step (a levelized layer, or an Atlas-style stage),
with the fixed schema required by the experiment harness:

    step_or_stage_id, local_ops, rank_nonlocal_ops, mpi_nonlocal_ops,
    read_sec, write_sec, kernel_sec, mpi_sec, commit_sec, checksum_sec,
    bytes_read, bytes_written, mpi_bytes_sent, recovery_mode

A runner instruments a stage like this::

    with profiler.stage(idx, local_ops=..., rank_nonlocal_ops=...) as h:
        with h.read(nbytes):  data = read_chunk(...)
        with h.kernel():      apply_1q(data, ...)
        with h.write(nbytes): write_chunk_atomic(...)
        with h.mpi(nbytes, peer=p): comm.Sendrecv(...)
        with h.commit():      wal.commit_step(...)

All timers are context managers and lock-guarded, so the threaded runners
(reader / worker / writer on separate threads) can share one handle.  Each
``*_sec`` value is the **total time spent inside that kind of operation**
during the stage, summed across threads — for pipelined runners these may
overlap in wall-clock time, which is the honest interpretation of where the
work went.  The handle also forwards fine-grained events to an optional
``IOProfiler`` / ``MPIProfiler`` so the io_profile.csv and mpi_profile.csv
artifacts stay consistent with the stage rows.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock
from time import perf_counter

STAGE_COLUMNS = [
    "step_or_stage_id",
    "local_ops",
    "rank_nonlocal_ops",
    "mpi_nonlocal_ops",
    "read_sec",
    "write_sec",
    "kernel_sec",
    "mpi_sec",
    "commit_sec",
    "checksum_sec",
    "bytes_read",
    "bytes_written",
    "mpi_bytes_sent",
    "recovery_mode",
]

_SEC_BUCKETS = ("read", "write", "kernel", "mpi", "commit", "checksum")


class _NoopTimer:
    """A context manager that does nothing — used by NullStageHandle."""
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


_NOOP = _NoopTimer()


class NullStageHandle:
    """Zero-overhead stand-in used when profiling is disabled.

    Exposes the same surface as ``StageHandle`` but every timer is a no-op
    context manager and every counter is dropped, so a runner can be written
    against the handle API unconditionally without paying for instrumentation
    when no profiler is attached.
    """
    stage_id = None
    recovery_mode = "normal"

    def read(self, nbytes: int = 0):
        return _NOOP

    def write(self, nbytes: int = 0):
        return _NOOP

    def mpi(self, nbytes: int = 0, peer: int = -1, op: str = "Sendrecv"):
        return _NOOP

    def kernel(self):
        return _NOOP

    def commit(self):
        return _NOOP

    def checksum(self):
        return _NOOP

    def add_bytes_read(self, n: int):
        pass

    def add_bytes_written(self, n: int):
        pass

    def add_mpi_bytes_sent(self, n: int):
        pass

    def set_recovery_mode(self, mode: str):
        pass


NULL_STAGE_HANDLE = NullStageHandle()


class StageHandle:
    """Thread-safe accumulator for one stage. Use the timer context managers."""

    def __init__(self, stage_id, local_ops: int, rank_nonlocal_ops: int,
                 mpi_nonlocal_ops: int, recovery_mode: str = "normal",
                 io_profiler=None, mpi_profiler=None):
        self.stage_id = stage_id
        self.local_ops = int(local_ops)
        self.rank_nonlocal_ops = int(rank_nonlocal_ops)
        self.mpi_nonlocal_ops = int(mpi_nonlocal_ops)
        self.recovery_mode = recovery_mode
        self._io = io_profiler
        self._mpi = mpi_profiler
        self._lock = Lock()
        self._sec: dict[str, float] = defaultdict(float)
        self._bytes: dict[str, int] = defaultdict(int)

    # ── timers ──────────────────────────────────────────────────────────
    @contextmanager
    def _timer(self, bucket: str):
        t0 = perf_counter()
        try:
            yield
        finally:
            dt = perf_counter() - t0
            with self._lock:
                self._sec[bucket] += dt

    @contextmanager
    def read(self, nbytes: int = 0):
        t0 = perf_counter()
        try:
            yield
        finally:
            dt = perf_counter() - t0
            with self._lock:
                self._sec["read"] += dt
                self._bytes["read"] += int(nbytes)
            if self._io is not None and nbytes:
                self._io.record(self.stage_id, "read", nbytes, dt)

    @contextmanager
    def write(self, nbytes: int = 0):
        t0 = perf_counter()
        try:
            yield
        finally:
            dt = perf_counter() - t0
            with self._lock:
                self._sec["write"] += dt
                self._bytes["write"] += int(nbytes)
            if self._io is not None and nbytes:
                self._io.record(self.stage_id, "write", nbytes, dt)

    @contextmanager
    def mpi(self, nbytes: int = 0, peer: int = -1, op: str = "Sendrecv"):
        t0 = perf_counter()
        try:
            yield
        finally:
            dt = perf_counter() - t0
            with self._lock:
                self._sec["mpi"] += dt
                self._bytes["mpi_sent"] += int(nbytes)
            if self._mpi is not None:
                self._mpi.record(self.stage_id, "sendrecv", op,
                                 bytes_sent=nbytes, seconds=dt, peer=peer)

    def kernel(self):
        return self._timer("kernel")

    def commit(self):
        return self._timer("commit")

    def checksum(self):
        return self._timer("checksum")

    # ── manual counters (when a region is not wrapped by a timer) ────────
    def add_bytes_read(self, n: int):
        with self._lock:
            self._bytes["read"] += int(n)

    def add_bytes_written(self, n: int):
        with self._lock:
            self._bytes["write"] += int(n)

    def add_mpi_bytes_sent(self, n: int):
        with self._lock:
            self._bytes["mpi_sent"] += int(n)

    def set_recovery_mode(self, mode: str):
        self.recovery_mode = mode

    # ── row ─────────────────────────────────────────────────────────────
    def row(self) -> dict:
        with self._lock:
            return {
                "step_or_stage_id": self.stage_id,
                "local_ops": self.local_ops,
                "rank_nonlocal_ops": self.rank_nonlocal_ops,
                "mpi_nonlocal_ops": self.mpi_nonlocal_ops,
                "read_sec": round(self._sec["read"], 6),
                "write_sec": round(self._sec["write"], 6),
                "kernel_sec": round(self._sec["kernel"], 6),
                "mpi_sec": round(self._sec["mpi"], 6),
                "commit_sec": round(self._sec["commit"], 6),
                "checksum_sec": round(self._sec["checksum"], 6),
                "bytes_read": self._bytes["read"],
                "bytes_written": self._bytes["write"],
                "mpi_bytes_sent": self._bytes["mpi_sent"],
                "recovery_mode": self.recovery_mode,
            }


@dataclass
class StageProfiler:
    io_profiler: object | None = None
    mpi_profiler: object | None = None
    rows: list[dict] = field(default_factory=list)
    _lock: Lock = field(default_factory=Lock, repr=False)

    @contextmanager
    def stage(self, stage_id, local_ops: int = 0, rank_nonlocal_ops: int = 0,
              mpi_nonlocal_ops: int = 0, recovery_mode: str = "normal"):
        h = StageHandle(
            stage_id, local_ops, rank_nonlocal_ops, mpi_nonlocal_ops,
            recovery_mode=recovery_mode,
            io_profiler=self.io_profiler, mpi_profiler=self.mpi_profiler,
        )
        try:
            yield h
        finally:
            row = h.row()
            with self._lock:
                self.rows.append(row)

    # ── derived ────────────────────────────────────────────────────────
    def totals(self) -> dict:
        agg = {c: 0 for c in (
            "local_ops", "rank_nonlocal_ops", "mpi_nonlocal_ops",
            "bytes_read", "bytes_written", "mpi_bytes_sent")}
        for c in ("read_sec", "write_sec", "kernel_sec", "mpi_sec",
                  "commit_sec", "checksum_sec"):
            agg[c] = 0.0
        for r in self.rows:
            for c in agg:
                agg[c] += r[c]
        agg["n_stages"] = len(self.rows)
        agg["wall_sec_lower_bound"] = round(
            agg["read_sec"] + agg["write_sec"] + agg["kernel_sec"]
            + agg["mpi_sec"] + agg["commit_sec"] + agg["checksum_sec"], 6)
        return agg

    def to_csv(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=STAGE_COLUMNS)
            w.writeheader()
            for r in self.rows:
                w.writerow(r)
        return path
