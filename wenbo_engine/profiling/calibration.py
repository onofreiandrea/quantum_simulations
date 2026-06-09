"""Machine calibration → cost_model.json.

Micro-benchmarks the primitives the simulator's hot path depends on so a
run can be predicted and compared against actuals:

  * NVMe sequential read / write bandwidth (via the real block_store
    chunk read/write path, including the atomic tmp+rename+fsync write)
  * fsync cost (per call)
  * rename cost (os.replace, per call)
  * checksum throughput (crc32 over an in-RAM buffer)
  * MPI Sendrecv bandwidth + collective (Barrier/Allreduce) cost, only when
    MPI is available and there is more than one rank

Nothing here runs in the simulation hot loop — it is a separate, explicit
calibration pass executed once at the start of an experiment.
"""
from __future__ import annotations

import json
import os
import zlib
from pathlib import Path
from time import perf_counter

import numpy as np

from wenbo_engine.storage.block_store import (
    DTYPE, chunk_filename, read_chunk, write_chunk_atomic,
)

_ITEMSIZE = np.dtype(DTYPE).itemsize


class CalibrationRunner:
    """Measure NVMe / checksum / (optional) MPI primitive costs."""

    def __init__(self, work_dir: str | Path,
                 chunk_size: int = 1 << 20, n_chunks: int = 8,
                 comm=None):
        self.work_dir = Path(work_dir)
        self.chunk_size = chunk_size
        self.n_chunks = n_chunks
        self.comm = comm

    # ── individual measurements ─────────────────────────────────────────
    def measure_nvme(self) -> dict:
        d = self.work_dir / "calib_nvme"
        d.mkdir(parents=True, exist_ok=True)
        bytes_per_chunk = self.chunk_size * _ITEMSIZE
        total_bytes = bytes_per_chunk * self.n_chunks
        rng = np.random.default_rng(0)
        data = rng.standard_normal(self.chunk_size).astype(np.float32).view(DTYPE)

        t0 = perf_counter()
        for i in range(self.n_chunks):
            write_chunk_atomic(d / chunk_filename(i), data)
        t_write = perf_counter() - t0

        t0 = perf_counter()
        for i in range(self.n_chunks):
            _ = read_chunk(d / chunk_filename(i))
        t_read = perf_counter() - t0

        for i in range(self.n_chunks):
            (d / chunk_filename(i)).unlink(missing_ok=True)

        return {
            "chunk_bytes": bytes_per_chunk,
            "n_chunks": self.n_chunks,
            "read_bandwidth_MBps": (total_bytes / 1e6 / t_read) if t_read > 0 else 0.0,
            "write_bandwidth_MBps": (total_bytes / 1e6 / t_write) if t_write > 0 else 0.0,
            "read_sec": t_read,
            "write_sec": t_write,
        }

    def measure_fsync(self, reps: int = 32) -> dict:
        d = self.work_dir / "calib_fsync"
        d.mkdir(parents=True, exist_ok=True)
        p = d / "fsync_probe.bin"
        buf = b"\x00" * 4096
        t_total = 0.0
        with open(p, "wb") as f:
            for _ in range(reps):
                f.write(buf)
                f.flush()
                t0 = perf_counter()
                os.fsync(f.fileno())
                t_total += perf_counter() - t0
        p.unlink(missing_ok=True)
        return {"fsync_sec_per_call": t_total / reps, "reps": reps}

    def measure_rename(self, reps: int = 64) -> dict:
        d = self.work_dir / "calib_rename"
        d.mkdir(parents=True, exist_ok=True)
        src = d / "r_src.tmp"
        dst = d / "r_dst.bin"
        t_total = 0.0
        for _ in range(reps):
            with open(src, "wb") as f:
                f.write(b"x")
            t0 = perf_counter()
            os.replace(str(src), str(dst))
            t_total += perf_counter() - t0
        dst.unlink(missing_ok=True)
        return {"rename_sec_per_call": t_total / reps, "reps": reps}

    def measure_checksum(self, reps: int = 8) -> dict:
        bytes_per_chunk = self.chunk_size * _ITEMSIZE
        rng = np.random.default_rng(1)
        data = rng.standard_normal(self.chunk_size).astype(np.float32).view(DTYPE)
        raw = data.tobytes()
        # warm up
        zlib.crc32(raw)
        t0 = perf_counter()
        for _ in range(reps):
            zlib.crc32(raw)
        dt = perf_counter() - t0
        total_bytes = bytes_per_chunk * reps
        return {
            "algorithm": "crc32",
            "checksum_throughput_MBps": (total_bytes / 1e6 / dt) if dt > 0 else 0.0,
            "checksum_sec_per_chunk": dt / reps,
        }

    def measure_mpi(self) -> dict:
        """Sendrecv bandwidth + collective cost. Requires MPI with size > 1."""
        comm = self.comm
        if comm is None:
            try:
                from mpi4py import MPI
                comm = MPI.COMM_WORLD
            except Exception:
                return {"available": False, "reason": "mpi4py not importable"}

        size = comm.Get_size()
        rank = comm.Get_rank()
        if size < 2:
            return {"available": False, "reason": "single rank"}

        from mpi4py import MPI

        send = np.ones(self.chunk_size, dtype=DTYPE)
        recv = np.empty(self.chunk_size, dtype=DTYPE)
        partner = rank ^ 1
        nbytes = self.chunk_size * _ITEMSIZE

        comm.Barrier()
        t0 = perf_counter()
        reps = 8
        for _ in range(reps):
            comm.Sendrecv(sendbuf=send, dest=partner, recvbuf=recv, source=partner)
        t_sr = (perf_counter() - t0) / reps

        # collective: Barrier
        comm.Barrier()
        t0 = perf_counter()
        for _ in range(reps):
            comm.Barrier()
        t_barrier = (perf_counter() - t0) / reps

        # collective: Allreduce of a scalar
        scal = np.array(1.0)
        out = np.array(0.0)
        t0 = perf_counter()
        for _ in range(reps):
            comm.Allreduce(scal, out, op=MPI.SUM)
        t_allreduce = (perf_counter() - t0) / reps

        # aggregate across ranks (max = slowest link)
        sr_bw = (nbytes / 1e6 / t_sr) if t_sr > 0 else 0.0
        return {
            "available": True,
            "n_ranks": size,
            "sendrecv_sec": float(comm.allreduce(t_sr, op=MPI.MAX)),
            "sendrecv_bandwidth_MBps": float(comm.allreduce(sr_bw, op=MPI.MIN)),
            "barrier_sec": float(comm.allreduce(t_barrier, op=MPI.MAX)),
            "allreduce_sec": float(comm.allreduce(t_allreduce, op=MPI.MAX)),
        }

    # ── orchestration ───────────────────────────────────────────────────
    def run(self, include_mpi: bool = True) -> dict:
        self.work_dir.mkdir(parents=True, exist_ok=True)
        model = {
            "chunk_size": self.chunk_size,
            "dtype": str(np.dtype(DTYPE)),
            "itemsize": _ITEMSIZE,
            "nvme": self.measure_nvme(),
            "fsync": self.measure_fsync(),
            "rename": self.measure_rename(),
            "checksum": self.measure_checksum(),
        }
        if include_mpi:
            model["mpi"] = self.measure_mpi()
        else:
            model["mpi"] = {"available": False, "reason": "disabled"}
        return model

    def write(self, path: str | Path, include_mpi: bool = True) -> dict:
        model = self.run(include_mpi=include_mpi)
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(model, f, indent=2)
        return model
