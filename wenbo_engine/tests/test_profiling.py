"""Tests for the profiling package (stage / io / mpi profilers)."""
from __future__ import annotations

import csv
from threading import Thread

from wenbo_engine.profiling import (
    StageProfiler, IOProfiler, MPIProfiler, NullStageHandle, STAGE_COLUMNS,
)
from wenbo_engine.profiling.io_profiler import IO_COLUMNS
from wenbo_engine.profiling.mpi_profiler import MPI_COLUMNS


def test_stage_row_has_full_schema():
    sp = StageProfiler()
    with sp.stage(0, local_ops=3, rank_nonlocal_ops=1, mpi_nonlocal_ops=2) as h:
        with h.read(1024):
            pass
        with h.kernel():
            pass
        with h.write(2048):
            pass
        with h.commit():
            pass
        with h.checksum():
            pass
    assert len(sp.rows) == 1
    row = sp.rows[0]
    assert set(row.keys()) == set(STAGE_COLUMNS)
    assert row["local_ops"] == 3
    assert row["rank_nonlocal_ops"] == 1
    assert row["mpi_nonlocal_ops"] == 2
    assert row["bytes_read"] == 1024
    assert row["bytes_written"] == 2048
    assert row["recovery_mode"] == "normal"
    # timers recorded non-negative durations
    for c in ("read_sec", "write_sec", "kernel_sec", "commit_sec", "checksum_sec"):
        assert row[c] >= 0.0


def test_stage_forwards_events_to_io_and_mpi():
    io, mpi = IOProfiler(), MPIProfiler()
    sp = StageProfiler(io_profiler=io, mpi_profiler=mpi)
    with sp.stage(0) as h:
        with h.read(100):
            pass
        with h.write(200):
            pass
        with h.mpi(300, peer=1):
            pass
    assert len(io.events) == 2
    assert len(mpi.events) == 1
    t = io.totals()
    assert t["read_bytes"] == 100
    assert t["write_bytes"] == 200
    assert mpi.totals()["mpi_bytes_sent"] == 300
    assert sp.rows[0]["mpi_bytes_sent"] == 300


def test_stage_csv_roundtrip(tmp_path):
    sp = StageProfiler()
    with sp.stage(0, local_ops=1):
        pass
    with sp.stage(1, local_ops=2, recovery_mode="resume"):
        pass
    p = sp.to_csv(tmp_path / "stage_profile.csv")
    with open(p, newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == STAGE_COLUMNS
        rows = list(reader)
    assert len(rows) == 2
    assert rows[1]["recovery_mode"] == "resume"


def test_io_and_mpi_csv_headers(tmp_path):
    io = IOProfiler()
    io.record(0, "read", 10, 0.001)
    p = io.to_csv(tmp_path / "io.csv")
    with open(p, newline="") as f:
        assert next(csv.reader(f)) == IO_COLUMNS

    mpi = MPIProfiler()
    mpi.record_sendrecv(0, peer=1, bytes_sent=10, seconds=0.002)
    mpi.record_collective(0, "Barrier", 0.0001)
    p = mpi.to_csv(tmp_path / "mpi.csv")
    with open(p, newline="") as f:
        rows = list(csv.reader(f))
    assert rows[0] == MPI_COLUMNS
    assert len(rows) == 3  # header + 2 events


def test_null_handle_is_noop():
    h = NullStageHandle()
    with h.read(10), h.write(10), h.kernel(), h.commit(), h.checksum():
        pass
    with h.mpi(10, peer=2):
        pass
    h.add_bytes_read(5)
    h.set_recovery_mode("resume")  # accepted, ignored


def test_stage_handle_thread_safe():
    sp = StageProfiler()
    with sp.stage(0) as h:
        def work():
            for _ in range(200):
                with h.read(8):
                    pass
        threads = [Thread(target=work) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    assert sp.rows[0]["bytes_read"] == 4 * 200 * 8


# ── runtime_timers (calibrated-cost-model telemetry) ────────────────────

from wenbo_engine.profiling import runtime_timers as rt


def test_runtime_timers_add_and_snapshot():
    rt.reset()
    rt.add("commit_time", 0.5)
    rt.add("commit_time", 0.25)
    rt.add_count("commit_count_runtime", 3)
    snap = rt.snapshot()
    assert snap["commit_time"] == 0.75
    assert snap["commit_count_runtime"] == 3


def test_runtime_timers_timed_context():
    rt.reset()
    with rt.timed("norm_time"):
        sum(range(10000))
    assert rt.get("norm_time") > 0
    # snapshot is a copy — mutating it does not affect the live timers
    s = rt.snapshot(); s["norm_time"] = -1
    assert rt.get("norm_time") > 0


def test_runtime_timers_reset_clears():
    rt.add("extent_pack_time", 1.0)
    rt.reset()
    assert rt.snapshot() == {}
