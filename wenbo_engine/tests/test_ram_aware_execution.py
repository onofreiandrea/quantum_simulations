"""RAM-aware execution control: budgets, early-fail, bounded overlay/cache.

The 8-node ladder OOM'd because the per-rank working set (compute-unit overlay
holding the whole partition, or the gate-aware remote cache growing unbounded)
exceeded node RAM, even though NVMe stayed >93% free.  These tests pin the
fix: bounded overlay (streams), bounded remote cache (LRU-evicts), early
failure before allocation, and end-to-end correctness with auto-selected
smaller chunk_bits — while generation recovery / gate-aware MPI / extents+direct
still work.
"""
import glob
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from wenbo_engine.storage.block_store import DTYPE
from wenbo_engine.mpi.remote_buffer_cache import RemoteBufferCache
from wenbo_engine.runtime.memory_overlay import MemoryOverlay

REPO = str(Path(__file__).resolve().parent.parent.parent)


# ── case 4: runner fails early before allocation if a chunk can't fit RAM ─

def test_apply_ram_budgets_fails_early_when_chunk_exceeds_budget():
    from wenbo_engine.mpi import mpi_runner
    # chunk_size 2^24 complex64 = 128 MiB; budget 0.05 GiB can't hold it + temp
    with pytest.raises(RuntimeError, match="RAM-infeasible"):
        mpi_runner._apply_ram_budgets(
            chunk_size=1 << 24, ram_budget_gib=0.05,
            max_overlay_chunks=None, max_remote_buffer_gib=None)


def test_apply_ram_budgets_derives_overlay_and_remote_budgets():
    from wenbo_engine.mpi import mpi_runner
    meta = mpi_runner._apply_ram_budgets(
        chunk_size=1 << 20, ram_budget_gib=1.0,    # 8 MiB chunk, 1 GiB budget
        max_overlay_chunks=None, max_remote_buffer_gib=None)
    assert meta["ram_budget_gib"] == 1.0
    assert meta["max_overlay_chunks"] >= 1
    assert mpi_runner._RAM["overlay_budget_chunks"] >= 1
    assert mpi_runner._RAM["remote_buffer_max_bytes"] > 0
    mpi_runner._reset_ram_budgets()   # restore unbounded


# ── case 5: compute_unit overlay respects RAM budget (streams, bounded peak) ─

def test_overlay_bounded_peak_streams():
    cs = 8
    data = {c: np.arange(cs, dtype=DTYPE) + c for c in range(8)}
    writes = []
    ov = MemoryOverlay(reader=lambda c: data[c].copy(),
                       writer=lambda c, a: writes.append(c),
                       ram_budget_chunks=1)
    for c in range(8):                 # a streaming local unit: load→dirty→write
        a = ov.get(c)
        ov.mark_dirty(c)
        ov.writeback(c)
    # with budget=1 the overlay never holds more than ~1 chunk resident
    assert ov.peak_resident_bytes <= 2 * data[0].nbytes
    assert ov.resident_count <= 1
    assert writes == list(range(8))


def test_overlay_unbounded_holds_everything():
    cs = 8
    data = {c: np.arange(cs, dtype=DTYPE) for c in range(8)}
    ov = MemoryOverlay(reader=lambda c: data[c].copy(),
                       writer=lambda c, a: None, ram_budget_chunks=0)
    for c in range(8):
        ov.get(c)                       # never evicts
    assert ov.resident_count == 8
    assert ov.peak_resident_bytes == 8 * data[0].nbytes


# ── case 6: remote_buffer_cache respects max_remote_buffer_gib ────────────

def test_remote_buffer_cache_bounded_evicts_lru():
    chunk = np.zeros(1024, dtype=DTYPE)        # 8 KiB each
    cap = 3 * chunk.nbytes                       # room for 3 chunks
    cache = RemoteBufferCache(max_bytes=cap)
    for i in range(6):
        cache.put(partner_rank=1, chunk_index=i, buf=chunk.copy())
    assert cache._bytes <= cap
    assert cache.peak_bytes <= cap
    assert cache.evictions >= 1
    # the oldest (LRU) entries were evicted; the newest survive
    assert cache.get(1, 5) is not None
    assert cache.get(1, 0) is None


def test_remote_buffer_cache_unbounded_keeps_all():
    chunk = np.zeros(1024, dtype=DTYPE)
    cache = RemoteBufferCache(max_bytes=0)       # unbounded
    for i in range(10):
        cache.put(1, i, chunk.copy())
    assert cache.evictions == 0
    assert cache.get(1, 0) is not None


# ── real-MPI: correctness + invariants with auto chunk_bits (cases 7–11) ──

pytest.importorskip("mpi4py")
_mark = pytest.mark.skipif(shutil.which("mpirun") is None, reason="no mpirun")


def _bench(tmp, kind, n, depth, ranks=2, extra=()):
    out = Path(tmp) / f"out_{kind}_{n}"
    wd = Path(tmp) / f"wd_{kind}_{n}"
    for p in (out, wd):
        shutil.rmtree(p, ignore_errors=True)
    cmd = ["mpirun", "-np", str(ranks), sys.executable, "-m",
           "wenbo_engine.bench.communication_workloads", "--kind", kind,
           "--n", str(n), "--depth", str(depth), "--recovery", "generation",
           "--planner", "recovery_aware_v1", *extra,
           "--output-dir", str(out), "--work-dir", str(wd)]
    r = subprocess.run(cmd, env=dict(os.environ, PYTHONPATH=REPO),
                       capture_output=True, timeout=400)
    assert r.returncode == 0, r.stderr.decode()[-2000:]
    fs = json.load(open(glob.glob(str(out / "**" / "final_summary.json"),
                                  recursive=True)[0]))
    rev = json.load(open(glob.glob(str(out / "**" / "recovery_events.json"),
                                   recursive=True)[0]))
    return fs, rev, out, wd


# A feasible per-rank RAM budget above the model's metadata floor.  At these
# small n the chunk is tiny, so auto-chunk-bits validates the path end-to-end;
# the real chunk_bits REDUCTION is exercised on the cluster (large n).
_BUDGET = ["--ram-budget-gib", "2.0", "--auto-chunk-bits"]


@_mark
def test_auto_chunk_bits_correct_and_invariants_light(tmp_path):
    fs, rev, out, wd = _bench(
        tmp_path, "communication_light", 12, 12, extra=[*_BUDGET, "--verify"])
    assert fs["correct"] is True
    assert abs(fs["final_norm"] - 1.0) < 1e-5
    assert fs["recovery_mode"] == "generation"          # case 8
    assert rev["source_of_truth"] == "global_commit_record"
    assert rev["wal_json_present"] is False
    # extents + direct path still selected + used (case 10)
    assert fs["selected_strategy"] == "extents+compute_unit+direct+gate_aware"
    assert fs["auto_chunk_bits_enabled"] is True
    # final_summary carries the RAM fields (case 11)
    for k in ("ram_budget_gib", "estimated_peak_ram_gib", "chunk_bits",
              "chunk_bytes", "auto_chunk_bits_enabled", "recommended_chunk_bits",
              "max_overlay_chunks", "max_remote_buffer_gib",
              "overlay_peak_ram_gib", "remote_buffer_peak_gib", "ram_feasible"):
        assert k in fs, k
    assert fs["ram_feasible"] is True


@_mark
def test_correct_with_explicit_smaller_chunk_bits(tmp_path):
    # case 7: final state matches reference at a smaller-than-default chunk_bits
    fs, rev, out, wd = _bench(
        tmp_path, "communication_light", 12, 12,
        extra=["--chunk-bits", "6", "--verify"])
    assert fs["chunk_bits"] == 6
    assert fs["correct"] is True
    assert abs(fs["final_norm"] - 1.0) < 1e-5


@_mark
def test_auto_chunk_bits_gate_aware_mpi_still_works(tmp_path):
    # case 9: gate-aware MPI preserved under RAM-aware control
    fs, rev, out, wd = _bench(
        tmp_path, "mpi_nonlocal_heavy", 8, 12, extra=[*_BUDGET, "--verify"])
    assert fs["correct"] is True
    assert "gate_aware" in fs["selected_strategy"]
    assert fs["measured_mpi_nonlocal_ops"] > 0
    assert fs["mpi_bytes_sent"] > 0
    assert fs["recovery_mode"] == "generation"
    assert rev["source_of_truth"] == "global_commit_record"
    assert rev["wal_json_present"] is False
    assert fs["ram_feasible"] is True
