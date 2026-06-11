"""Gate-aware MPI exchange: unit + real-MPI tests.

Unit tests (no MPI): the exchange planner (partner/chunk resolution, grouping)
and the remote-buffer cache (reuse).  MPI tests (real ``mpirun -np 2``, skipped
when mpi4py/mpirun absent): final-state equivalence vs naive, generation
recovery, preserved MPI stress, measured (not estimated) metrics.
"""
import glob
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from wenbo_engine.storage.block_store import DTYPE
from wenbo_engine.mpi.exchange_planner import (
    resolve_partner, classify_gate, plan_stage, group_by_partner, fallback_gates,
)
from wenbo_engine.mpi.remote_buffer_cache import RemoteBufferCache

REPO = str(Path(__file__).resolve().parent.parent.parent)
_U1 = np.eye(2, dtype=DTYPE)
_U2 = np.eye(4, dtype=DTYPE)


# ── 1. partner / chunk resolver ─────────────────────────────────────────

def test_resolve_partner_pairs():
    # k=2, n_local_bits=1 → rank bit 0 is qubit k+n_local_bits = 3.
    rb, partner, low = resolve_partner(rank=0, mpi_q=3, k=2, n_local_bits=1)
    assert rb == 0 and partner == 1 and low is True
    rb, partner, low = resolve_partner(rank=1, mpi_q=3, k=2, n_local_bits=1)
    assert rb == 0 and partner == 0 and low is False
    # qubit 4 → rank bit 1 → partner flips bit 1
    rb, partner, _ = resolve_partner(rank=0, mpi_q=4, k=2, n_local_bits=1)
    assert rb == 1 and partner == 2


def test_classify_gate_kinds_and_chunk_pairs():
    # 1q MPI gate
    ge = classify_gate([3], _U1, rank=0, k=2, n_local_bits=1, n_chunks_per_rank=2)
    assert ge.kind == "1q" and ge.batchable
    assert ge.chunk_pairs == [(0, 0), (1, 1)]      # local idx == remote idx
    # 2q one MPI + one local (qubit 0 < k)
    ge = classify_gate([0, 3], _U2, rank=0, k=2, n_local_bits=1, n_chunks_per_rank=2)
    assert ge.kind == "2q_one_local" and ge.batchable and ge.other_q == 0
    # 2q both MPI → fallback
    ge = classify_gate([3, 4], _U2, rank=0, k=2, n_local_bits=1, n_chunks_per_rank=2)
    assert ge.kind == "fallback" and not ge.batchable


# ── 2. batches group by partner rank ────────────────────────────────────

def test_group_by_partner():
    # k=2, n_local_bits=1 → qubits 3,4 are MPI (rank bits 0,1; partners 1,2).
    ops = [([3], _U1), ([4], _U1), ([0, 3], _U2), ([3, 4], _U2)]
    plan = plan_stage(ops, rank=0, k=2, n_local_bits=1, n_chunks_per_rank=2)
    groups = group_by_partner(plan)
    assert set(groups.keys()) == {1, 2}
    assert len(groups[1]) == 2          # [3] and [0,3] both partner 1
    assert len(groups[2]) == 1          # [4]
    assert len(fallback_gates(plan)) == 1   # [3,4] both-MPI


# ── 3. repeated remote chunk requests are reused ────────────────────────

class _FakeComm:
    def __init__(self, rank=0):
        self.calls = 0
        self._rank = rank

    def Get_rank(self):
        return self._rank

    def Sendrecv(self, sendbuf, dest, recvbuf, source):
        self.calls += 1
        recvbuf[:] = sendbuf            # echo partner = self for the test


def test_remote_buffer_cache_reuse():
    cache = RemoteBufferCache()
    comm = _FakeComm()
    buf = np.arange(4, dtype=DTYPE)
    r1 = cache.exchange(comm, partner_rank=1, chunk_index=0, send_buf=buf, chunk_size=4)
    r2 = cache.exchange(comm, partner_rank=1, chunk_index=0, send_buf=buf, chunk_size=4)
    assert comm.calls == 1               # second request reused, no MPI
    assert cache.hits == 1 and cache.misses == 1
    assert np.array_equal(r1, r2)
    # a different chunk index is a real miss
    cache.exchange(comm, 1, 1, buf, 4)
    assert comm.calls == 2 and cache.misses == 2


# ── 9. metrics are measured (bytes from the real send buffer) ───────────

def test_profiling_comm_measures_real_bytes():
    from wenbo_engine.bench.communication_workloads import ProfilingComm, Metrics
    m = Metrics()
    pc = ProfilingComm(_FakeComm(), m)
    send = np.ones(8, dtype=DTYPE)       # 8 * itemsize bytes
    recv = np.empty(8, dtype=DTYPE)
    pc.Sendrecv(sendbuf=send, dest=1, recvbuf=recv, source=1)
    assert m.sendrecv_count == 1
    assert m.mpi_bytes_sent == send.nbytes      # measured from the actual buffer
    assert frozenset((0, 1)) in m.observed_partner_pairs   # real pair recorded


# ── 12. no shared work_dir assumption: planner uses only local indices ──

def test_planner_uses_only_local_chunk_indices():
    plan = plan_stage([([3], _U1), ([0, 3], _U2)], rank=1, k=2, n_local_bits=1,
                      n_chunks_per_rank=4)
    for ge in plan:
        if ge.batchable:
            # remote chunk index always equals the LOCAL index (same offset on
            # the partner) — never a path into another rank's directory.
            assert all(lci == rci for (lci, rci) in ge.chunk_pairs)
            assert all(0 <= lci < 4 for (lci, _) in ge.chunk_pairs)


# ── real-MPI tests (mpirun -np 2) ───────────────────────────────────────

mpi4py = pytest.importorskip("mpi4py")
import shutil  # noqa: E402

_HAVE_MPIRUN = shutil.which("mpirun") is not None
_pytestmark = pytest.mark.skipif(not _HAVE_MPIRUN, reason="mpirun not available")


def _bench(mode, kind, n, depth, recovery, tmp, *, verify=False):
    out = Path(tmp) / f"out_{mode}_{kind}_{recovery}"
    wd = Path(tmp) / f"wd_{mode}_{kind}_{recovery}"
    shutil.rmtree(out, ignore_errors=True)
    shutil.rmtree(wd, ignore_errors=True)
    cmd = ["mpirun", "-np", "2", sys.executable, "-m",
           "wenbo_engine.bench.communication_workloads",
           "--kind", kind, "--n", str(n), "--depth", str(depth),
           "--recovery", recovery, "--mpi-exchange-mode", mode,
           "--output-dir", str(out), "--work-dir", str(wd)]
    if verify:
        cmd.append("--verify")
    env = dict(os.environ, PYTHONPATH=REPO)
    r = subprocess.run(cmd, env=env, capture_output=True, timeout=400)
    assert r.returncode == 0, r.stderr.decode()[-1500:]
    fs = glob.glob(str(out / "**" / "final_summary.json"), recursive=True)
    return json.load(open(fs[0])), out, wd


@_pytestmark
def test_gate_aware_matches_naive_final_state(tmp_path):
    # case 4: identical correct final state (both verified against ref_dense)
    naive, _, _ = _bench("naive", "mpi_nonlocal_heavy", 6, 6, "generation",
                         tmp_path, verify=True)
    ga, _, _ = _bench("gate_aware", "mpi_nonlocal_heavy", 6, 6, "generation",
                      tmp_path, verify=True)
    assert naive["correct"] is True and ga["correct"] is True
    assert abs(naive["final_norm"] - ga["final_norm"]) < 1e-9


@_pytestmark
def test_gate_aware_with_generation_recovery(tmp_path):
    # cases 5, 7, 10, 11
    s, out, wd = _bench("gate_aware", "mpi_nonlocal_heavy", 8, 8, "generation",
                        tmp_path)
    assert abs(s["final_norm"] - 1.0) < 1e-5
    assert s["recovery_mode"] == "generation"
    assert s["measured_mpi_nonlocal_ops"] > 0        # case 7: MPI stress kept
    # case 10: generation recovery wrote commits + manifests
    assert glob.glob(str(wd / "**" / "commits" / "commit_*.json"), recursive=True)
    assert glob.glob(str(wd / "**" / "generations" / "gen_*" / "manifest.json"),
                     recursive=True)
    # case 11: no wal.json in generation mode
    assert not glob.glob(str(wd / "**" / "wal.json"), recursive=True)


@_pytestmark
def test_gate_aware_preserves_mpi_stress(tmp_path):
    # case 6: gate_aware must NOT silently remove MPI traffic
    s, _, _ = _bench("gate_aware", "mpi_nonlocal_heavy", 8, 8, "generation",
                     tmp_path)
    assert s["mpi_bytes_sent"] > 0
    assert s["sendrecv_count"] > 0
    assert s["metrics_are_measured"] is True         # case 9 end-to-end


@_pytestmark
def test_mixed_staged_keeps_all_phases(tmp_path):
    # case 8: mixed_staged still exercises local + rank-nl + MPI-nl
    s, _, _ = _bench("gate_aware", "mixed_staged", 10, 16, "generation", tmp_path)
    assert s["measured_local_ops"] > 0
    assert s["measured_rank_nonlocal_ops"] > 0
    assert s["measured_mpi_nonlocal_ops"] > 0


# ── RAM-aware: remote buffer cache stays correct under a byte budget ─────

def test_remote_buffer_cache_correct_under_budget():
    import numpy as np
    from wenbo_engine.mpi.remote_buffer_cache import RemoteBufferCache
    from wenbo_engine.storage.block_store import DTYPE
    a = np.arange(256, dtype=DTYPE)
    b = (np.arange(256, dtype=DTYPE) + 1000)
    cache = RemoteBufferCache(max_bytes=2 * a.nbytes)   # holds ~2 chunks
    cache.put(1, 0, a.copy())
    cache.put(1, 1, b.copy())
    assert np.array_equal(cache.get(1, 0), a)           # still cached
    assert np.array_equal(cache.get(1, 1), b)
    cache.put(2, 0, a.copy())                            # exceeds -> LRU evict
    assert cache.evictions >= 1
    assert cache._bytes <= cache.max_bytes


# ── backend selection: nonlocal pair kernel equivalent numpy vs numba ───

def test_nonlocal_kernel_backend_equivalent():
    import numpy as np
    from wenbo_engine.kernel import backend
    from wenbo_engine.kernel.cpu_nonlocal import apply_1q_pair
    from wenbo_engine.storage.block_store import DTYPE
    rng = np.random.default_rng(5)
    a0 = (rng.standard_normal(8) + 1j*rng.standard_normal(8)).astype(DTYPE)
    b0 = (rng.standard_normal(8) + 1j*rng.standard_normal(8)).astype(DTYPE)
    H = (np.array([[1, 1], [1, -1]], dtype=DTYPE) / np.sqrt(2))
    out = {}
    backends = ["numpy", "numba"] if backend.numba_available() else ["numpy"]
    try:
        for be in backends:
            backend.set_backend(be)
            a, b = a0.copy(), b0.copy()
            apply_1q_pair(a, b, H)
            out[be] = (a, b)
            assert a.dtype == DTYPE and b.dtype == DTYPE
    finally:
        backend.set_backend("numpy")
    if "numba" in out:
        assert np.allclose(out["numpy"][0], out["numba"][0], atol=1e-5)
        assert np.allclose(out["numpy"][1], out["numba"][1], atol=1e-5)
