"""Tests for gate fusion and level batching."""
from __future__ import annotations

import math
import tempfile

import numpy as np
import pytest

from wenbo_engine.circuit.io import validate_circuit_dict, levelize
from wenbo_engine.circuit.fusion import fuse_1q_ops, batch_levels, fusion_stats
from wenbo_engine.kernel import gates as gmod
from wenbo_engine.kernel.ref_dense import simulate
from wenbo_engine.runner.single_node import run as sn_run, collect_state
from wenbo_engine.runner.pipeline import run as pl_run
from wenbo_engine.tests.fixtures.circuits import bell_2q, ghz, qft, x_on_q0_3q


# ── Unit tests: fuse_1q_ops ──────────────────────────────────────────

def test_fuse_consecutive_1q():
    """Two consecutive 1Q gates on q0 should be fused into one."""
    H = gmod.gate_matrix("H", {})
    T = gmod.gate_matrix("T", {})
    ops = [([0], H), ([0], T)]
    fused = fuse_1q_ops(ops)
    assert len(fused) == 1
    assert fused[0][0] == [0]
    np.testing.assert_allclose(fused[0][1], T @ H, atol=1e-14)


def test_fuse_interrupted_by_2q():
    """1Q-2Q-1Q on same qubit: no fusion across the 2Q gate."""
    H = gmod.gate_matrix("H", {})
    CX = gmod.gate_matrix("CNOT", {})
    T = gmod.gate_matrix("T", {})
    ops = [([0], H), ([0, 1], CX), ([0], T)]
    fused = fuse_1q_ops(ops)
    # Should be: H(q0), CNOT(q0,q1), T(q0)
    assert len(fused) == 3
    np.testing.assert_allclose(fused[0][1], H, atol=1e-14)
    np.testing.assert_allclose(fused[2][1], T, atol=1e-14)


def test_fuse_different_qubits():
    """1Q on q0 then 1Q on q1: both kept (different qubits)."""
    H = gmod.gate_matrix("H", {})
    X = gmod.gate_matrix("X", {})
    ops = [([0], H), ([1], X)]
    fused = fuse_1q_ops(ops)
    assert len(fused) == 2


def test_fuse_three_1q_same_qubit():
    """H, T, S on q0 → single fused gate."""
    H = gmod.gate_matrix("H", {})
    T = gmod.gate_matrix("T", {})
    S = gmod.gate_matrix("S", {})
    ops = [([0], H), ([0], T), ([0], S)]
    fused = fuse_1q_ops(ops)
    assert len(fused) == 1
    np.testing.assert_allclose(fused[0][1], S @ T @ H, atol=1e-14)


def test_fuse_empty():
    assert fuse_1q_ops([]) == []


# ── Unit tests: batch_levels ─────────────────────────────────────────

def test_batch_all_local():
    """QFT-4 with chunk_size >= 2^4: all gates local, all levels batch."""
    cd = validate_circuit_dict(qft(4))
    levels = levelize(cd)
    k = 4  # log2(chunk_size) >= n_qubits
    passes = batch_levels(levels, k)
    # Should collapse into 1 fused pass (all local)
    assert len(passes) == 1
    assert passes[0]["nonlocal_ops"] == []
    stats = fusion_stats(levels, k)
    assert stats["fused_passes"] == 1


def test_batch_with_nonlocal():
    """GHZ-4 with chunk_size=4 (k=2): CNOT on q2,q3 are non-local."""
    cd = validate_circuit_dict(ghz(4))
    levels = levelize(cd)
    k = 2  # only q0, q1 are local
    passes = batch_levels(levels, k)
    # Should have more than 1 pass (some non-local)
    assert len(passes) >= 1
    has_nonlocal = any(p["nonlocal_ops"] for p in passes)
    assert has_nonlocal


# ── Integration tests: fusion produces same result as non-fusion ─────

def _compare_fusion(circ_fn, chunk_size=0):
    cd = circ_fn()
    n = cd["number_of_qubits"]
    N = 1 << n
    if chunk_size == 0:
        chunk_size = N

    ref = simulate(cd)

    with tempfile.TemporaryDirectory() as td1:
        v1 = sn_run(cd, td1, chunk_size=chunk_size, use_wal=False, use_fusion=False)
        sv1 = collect_state(v1)

    with tempfile.TemporaryDirectory() as td2:
        v2 = sn_run(cd, td2, chunk_size=chunk_size, use_wal=False, use_fusion=True)
        sv2 = collect_state(v2)

    np.testing.assert_allclose(sv1, ref, atol=1e-6,
                               err_msg="non-fused vs ref mismatch")
    np.testing.assert_allclose(sv2, ref, atol=1e-6,
                               err_msg="fused vs ref mismatch")


def test_fusion_bell():
    _compare_fusion(bell_2q)


def test_fusion_ghz6():
    _compare_fusion(lambda: ghz(6))


def test_fusion_qft4():
    _compare_fusion(lambda: qft(4))


def test_fusion_qft6():
    _compare_fusion(lambda: qft(6))


def test_fusion_x_on_q0():
    _compare_fusion(x_on_q0_3q)


def test_fusion_out_of_core():
    """Test fusion with chunk_size < state_size (multi-chunk)."""
    cd = qft(4)
    n = cd["number_of_qubits"]
    chunk_size = 4  # 2^2, so k=2, some gates are non-local
    _compare_fusion(lambda: qft(4), chunk_size=chunk_size)


def test_fusion_pipeline():
    """Test fusion in the pipeline runner."""
    cd = qft(4)
    ref = simulate(cd)

    with tempfile.TemporaryDirectory() as td:
        v = pl_run(cd, td, chunk_size=(1 << 4), use_wal=False, use_fusion=True)
        # collect state from pipeline output
        from wenbo_engine.runner.single_node import collect_state
        sv = collect_state(v)

    np.testing.assert_allclose(sv, ref, atol=1e-6)
