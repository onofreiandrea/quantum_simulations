"""WAL + crash recovery tests (double-buffer design).

Tests:
  1. Normal run with WAL → verify committed state
  2. Simulated crash → recover → correct result
  3. WE_CRASH_AFTER_CHUNK env var → subprocess crash → run again → correct result
  4. Fencing lock prevents concurrent access
  5. WAL circuit hash mismatch
"""
import os
import subprocess
import sys
import tempfile
import numpy as np
import pytest
from pathlib import Path

from wenbo_engine.kernel.ref_dense import simulate
from wenbo_engine.wal.wal import WAL
from wenbo_engine.wal.recovery import recover
from wenbo_engine.wal.fencing import FencingLock
from wenbo_engine.runner.single_node import run, collect_state, _buf_dir, _wipe_buf


# ── helpers ──────────────────────────────────────────────────────────

def _single_gate_circuit():
    return {"number_of_qubits": 2, "gates": [{"qubits": [0], "gate": "X"}]}


def _bell():
    return {"number_of_qubits": 2, "gates": [
        {"qubits": [0], "gate": "H"},
        {"qubits": [0, 1], "gate": "CNOT"},
    ]}


# ── Test 1: normal run with WAL ─────────────────────────────────────

def test_runner_wal_commits():
    cd = _bell()
    cs = 1 << cd["number_of_qubits"]
    with tempfile.TemporaryDirectory() as td:
        run(cd, td, chunk_size=cs, use_wal=True)
        wal = WAL(Path(td) / "wal.json", circuit_dict=cd)
        assert wal.done_steps >= 1
        # committed buffer should have a manifest
        buf = _buf_dir(Path(td), wal.committed_buf)
        assert (buf / "manifest.json").exists()


def test_correctness_with_wal():
    cd = _bell()
    ref = simulate(cd)
    cs = 1 << cd["number_of_qubits"]
    with tempfile.TemporaryDirectory() as td:
        final = run(cd, td, chunk_size=cs, use_wal=True)
        got = collect_state(final)
    np.testing.assert_allclose(got, ref, atol=1e-6)


# ── Test 2: simulate crash + recover ────────────────────────────────

def _simulate_crash(cd, work_dir, chunk_size):
    """Write initial state + WAL with done_steps=0, then write partial dst.

    Simulates: the runner started step 0 but crashed before committing.
    """
    from wenbo_engine.storage.block_store import init_zero_state
    work = Path(work_dir)

    # Write initial state to state_a
    a_dir = _buf_dir(work, "a")
    init_zero_state(str(a_dir), cd["number_of_qubits"], chunk_size)

    # Write WAL with done_steps=0 (no steps completed)
    wal = WAL(work / "wal.json", circuit_dict=cd)
    wal.close()


def test_crash_and_recover_single_gate():
    cd = _single_gate_circuit()
    ref = simulate(cd)
    cs = 1 << cd["number_of_qubits"]
    with tempfile.TemporaryDirectory() as td:
        _simulate_crash(cd, td, cs)
        # WAL exists with done_steps=0 → recover should re-run from scratch
        final = recover(cd, td, chunk_size=cs)
        assert final is not None
        np.testing.assert_allclose(collect_state(final), ref, atol=1e-6)


def test_crash_and_recover_bell():
    cd = _bell()
    ref = simulate(cd)
    cs = 1 << cd["number_of_qubits"]
    with tempfile.TemporaryDirectory() as td:
        _simulate_crash(cd, td, cs)
        final = recover(cd, td, chunk_size=cs)
        assert final is not None
        np.testing.assert_allclose(collect_state(final), ref, atol=1e-6)


def test_partial_dst_cleaned_on_recover():
    """If dst has partial chunks from a crash, they are wiped on recover."""
    cd = _single_gate_circuit()
    ref = simulate(cd)
    cs = 1 << cd["number_of_qubits"]
    with tempfile.TemporaryDirectory() as td:
        _simulate_crash(cd, td, cs)
        # Manually write garbage to state_b (the dst)
        b_dir = _buf_dir(Path(td), "b")
        (b_dir / "chunks").mkdir(parents=True, exist_ok=True)
        (b_dir / "chunks" / "garbage.bin").write_bytes(b"\xff" * 100)

        final = recover(cd, td, chunk_size=cs)
        assert final is not None
        np.testing.assert_allclose(collect_state(final), ref, atol=1e-6)
        # Garbage should be gone (wipe_buf clears entire chunks dir)
        assert not (b_dir / "chunks" / "garbage.bin").exists()


# ── Test 3: subprocess crash via WE_CRASH_AFTER_CHUNK ────────────

_CRASH_SCRIPT = '''
import sys, json
sys.path.insert(0, "{repo_root}")
from wenbo_engine.runner.single_node import run
cd = json.loads('{cd_json}')
run(cd, "{work_dir}", chunk_size={cs}, use_wal=True)
'''


def test_crash_env_var_and_recover():
    cd = _single_gate_circuit()
    ref = simulate(cd)
    cs = 1 << cd["number_of_qubits"]
    repo_root = str(Path(__file__).resolve().parent.parent.parent)

    with tempfile.TemporaryDirectory() as td:
        import json
        cd_json = json.dumps(cd)
        script = _CRASH_SCRIPT.format(
            repo_root=repo_root, cd_json=cd_json, work_dir=td, cs=cs,
        )
        env = os.environ.copy()
        env["WE_CRASH_AFTER_CHUNK"] = "0"
        result = subprocess.run(
            [sys.executable, "-c", script],
            env=env, capture_output=True, timeout=10,
        )
        assert result.returncode != 0

        # WAL should exist with done_steps=0 (crash before commit)
        wal = WAL(Path(td) / "wal.json", circuit_dict=cd)
        assert wal.done_steps == 0
        wal.close()

        # Recover (= just run again)
        final = recover(cd, td, chunk_size=cs)
        assert final is not None
        np.testing.assert_allclose(collect_state(final), ref, atol=1e-6)


# ── Test 4: no crash → no recovery ──────────────────────────────────

def test_no_crash_no_recovery():
    cd = _single_gate_circuit()
    with tempfile.TemporaryDirectory() as td:
        assert recover(cd, td) is None


# ── Test 5: fencing lock ────────────────────────────────────────────

def test_fencing_lock():
    with tempfile.TemporaryDirectory() as td:
        lock = FencingLock(td)
        lock.acquire()
        assert (Path(td) / "run.lock").exists()
        lock.release()
        assert not (Path(td) / "run.lock").exists()


def test_fencing_lock_context_manager():
    with tempfile.TemporaryDirectory() as td:
        with FencingLock(td):
            assert (Path(td) / "run.lock").exists()
        assert not (Path(td) / "run.lock").exists()


# ── Test 6: WAL circuit hash mismatch ───────────────────────────────

def test_wal_circuit_hash_mismatch():
    cd1 = _single_gate_circuit()
    cd2 = _bell()
    with tempfile.TemporaryDirectory() as td:
        wal = WAL(Path(td) / "wal.json", circuit_dict=cd1)
        wal.close()
        with pytest.raises(ValueError, match="circuit hash mismatch"):
            WAL(Path(td) / "wal.json", circuit_dict=cd2)


# ── Test 7: double-buffer alternation ──────────────────────────────

def test_double_buffer_alternates():
    """Verify that buffers alternate: a→b→a→b..."""
    cd = _bell()  # 2 levels → 2 steps
    cs = 1 << cd["number_of_qubits"]
    with tempfile.TemporaryDirectory() as td:
        run(cd, td, chunk_size=cs, use_wal=True)
        wal = WAL(Path(td) / "wal.json", circuit_dict=cd)
        # Bell has H then CNOT = 2 levels = 2 steps
        # Start: a. After step 0: b. After step 1: a.
        assert wal.done_steps == 2
        assert wal.committed_buf == "a"  # a → b → a


# ── Test 8: resume from partial run (subprocess) ────────────────────

_RESUME_SCRIPT = '''
import sys, json
sys.path.insert(0, "{repo_root}")
from wenbo_engine.runner.single_node import run
cd = json.loads('{cd_json}')
run(cd, "{work_dir}", chunk_size={cs}, use_wal=True)
'''


def test_resume_from_partial():
    """Subprocess crash, then re-run without crash → same result as fresh run."""
    cd = {"number_of_qubits": 3, "gates": [
        {"qubits": [0], "gate": "H"},
        {"qubits": [1], "gate": "H"},
        {"qubits": [2], "gate": "H"},
    ]}
    ref = simulate(cd)
    cs = 1 << cd["number_of_qubits"]
    repo_root = str(Path(__file__).resolve().parent.parent.parent)

    with tempfile.TemporaryDirectory() as td:
        import json
        cd_json = json.dumps(cd)
        script = _RESUME_SCRIPT.format(
            repo_root=repo_root, cd_json=cd_json, work_dir=td, cs=cs,
        )

        # First run: crash in subprocess
        env = os.environ.copy()
        env["WE_CRASH_AFTER_CHUNK"] = "0"
        result = subprocess.run(
            [sys.executable, "-c", script],
            env=env, capture_output=True, timeout=10,
        )
        assert result.returncode != 0

        # WAL should show 0 completed steps
        wal = WAL(Path(td) / "wal.json", circuit_dict=cd)
        assert wal.done_steps == 0
        wal.close()

        # Second run: no crash, should complete
        final = run(cd, td, chunk_size=cs, use_wal=True)
        np.testing.assert_allclose(collect_state(final), ref, atol=1e-6)
