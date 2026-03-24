"""RDD runner tests: correctness, crash recovery, non-local gates.

Tests run with local[2] Spark to simulate multi-partition execution.
All results compared against ref_dense oracle.
"""
import os
import tempfile
import numpy as np
import pytest
from pathlib import Path

try:
    from pyspark import SparkContext, SparkConf
    HAS_SPARK = True
except ImportError:
    HAS_SPARK = False

from wenbo_engine.kernel.ref_dense import simulate
from wenbo_engine.wal.wal import WAL
from wenbo_engine.tests.fixtures.circuits import (
    bell_2q, ghz, qft, quest_random, ry_theta, cr3_encoded,
)


@pytest.mark.skipif(not HAS_SPARK, reason="pyspark not installed")
class TestRDDRunner:
    @pytest.fixture(scope="class")
    def sc(self):
        conf = (
            SparkConf()
            .setMaster("local[2]")
            .setAppName("rdd_test")
            .set("spark.ui.enabled", "false")
            .set("spark.driver.host", "127.0.0.1")
        )
        try:
            ctx = SparkContext(conf=conf)
        except Exception as e:
            pytest.skip(f"SparkContext failed (no Java?): {e}")
        yield ctx
        ctx.stop()

    # ── local-only circuits ──────────────────────────────────────

    @pytest.mark.parametrize("circ_fn", [bell_2q, lambda: ghz(3)])
    def test_local_circuits(self, sc, circ_fn):
        from wenbo_engine.runner.rdd_runner import run, collect_state_rdd
        cd = circ_fn()
        ref = simulate(cd)
        n = cd["number_of_qubits"]
        result = run(cd, sc, chunk_size=1 << n, use_wal=False)
        got = collect_state_rdd(result)
        result["state_rdd"].unpersist()
        np.testing.assert_allclose(got, ref, atol=1e-6)

    # ── non-local circuits (tiny chunks force non-local gates) ───

    def test_bell_nonlocal(self, sc):
        from wenbo_engine.runner.rdd_runner import run, collect_state_rdd
        cd = bell_2q()
        ref = simulate(cd)
        result = run(cd, sc, chunk_size=2, use_wal=False)
        got = collect_state_rdd(result)
        result["state_rdd"].unpersist()
        np.testing.assert_allclose(got, ref, atol=1e-6)

    def test_qft3_nonlocal(self, sc):
        from wenbo_engine.runner.rdd_runner import run, collect_state_rdd
        cd = qft(3)
        ref = simulate(cd)
        result = run(cd, sc, chunk_size=2, use_wal=False)
        got = collect_state_rdd(result)
        result["state_rdd"].unpersist()
        np.testing.assert_allclose(got, ref, atol=1e-6)

    def test_qft5_nonlocal(self, sc):
        from wenbo_engine.runner.rdd_runner import run, collect_state_rdd
        cd = qft(5)
        ref = simulate(cd)
        result = run(cd, sc, chunk_size=4, use_wal=False)
        got = collect_state_rdd(result)
        result["state_rdd"].unpersist()
        np.testing.assert_allclose(got, ref, atol=1e-6)

    # ── non-stabilizer circuits ──────────────────────────────────

    def test_quest_random_8q(self, sc):
        from wenbo_engine.runner.rdd_runner import run, collect_state_rdd
        cd = quest_random(8, n_gates=30, seed=42)
        ref = simulate(cd)
        result = run(cd, sc, chunk_size=16, use_wal=False)
        got = collect_state_rdd(result)
        result["state_rdd"].unpersist()
        np.testing.assert_allclose(got, ref, atol=1e-6)

    def test_quest_random_10q(self, sc):
        from wenbo_engine.runner.rdd_runner import run, collect_state_rdd
        cd = quest_random(10, n_gates=50, seed=99)
        ref = simulate(cd)
        result = run(cd, sc, chunk_size=64, use_wal=False)
        got = collect_state_rdd(result)
        result["state_rdd"].unpersist()
        np.testing.assert_allclose(got, ref, atol=1e-6)

    def test_ry_param(self, sc):
        from wenbo_engine.runner.rdd_runner import run, collect_state_rdd
        cd = ry_theta()
        ref = simulate(cd)
        result = run(cd, sc, chunk_size=2, use_wal=False)
        got = collect_state_rdd(result)
        result["state_rdd"].unpersist()
        np.testing.assert_allclose(got, ref, atol=1e-6)

    def test_cr3_encoded(self, sc):
        from wenbo_engine.runner.rdd_runner import run, collect_state_rdd
        cd = cr3_encoded()
        ref = simulate(cd)
        result = run(cd, sc, chunk_size=2, use_wal=False)
        got = collect_state_rdd(result)
        result["state_rdd"].unpersist()
        np.testing.assert_allclose(got, ref, atol=1e-6)

    # ── WAL and crash recovery ───────────────────────────────────

    def test_wal_committed(self, sc):
        from wenbo_engine.runner.rdd_runner import run
        cd = bell_2q()
        with tempfile.TemporaryDirectory() as td:
            result = run(cd, sc, chunk_size=2, checkpoint_dir=td, use_wal=True)
            result["state_rdd"].unpersist()
            wal = WAL(Path(td) / "wal.json", circuit_dict=cd)
            assert wal.done_steps >= 1

    def test_crash_and_recover(self, sc):
        """Crash after step 1, recover, verify correctness."""
        from wenbo_engine.runner.rdd_runner import run, collect_state_rdd
        cd = quest_random(8, n_gates=20, seed=42)
        ref = simulate(cd)

        with tempfile.TemporaryDirectory() as td:
            # Phase 1: crash after step 1
            os.environ["WE_CRASH_AFTER_STEP"] = "1"
            try:
                run(cd, sc, chunk_size=16, checkpoint_dir=td, use_wal=True)
            except SystemExit:
                pass
            finally:
                del os.environ["WE_CRASH_AFTER_STEP"]

            # Verify WAL recorded step
            wal = WAL(Path(td) / "wal.json", circuit_dict=cd)
            assert wal.done_steps == 1

            # Phase 2: recover and finish
            result = run(cd, sc, chunk_size=16, checkpoint_dir=td, use_wal=True)
            got = collect_state_rdd(result)
            result["state_rdd"].unpersist()

        np.testing.assert_allclose(got, ref, atol=1e-6)

    def test_crash_halfway_recover(self, sc):
        """Crash halfway through, recover, verify correctness."""
        from wenbo_engine.runner.rdd_runner import run, collect_state_rdd
        from wenbo_engine.circuit.io import validate_circuit_dict, levelize

        cd = quest_random(8, n_gates=30, seed=77)
        ref = simulate(cd)

        validated = validate_circuit_dict(cd)
        total_steps = len([lv for lv in levelize(validated) if lv])
        crash_at = max(1, total_steps // 2)

        with tempfile.TemporaryDirectory() as td:
            os.environ["WE_CRASH_AFTER_STEP"] = str(crash_at)
            try:
                run(cd, sc, chunk_size=16, checkpoint_dir=td, use_wal=True)
            except SystemExit:
                pass
            finally:
                del os.environ["WE_CRASH_AFTER_STEP"]

            # Recover
            result = run(cd, sc, chunk_size=16, checkpoint_dir=td, use_wal=True)
            got = collect_state_rdd(result)
            result["state_rdd"].unpersist()

        np.testing.assert_allclose(got, ref, atol=1e-6)

    # ── edge cases ───────────────────────────────────────────────

    def test_single_chunk(self, sc):
        """All amplitudes in one chunk — no nonlocal possible."""
        from wenbo_engine.runner.rdd_runner import run, collect_state_rdd
        cd = qft(4)
        ref = simulate(cd)
        result = run(cd, sc, chunk_size=16, use_wal=False)
        got = collect_state_rdd(result)
        result["state_rdd"].unpersist()
        np.testing.assert_allclose(got, ref, atol=1e-6)
