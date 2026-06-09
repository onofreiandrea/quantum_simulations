"""Test the exact circuit we will run on the cluster.

quest_random(n, n_gates=10, seed=42) — same parameters as benchmark.py defaults.
Tests at multiple qubit counts, with and without crash recovery.
Also validates the 38q circuit builds and levelizes correctly.
"""
import os
import tempfile
import numpy as np
import pytest

try:
    from pyspark import SparkContext, SparkConf
    HAS_SPARK = True
except ImportError:
    HAS_SPARK = False

from wenbo_engine.kernel.ref_dense import simulate
from wenbo_engine.circuit.io import validate_circuit_dict, levelize
from wenbo_engine.tests.fixtures.circuits import quest_random


def test_38q_circuit_builds():
    """The 38q circuit must validate and levelize without errors."""
    cd = quest_random(38, n_gates=10, seed=42)
    validated = validate_circuit_dict(cd)
    levels = [lv for lv in levelize(validated) if lv]

    assert len(cd["gates"]) == 10
    assert validated["number_of_qubits"] == 38
    assert len(levels) > 0

    # Print structure for manual inspection
    print(f"\n38q circuit: {len(cd['gates'])} gates -> {len(levels)} steps")
    crash_step = len(levels) // 2
    print(f"Crash at step {crash_step}/{len(levels)}")
    for i, lv in enumerate(levels):
        gate_names = [g["gate"] for g in lv]
        print(f"  Step {i}: {len(lv)} gates — {gate_names}")


@pytest.mark.skipif(not HAS_SPARK, reason="pyspark not installed")
class TestExactBenchmarkCircuit:
    @pytest.fixture(scope="class")
    def sc(self):
        conf = (
            SparkConf()
            .setMaster("local[2]")
            .setAppName("exact_bench_test")
            .set("spark.ui.enabled", "false")
            .set("spark.driver.host", "127.0.0.1")
        )
        try:
            ctx = SparkContext(conf=conf)
        except Exception as e:
            pytest.skip(f"SparkContext failed: {e}")
        yield ctx
        ctx.stop()

    @pytest.mark.parametrize("n", [8, 10, 12, 14, 16])
    def test_correctness_at_scale(self, sc, n):
        """quest_random(n, 10 gates, seed=42) matches ref_dense at each qubit count."""
        from wenbo_engine.runner.distributed_runner import run, collect_state_distributed

        cd = quest_random(n, n_gates=10, seed=42)
        ref = simulate(cd)
        chunk_size = max(4, (1 << n) // 8)

        with tempfile.TemporaryDirectory() as td:
            result = run(cd, sc, local_dir=td, chunk_size=chunk_size,
                         n_partitions=2, use_wal=True)
            got = collect_state_distributed(sc, result)

        err = np.max(np.abs(got - ref))
        norm = np.sum(np.abs(got) ** 2)
        print(f"\n{n}q: max_err={err:.2e}, norm={norm:.12f}")
        assert abs(norm - 1.0) < 1e-6, f"norm is {norm}, should be ~1.0"
        np.testing.assert_allclose(got, ref, atol=1e-6)

    @pytest.mark.parametrize("n", [8, 10, 12, 14])
    def test_crash_recover_at_scale(self, sc, n):
        """Crash halfway, recover, result matches ref_dense."""
        from wenbo_engine.runner.distributed_runner import run, collect_state_distributed

        cd = quest_random(n, n_gates=10, seed=42)
        ref = simulate(cd)
        chunk_size = max(4, (1 << n) // 8)

        validated = validate_circuit_dict(cd)
        total_steps = len([lv for lv in levelize(validated) if lv])
        crash_at = max(1, total_steps // 2)

        with tempfile.TemporaryDirectory() as td:
            # Phase 1: crash
            os.environ["WE_CRASH_AFTER_STEP"] = str(crash_at)
            try:
                run(cd, sc, local_dir=td, chunk_size=chunk_size,
                    n_partitions=2, use_wal=True)
            except SystemExit:
                pass
            finally:
                del os.environ["WE_CRASH_AFTER_STEP"]

            # Phase 2: recover
            result = run(cd, sc, local_dir=td, chunk_size=chunk_size,
                         n_partitions=2, use_wal=True)
            got = collect_state_distributed(sc, result)

        err = np.max(np.abs(got - ref))
        print(f"\n{n}q: crash@{crash_at}/{total_steps}, recovered max_err={err:.2e}")
        np.testing.assert_allclose(got, ref, atol=1e-6)

    def test_crash_recover_matches_clean(self, sc):
        """Recovered result must be BIT-FOR-BIT identical to clean run."""
        from wenbo_engine.runner.distributed_runner import run, collect_state_distributed

        n = 10
        cd = quest_random(n, n_gates=10, seed=42)
        chunk_size = max(4, (1 << n) // 8)

        validated = validate_circuit_dict(cd)
        total_steps = len([lv for lv in levelize(validated) if lv])
        crash_at = max(1, total_steps // 2)

        # Crash + recover
        with tempfile.TemporaryDirectory() as td:
            os.environ["WE_CRASH_AFTER_STEP"] = str(crash_at)
            try:
                run(cd, sc, local_dir=td, chunk_size=chunk_size,
                    n_partitions=2, use_wal=True)
            except SystemExit:
                pass
            finally:
                del os.environ["WE_CRASH_AFTER_STEP"]

            result = run(cd, sc, local_dir=td, chunk_size=chunk_size,
                         n_partitions=2, use_wal=True)
            recovered = collect_state_distributed(sc, result)

        # Clean run
        with tempfile.TemporaryDirectory() as td:
            result2 = run(cd, sc, local_dir=td, chunk_size=chunk_size,
                          n_partitions=2, use_wal=False)
            clean = collect_state_distributed(sc, result2)

        diff = np.max(np.abs(recovered - clean))
        print(f"\nRecovered vs clean diff: {diff:.2e}")
        assert diff == 0.0, f"recovered and clean differ by {diff}"
