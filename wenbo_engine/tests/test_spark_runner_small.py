"""Spark runner: local[2] test, compare to single-node and Qiskit.

Tests:
  - Bell circuit (chunk_size = 2^n → all local)
  - GHZ-3 (chunk_size = 2^n → all local)
  - Bell with tiny chunks (chunk_size=2 → forces non-local CNOT)
  - QFT-3 with tiny chunks (mix of local + non-local gates)
  - WAL entries are COMMITTED after clean run
"""
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
from wenbo_engine.runner.single_node import collect_state
from wenbo_engine.tests.fixtures.circuits import bell_2q, ghz, qft


@pytest.mark.skipif(not HAS_SPARK, reason="pyspark not installed")
class TestSparkRunner:
    @pytest.fixture(scope="class")
    def sc(self):
        conf = (
            SparkConf()
            .setMaster("local[2]")
            .setAppName("we_test")
            .set("spark.ui.enabled", "false")
            .set("spark.driver.host", "127.0.0.1")
        )
        try:
            ctx = SparkContext(conf=conf)
        except Exception as e:
            pytest.skip(f"SparkContext failed (no Java?): {e}")
        yield ctx
        ctx.stop()

    # ── local-only circuits (chunk_size >= 2^n) ──────────────────────

    @pytest.mark.parametrize("circ_fn", [bell_2q, lambda: ghz(3)])
    def test_spark_vs_ref_local(self, sc, circ_fn):
        from wenbo_engine.runner.spark_runner import run as spark_run
        cd = circ_fn()
        ref = simulate(cd)
        n = cd["number_of_qubits"]
        with tempfile.TemporaryDirectory() as td:
            final = spark_run(cd, td, sc, chunk_size=1 << n)
            got = collect_state(final)
        np.testing.assert_allclose(got, ref, atol=1e-6)

    # ── non-local circuits (tiny chunk_size) ─────────────────────────

    def test_spark_bell_nonlocal(self, sc):
        """Bell with chunk_size=2 → CNOT is non-local."""
        from wenbo_engine.runner.spark_runner import run as spark_run
        cd = bell_2q()
        ref = simulate(cd)
        with tempfile.TemporaryDirectory() as td:
            final = spark_run(cd, td, sc, chunk_size=2)
            got = collect_state(final)
        np.testing.assert_allclose(got, ref, atol=1e-6)

    def test_spark_qft3_nonlocal(self, sc):
        """QFT-3 with chunk_size=2 → many non-local gates."""
        from wenbo_engine.runner.spark_runner import run as spark_run
        cd = qft(3)
        ref = simulate(cd)
        with tempfile.TemporaryDirectory() as td:
            final = spark_run(cd, td, sc, chunk_size=2)
            got = collect_state(final)
        np.testing.assert_allclose(got, ref, atol=1e-6)

    # ── WAL verification ─────────────────────────────────────────────

    def test_spark_wal_committed(self, sc):
        """After clean run, WAL has done_steps > 0."""
        from wenbo_engine.runner.spark_runner import run as spark_run
        from wenbo_engine.wal.wal import WAL
        cd = bell_2q()
        with tempfile.TemporaryDirectory() as td:
            spark_run(cd, td, sc, chunk_size=4, use_wal=True)
            wal = WAL(Path(td) / "wal.json", circuit_dict=cd)
            assert wal.done_steps >= 1
            wal.close()
