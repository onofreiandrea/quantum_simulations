#!/usr/bin/env python3
"""Test RDD runner on a multi-node Spark cluster with shared volume.

Same logic as smoke_test.py but designed for Docker compose cluster.
"""
from __future__ import annotations
import os, sys, tempfile
from pathlib import Path

sys.path.insert(0, "/code")

import numpy as np
from pyspark import SparkContext, SparkConf

from wenbo_engine.runner.rdd_runner import run, collect_state_rdd
from wenbo_engine.tests.fixtures.circuits import quest_random
from wenbo_engine.kernel.ref_dense import simulate
from wenbo_engine.circuit.io import validate_circuit_dict, levelize


def main():
    n = 10
    cd = quest_random(n, n_gates=10, seed=42)
    ref = simulate(cd)

    conf = SparkConf().setAppName("docker_cluster_test")
    sc = SparkContext(conf=conf)
    print(f"Spark: {sc.master}, parallelism={sc.defaultParallelism}")

    chunk_size = max(4, (1 << n) // 8)
    validated = validate_circuit_dict(cd)
    total_steps = len([lv for lv in levelize(validated) if lv])
    crash_at = max(1, total_steps // 2)

    errors = []

    # Test 1: clean run
    print(f"[1/3] Clean run: {n}q, 10 gates, {total_steps} steps")
    with tempfile.TemporaryDirectory() as td:
        result = run(cd, sc, chunk_size=chunk_size,
                     checkpoint_dir=td, use_wal=True)
        got = collect_state_rdd(result)
        result["state_rdd"].unpersist()
    err = np.max(np.abs(got - ref))
    if err > 1e-6:
        errors.append(f"Clean run: error {err:.2e} > 1e-6")
    print(f"  max error vs ref: {err:.2e} — {'OK' if err < 1e-6 else 'FAIL'}")

    # Test 2: crash + recover
    print(f"[2/3] Crash at step {crash_at}/{total_steps}, then recover")
    with tempfile.TemporaryDirectory() as td:
        os.environ["WE_CRASH_AFTER_STEP"] = str(crash_at)
        try:
            run(cd, sc, chunk_size=chunk_size,
                checkpoint_dir=td, use_wal=True)
        except SystemExit:
            pass
        del os.environ["WE_CRASH_AFTER_STEP"]

        result = run(cd, sc, chunk_size=chunk_size,
                     checkpoint_dir=td, use_wal=True)
        recovered = collect_state_rdd(result)
        result["state_rdd"].unpersist()
    err2 = np.max(np.abs(recovered - ref))
    if err2 > 1e-6:
        errors.append(f"Recovery: error {err2:.2e} > 1e-6")
    print(f"  max error vs ref: {err2:.2e} — {'OK' if err2 < 1e-6 else 'FAIL'}")

    # Test 3: recovered matches clean
    diff = np.max(np.abs(got - recovered))
    if diff > 1e-12:
        errors.append(f"Recovered vs clean: diff {diff:.2e}")
    print(f"[3/3] Recovered vs clean diff: {diff:.2e} — {'OK' if diff < 1e-12 else 'FAIL'}")

    sc.stop()

    print()
    if errors:
        print("DOCKER CLUSTER TEST FAILED:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print("DOCKER CLUSTER TEST PASSED")


if __name__ == "__main__":
    main()
