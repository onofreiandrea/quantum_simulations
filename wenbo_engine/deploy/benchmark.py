#!/usr/bin/env python3
"""38-qubit distributed benchmark with crash recovery demo.

Runs a quest_random non-stabilizer circuit at 38 qubits (2 TB state vector)
across a Spark cluster. State lives as a Spark RDD with DISK_ONLY persistence;
Spark handles scheduling + data transport, custom kernels do computation.

Three phases:
  Phase 1: Run with crash injection — dies after N steps, WAL committed
  Phase 2: Re-run same circuit — WAL skips completed steps, finishes
  Phase 3: Clean run (no crash) for comparison

Usage:
    spark-submit --master spark://<ip>:7077 \
        --py-files wenbo_engine.zip benchmark.py

    spark-submit --master spark://<ip>:7077 \
        --py-files wenbo_engine.zip benchmark.py --gates 50 --crash-step 5
"""
from __future__ import annotations

import argparse
import csv
import logging
import os
import sys
import tempfile
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("benchmark")


def main():
    parser = argparse.ArgumentParser(description="38q distributed benchmark + crash recovery")
    parser.add_argument("--qubits", "-n", type=int, default=38)
    parser.add_argument("--gates", type=int, default=10,
                        help="Number of gates in quest_random circuit (default: 10)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-bits", type=int, default=20,
                        help="log2(chunk_size), default 20 = 1M amplitudes = 8MB chunks")
    parser.add_argument("--max-partitions", type=int, default=1024,
                        help="Max Spark partitions (default: 1024)")
    parser.add_argument("--checkpoint-interval", type=int, default=1,
                        help="Checkpoint every N steps (default: 1). "
                             "Higher = less I/O, but lose more work on crash")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="Shared dir for WAL/checkpoints (NFS mount or local). "
                             "Default: temp dir")
    parser.add_argument("--crash-step", type=int, default=None,
                        help="Step to crash after in phase 1 (default: halfway)")
    parser.add_argument("--output", type=str, default="benchmark_results.csv")
    args = parser.parse_args()

    from pyspark import SparkContext, SparkConf
    from wenbo_engine.runner.rdd_runner import run, collect_state_rdd
    from wenbo_engine.tests.fixtures.circuits import quest_random

    n = args.qubits
    chunk_size = 1 << args.chunk_bits
    state_gb = (1 << n) * 8 / (1024**3)
    n_chunks = (1 << n) // chunk_size
    n_parts = min(n_chunks, args.max_partitions)

    cd = quest_random(n, n_gates=args.gates, seed=args.seed)

    conf = SparkConf().setAppName(f"wenbo_{n}q_bench")
    sc = SparkContext(conf=conf)

    from wenbo_engine.circuit.io import validate_circuit_dict, levelize
    validated = validate_circuit_dict(cd)
    total_steps = len([lv for lv in levelize(validated) if lv])
    crash_step = args.crash_step or (total_steps // 2)
    crash_step = max(1, min(crash_step, total_steps - 1))

    log.info("=" * 60)
    log.info(f"  wenbo_engine {n}q benchmark (RDD-based)")
    log.info(f"  Qubits:      {n}")
    log.info(f"  State:       {state_gb:.1f} GB")
    log.info(f"  Gates:       {len(cd['gates'])} (quest_random, non-stabilizer)")
    log.info(f"  Chunk size:  {chunk_size} ({args.chunk_bits} bits, "
             f"{chunk_size * 8 / 1024**2:.1f} MB)")
    log.info(f"  Chunks:      {n_chunks}")
    log.info(f"  Partitions:  {n_parts}")
    log.info(f"  Executors:   {sc.defaultParallelism}")
    log.info(f"  Total steps: {total_steps}")
    log.info(f"  Crash after: step {crash_step}")
    log.info("=" * 60)

    results = []

    # Use provided checkpoint dir or create temp
    ckpt_base = args.checkpoint_dir
    use_temp = ckpt_base is None

    if use_temp:
        import tempfile as _tf
        _tmpdir = _tf.mkdtemp(prefix="wenbo_bench_")
        ckpt_base = _tmpdir
        log.info(f"Using temp checkpoint dir: {ckpt_base}")

    ckpt_crash = str(Path(ckpt_base) / "crash_run")
    ckpt_clean = str(Path(ckpt_base) / "clean_run")

    # ── Phase 1: crash ────────────────────────────────────────────
    log.info(f"PHASE 1: run with crash after step {crash_step}/{total_steps}")
    os.environ["WE_CRASH_AFTER_STEP"] = str(crash_step)

    t0 = time.time()
    try:
        run(circuit_dict=cd, sc=sc, chunk_size=chunk_size,
            max_partitions=args.max_partitions,
            checkpoint_dir=ckpt_crash, use_wal=True,
            checkpoint_interval=args.checkpoint_interval)
    except SystemExit:
        pass
    crash_time = time.time() - t0

    log.info(f"PHASE 1: crashed after {crash_time:.2f}s "
             f"(completed {crash_step}/{total_steps} steps, WAL committed)")
    results.append(("crash", crash_step, f"{crash_time:.2f}"))

    # ── Phase 2: recover ──────────────────────────────────────────
    del os.environ["WE_CRASH_AFTER_STEP"]
    log.info("PHASE 2: recovering from WAL — should skip completed steps")

    t0 = time.time()
    result = run(circuit_dict=cd, sc=sc, chunk_size=chunk_size,
                 max_partitions=args.max_partitions,
                 checkpoint_dir=ckpt_crash, use_wal=True,
                 checkpoint_interval=args.checkpoint_interval)
    recover_time = time.time() - t0
    result["state_rdd"].unpersist()

    log.info(f"PHASE 2: recovered and finished in {recover_time:.2f}s")
    results.append(("recover", len(cd["gates"]), f"{recover_time:.2f}"))

    # ── Phase 3: clean run ────────────────────────────────────────
    log.info("PHASE 3: clean run from scratch (no crash, no WAL)")

    t0 = time.time()
    clean_result = run(circuit_dict=cd, sc=sc, chunk_size=chunk_size,
                       max_partitions=args.max_partitions,
                       checkpoint_dir=ckpt_clean, use_wal=False)
    clean_time = time.time() - t0
    clean_result["state_rdd"].unpersist()

    log.info(f"PHASE 3: clean run in {clean_time:.2f}s")
    results.append(("clean", len(cd["gates"]), f"{clean_time:.2f}"))

    sc.stop()

    # ── Summary ───────────────────────────────────────────────────
    total_time = crash_time + recover_time
    overhead = (total_time / clean_time - 1) * 100 if clean_time > 0 else 0

    log.info("")
    log.info("=" * 60)
    log.info("  RESULTS")
    log.info("=" * 60)
    log.info(f"  Phase 1 (crash):    {crash_time:.2f}s  ({crash_step}/{total_steps} steps done)")
    log.info(f"  Phase 2 (recover):  {recover_time:.2f}s  (resumed, finished all)")
    log.info(f"  Phase 3 (clean):    {clean_time:.2f}s  (baseline)")
    log.info(f"  Total (crash+rec):  {total_time:.2f}s")
    log.info(f"  Recovery overhead:  {overhead:.1f}%")
    log.info(f"  Qubits:             {n}")
    log.info(f"  State vector:       {state_gb:.1f} GB")
    log.info(f"  Gates:              {len(cd['gates'])}")
    log.info(f"  Partitions:         {n_parts}")
    log.info(f"  Chunks:             {n_chunks}")
    log.info("=" * 60)

    with open(args.output, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["phase", "steps", "time_s"])
        w.writerows(results)
    log.info(f"CSV written to {args.output}")

    # Clean up temp dir
    if use_temp:
        import shutil
        shutil.rmtree(ckpt_base, ignore_errors=True)


if __name__ == "__main__":
    main()
