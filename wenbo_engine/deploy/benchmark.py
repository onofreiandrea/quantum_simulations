#!/usr/bin/env python3
"""38-qubit distributed benchmark.

Runs a quest_random non-stabilizer circuit at 38 qubits (2 TB state vector)
across a Spark cluster. State lives as a Spark RDD with DISK_ONLY persistence;
Spark handles scheduling + data transport, custom kernels do computation.

Nonlocal gates are applied one at a time (bounded group size, no OOM).
Checkpoints every 2 nonlocal gates — if it crashes, restart and it recovers.

Usage:
    spark-submit --master spark://<ip>:7077 \
        --py-files wenbo_engine.zip benchmark.py

    spark-submit --master spark://<ip>:7077 \
        --py-files wenbo_engine.zip benchmark.py \
        --chunk-bits 24 --gate-ckpt-interval 2
"""
from __future__ import annotations

import argparse
import csv
import logging
import sys
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
    parser = argparse.ArgumentParser(description="38q distributed benchmark")
    parser.add_argument("--qubits", "-n", type=int, default=38)
    parser.add_argument("--gates", type=int, default=10,
                        help="Number of gates in quest_random circuit (default: 10)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-bits", type=int, default=24,
                        help="log2(chunk_size), default 24 = 16M amplitudes = 128MB chunks")
    parser.add_argument("--max-partitions", type=int, default=1024,
                        help="Max Spark partitions (default: 1024)")
    parser.add_argument("--gate-ckpt-interval", type=int, default=2,
                        help="Checkpoint every N nonlocal gates (default: 2). "
                             "Lower = less lost work on crash, more I/O.")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="Shared dir for WAL/checkpoints (NFS mount). "
                             "Default: /mnt/nvme/wenbo_data/benchmark_ckpt")
    parser.add_argument("--output", type=str, default="benchmark_results.csv")
    args = parser.parse_args()

    from pyspark import SparkContext, SparkConf
    from wenbo_engine.runner.rdd_runner import run
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

    ckpt_dir = args.checkpoint_dir or "/mnt/nvme/wenbo_data/benchmark_ckpt"
    Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

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
    log.info(f"  Gate ckpt:   every {args.gate_ckpt_interval} nonlocal gates")
    log.info(f"  Checkpoint:  {ckpt_dir}")
    log.info("=" * 60)

    t0 = time.time()
    result = run(
        circuit_dict=cd,
        sc=sc,
        chunk_size=chunk_size,
        max_partitions=args.max_partitions,
        checkpoint_dir=ckpt_dir,
        gate_ckpt_interval=args.gate_ckpt_interval,
        use_wal=True,
    )
    elapsed = time.time() - t0

    result["state_rdd"].unpersist()
    sc.stop()

    log.info("")
    log.info("=" * 60)
    log.info("  RESULTS")
    log.info("=" * 60)
    log.info(f"  Time:          {elapsed:.2f}s ({elapsed/60:.1f} min)")
    log.info(f"  Qubits:        {n}")
    log.info(f"  State vector:  {state_gb:.1f} GB")
    log.info(f"  Gates:         {len(cd['gates'])}")
    log.info(f"  Partitions:    {n_parts}")
    log.info(f"  Chunks:        {n_chunks}")
    log.info("=" * 60)

    with open(args.output, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["qubits", "gates", "chunks", "partitions",
                     "chunk_bits", "time_s"])
        w.writerow([n, len(cd["gates"]), n_chunks, n_parts,
                     args.chunk_bits, f"{elapsed:.2f}"])
    log.info(f"CSV written to {args.output}")


if __name__ == "__main__":
    main()
