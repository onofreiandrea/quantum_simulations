#!/usr/bin/env python3
"""Run wenbo_engine on a Spark cluster.

Uses the RDD-based runner: state vector lives as a Spark RDD with
DISK_ONLY persistence. Spark handles scheduling + data transport,
custom kernels do the computation.

Usage:
    # GHZ circuit, 35 qubits (default)
    spark-submit --master spark://<master-ip>:7077 \
        --py-files wenbo_engine.zip run_distributed.py

    # QFT circuit, 30 qubits, custom chunk size
    spark-submit --master spark://<master-ip>:7077 \
        --py-files wenbo_engine.zip \
        run_distributed.py --circuit qft --qubits 30 --chunk-size 20

    # From a Qiskit QASM file
    spark-submit --master spark://<master-ip>:7077 \
        --py-files wenbo_engine.zip \
        run_distributed.py --qasm circuit.qasm --qubits 35
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import tempfile
import time
from pathlib import Path

# Ensure wenbo_engine is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from pyspark import SparkContext, SparkConf

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("wenbo_run")


def build_circuit(args) -> dict:
    """Build a circuit dict from CLI arguments."""
    n = args.qubits

    if args.qasm:
        from qiskit import QuantumCircuit
        from wenbo_engine.circuit.import_qiskit import qiskit_to_dict
        qc = QuantumCircuit.from_qasm_file(args.qasm)
        cd = qiskit_to_dict(qc)
        log.info(f"Loaded QASM circuit: {qc.num_qubits} qubits, {len(cd['gates'])} gates")
        return cd

    if args.circuit_json:
        with open(args.circuit_json) as f:
            cd = json.load(f)
        log.info(f"Loaded circuit from {args.circuit_json}")
        return cd

    from wenbo_engine.tests.fixtures.circuits import (
        ghz, qft, random_clifford_t, hardware_efficient_ansatz,
        supremacy_like, quest_random,
    )
    depth = args.depth
    if depth is None:
        depth = 1000 if args.circuit == "quest_random" else 10
    builders = {
        "quest_random": lambda: quest_random(n, n_gates=depth, seed=args.seed),
        "random_ct": lambda: random_clifford_t(n, depth=depth, seed=args.seed),
        "hea": lambda: hardware_efficient_ansatz(n, layers=depth, seed=args.seed),
        "supremacy": lambda: supremacy_like(n, cycles=depth, seed=args.seed),
        "ghz": lambda: ghz(n),
        "qft": lambda: qft(n),
    }
    cd = builders[args.circuit]()
    log.info(f"{args.circuit} circuit: {n} qubits, {len(cd['gates'])} gates")

    return cd


def main():
    parser = argparse.ArgumentParser(description="Run wenbo_engine distributed simulation")
    parser.add_argument("--qubits", "-n", type=int, default=35,
                        help="Number of qubits (default: 35)")
    parser.add_argument("--circuit",
                        choices=["quest_random", "random_ct", "hea", "supremacy", "qft", "ghz"],
                        default="quest_random",
                        help="Circuit type (default: quest_random — matches QuEST/AWS benchmark)")
    parser.add_argument("--circuit-json", type=str, default=None,
                        help="Path to circuit JSON file")
    parser.add_argument("--qasm", type=str, default=None,
                        help="Path to QASM file (requires qiskit)")
    parser.add_argument("--depth", type=int, default=None,
                        help="Circuit depth/layers/n_gates. Default: 1000 for quest_random, 10 for others")
    parser.add_argument("--seed", type=int, default=42,
                        help="RNG seed for random circuits")
    parser.add_argument("--chunk-size", type=int, default=20,
                        help="log2(chunk_size) — e.g. 20 = 1M amplitudes = 8MB chunks (default: 20)")
    parser.add_argument("--max-partitions", type=int, default=1024,
                        help="Max Spark partitions (default: 1024)")
    parser.add_argument("--checkpoint-dir", type=str, default=None,
                        help="Shared dir for WAL/checkpoints (default: temp dir)")
    parser.add_argument("--no-wal", action="store_true",
                        help="Disable write-ahead log")
    parser.add_argument("--recovery", choices=["none", "wal", "generation"],
                        default=None,
                        help="Crash-recovery mode. Overrides --no-wal. "
                             "'generation' requires the MPI runner "
                             "(python -m wenbo_engine.mpi.mpi_benchmark).")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print config and exit without running")
    args = parser.parse_args()

    # Resolve recovery mode (--recovery overrides --no-wal).
    if args.recovery == "generation":
        parser.error("--recovery=generation is only supported by the MPI "
                     "runner: python -m wenbo_engine.mpi.mpi_benchmark "
                     "--recovery generation")
    if args.recovery is not None:
        args.no_wal = (args.recovery == "none")

    chunk_size = 1 << args.chunk_size
    n = args.qubits
    state_size_gb = (1 << n) * 8 / (1024**3)
    n_chunks = (1 << n) // chunk_size
    n_parts = min(n_chunks, args.max_partitions)

    log.info("=" * 60)
    log.info("wenbo_engine distributed simulation (RDD-based)")
    log.info("=" * 60)
    log.info(f"  Qubits:        {n}")
    log.info(f"  State size:    {state_size_gb:.1f} GB")
    log.info(f"  Chunk size:    {chunk_size} ({args.chunk_size} bits, {chunk_size * 8 / 1024**2:.1f} MB)")
    log.info(f"  Num chunks:    {n_chunks}")
    log.info(f"  Partitions:    {n_parts}")
    log.info(f"  WAL:           {'off' if args.no_wal else 'on'}")
    log.info("=" * 60)

    if args.dry_run:
        log.info("Dry run — exiting.")
        return

    cd = build_circuit(args)

    # Create Spark context
    # Note: use --py-files wenbo_engine.zip with spark-submit
    conf = SparkConf()
    conf.set("spark.app.name", f"wenbo_{n}q_{args.circuit}")
    sc = SparkContext(conf=conf)

    log.info(f"Spark context created: {sc.master}")
    log.info(f"  Executors: {sc.defaultParallelism}")

    from wenbo_engine.runner.rdd_runner import run

    ckpt_dir = args.checkpoint_dir
    if ckpt_dir is None:
        import tempfile as _tf
        ckpt_dir = _tf.mkdtemp(prefix="wenbo_run_")
        log.info(f"Using temp checkpoint dir: {ckpt_dir}")

    log.info("Starting simulation...")
    t0 = time.time()

    result = run(
        circuit_dict=cd,
        sc=sc,
        chunk_size=chunk_size,
        max_partitions=args.max_partitions,
        checkpoint_dir=ckpt_dir,
        use_wal=not args.no_wal,
    )

    elapsed = time.time() - t0
    result["state_rdd"].unpersist()

    sc.stop()

    log.info("=" * 60)
    log.info("DONE")
    log.info(f"  Time:        {elapsed:.1f}s")
    log.info(f"  Chunks:      {result['n_chunks']}")
    log.info(f"  Partitions:  {result['n_partitions']}")
    log.info("=" * 60)

    # Clean up temp dir
    if args.checkpoint_dir is None:
        import shutil
        shutil.rmtree(ckpt_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
