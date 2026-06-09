#!/usr/bin/env python3
"""MPI distributed benchmark for wenbo_engine.

Runs quest_random circuits at 38-40 qubits across MPI ranks.
Each rank stores its state partition on local NVMe.

Usage:
    mpirun -np 4 --hostfile hosts.txt \
        python -m wenbo_engine.mpi.mpi_benchmark \
        --qubits 40 --gates 10 --chunk-bits 20 \
        --work-dir /mnt/nvme/wenbo_data/mpi_40q
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

from mpi4py import MPI

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("mpi_benchmark")


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_ranks = comm.Get_size()

    parser = argparse.ArgumentParser(description="MPI distributed benchmark")
    parser.add_argument("--qubits", "-n", type=int, default=38)
    parser.add_argument("--gates", type=int, default=10,
                        help="Number of gates in quest_random circuit")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk-bits", type=int, default=24,
                        help="log2(chunk_size), default 24 = 16M amps = 128 MB chunks")
    parser.add_argument("--work-dir", type=str,
                        default="/mnt/nvme/wenbo_data/mpi_bench")
    parser.add_argument("--output", type=str, default=None,
                        help="JSON output file (default: <work-dir>/results.json)")
    parser.add_argument("--no-reorder", action="store_true",
                        help="Disable qubit reordering optimisation")
    parser.add_argument("--recovery", choices=["none", "wal", "generation"],
                        default="wal",
                        help="Crash-recovery mode (default: wal)")
    args = parser.parse_args()

    from wenbo_engine.tests.fixtures.circuits import quest_random
    from wenbo_engine.circuit.io import validate_circuit_dict, levelize
    from wenbo_engine.circuit.reorder import reorder_qubits
    from wenbo_engine.mpi.mpi_runner import run, compute_norm

    n = args.qubits
    chunk_size = 1 << args.chunk_bits
    state_gb = (1 << n) * 8 / (1024 ** 3)
    n_chunks = (1 << n) // chunk_size
    n_chunks_per_rank = n_chunks // n_ranks

    cd = quest_random(n, n_gates=args.gates, seed=args.seed)

    if not args.no_reorder:
        cd, _perm = reorder_qubits(cd)

    if rank == 0:
        validated = validate_circuit_dict(cd)
        total_levels = len([lv for lv in levelize(validated) if lv])
        log.info("=" * 60)
        log.info(f"  wenbo_engine MPI benchmark")
        log.info(f"  Qubits:        {n}")
        log.info(f"  State:         {state_gb:.1f} GB")
        log.info(f"  Gates:         {args.gates} (quest_random, non-stabilizer)")
        log.info(f"  Reorder:       {'OFF' if args.no_reorder else 'ON'}")
        log.info(f"  Chunk size:    2^{args.chunk_bits} = {chunk_size} "
                 f"({chunk_size * 8 / 1024**2:.1f} MB)")
        log.info(f"  Total chunks:  {n_chunks}")
        log.info(f"  Chunks/rank:   {n_chunks_per_rank}")
        log.info(f"  MPI ranks:     {n_ranks}")
        log.info(f"  Levels:        {total_levels}")
        log.info(f"  Work dir:      {args.work_dir}")
        log.info(f"  Recovery:      {args.recovery}")
        log.info("=" * 60)

    comm.Barrier()
    t0 = time.time()

    result_path = run(
        circuit_dict=cd,
        work_dir=args.work_dir,
        chunk_size=chunk_size,
        comm=comm,
        recovery=args.recovery,
    )

    elapsed = time.time() - t0
    comm.Barrier()

    # Compute norm for correctness check
    t_norm = time.time()
    norm = compute_norm(args.work_dir, comm)
    dt_norm = time.time() - t_norm

    if rank == 0:
        norm_err = abs(norm - 1.0)
        log.info("")
        log.info("=" * 60)
        log.info("  RESULTS")
        log.info("=" * 60)
        log.info(f"  Time:          {elapsed:.2f}s ({elapsed / 60:.1f} min)")
        log.info(f"  Norm:          {norm:.10f} (error: {norm_err:.2e})")
        log.info(f"  Norm check:    {dt_norm:.1f}s")
        log.info(f"  Qubits:        {n}")
        log.info(f"  State:         {state_gb:.1f} GB")
        log.info(f"  MPI ranks:     {n_ranks}")
        log.info(f"  Chunks/rank:   {n_chunks_per_rank}")
        log.info(f"  PASS:          {'YES' if norm_err < 1e-4 else 'NO'}")
        log.info("=" * 60)

        output = args.output or str(Path(args.work_dir) / "results.json")
        results = {
            "qubits": n,
            "gates": args.gates,
            "seed": args.seed,
            "chunk_bits": args.chunk_bits,
            "n_ranks": n_ranks,
            "reorder": not args.no_reorder,
            "time_s": round(elapsed, 2),
            "norm": round(norm, 10),
            "norm_error": float(norm_err),
            "pass": norm_err < 1e-4,
        }
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(results, indent=2))
        log.info(f"Results written to {output}")


if __name__ == "__main__":
    main()
