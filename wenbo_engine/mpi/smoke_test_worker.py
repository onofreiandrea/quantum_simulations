#!/usr/bin/env python3
"""MPI smoke test worker — runs inside mpirun, called by smoke_test.py."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

import numpy as np
from mpi4py import MPI

from wenbo_engine.mpi.mpi_runner import run, collect_state, compute_norm
from wenbo_engine.tests.fixtures.circuits import quest_random
from wenbo_engine.kernel.ref_dense import simulate


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_ranks = comm.Get_size()

    parser = argparse.ArgumentParser()
    parser.add_argument("--work-dir", required=True)
    parser.add_argument("--mode", default="run")
    args = parser.parse_args()

    n = 10
    cd = quest_random(n, n_gates=10, seed=42)
    chunk_size = max(4, (1 << n) // (n_ranks * 4))

    run(cd, args.work_dir, chunk_size=chunk_size, use_wal=True, comm=comm)

    # Collect results on rank 0
    got = collect_state(args.work_dir, comm)
    norm = compute_norm(args.work_dir, comm)

    if rank == 0:
        ref = simulate(cd)
        max_error = float(np.max(np.abs(got - ref)))
        norm_error = float(abs(norm - 1.0))

        results = {
            "max_error": max_error,
            "norm_error": norm_error,
            "n_ranks": n_ranks,
        }
        Path(args.work_dir, "test_results.json").write_text(
            json.dumps(results, indent=2)
        )


if __name__ == "__main__":
    main()
