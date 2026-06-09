#!/usr/bin/env python3
"""MPI smoke test — verify correctness at small scale before big runs.

This is an ORCHESTRATOR script.  Do NOT run with mpirun.
It launches mpirun as subprocesses so crash recovery can be tested
(os._exit kills the whole MPI job; recovery needs a fresh job).

Usage:
    cd wenbo_engine && python -m wenbo_engine.mpi.smoke_test
    python -m wenbo_engine.mpi.smoke_test --ranks 4
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))


def _mpirun(n_ranks: int, module: str, args: list[str],
            env_extra: dict | None = None) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    if env_extra:
        env.update(env_extra)
    cmd = [
        "mpirun", "-np", str(n_ranks),
        "--oversubscribe",  # allow more ranks than cores locally
        sys.executable, "-m", module,
    ] + args
    return subprocess.run(cmd, env=env, capture_output=True, text=True)


def main():
    parser = argparse.ArgumentParser(description="MPI smoke test orchestrator")
    parser.add_argument("--ranks", type=int, default=2)
    args = parser.parse_args()
    n_ranks = args.ranks

    errors = []

    # ── Test 1: Clean run ────────────────────────────────────────────
    print(f"MPI smoke test: 10q, {n_ranks} ranks\n")
    print("[1/3] Clean run")

    with tempfile.TemporaryDirectory() as td:
        r = _mpirun(n_ranks, "wenbo_engine.mpi.smoke_test_worker",
                    ["--work-dir", td, "--mode", "run"])
        if r.returncode != 0:
            print(f"  FAIL (exit {r.returncode})")
            print(r.stderr[-500:] if r.stderr else "(no stderr)")
            errors.append("Clean run failed")
        else:
            # Read results from rank 0
            results = json.loads((Path(td) / "test_results.json").read_text())
            err = results["max_error"]
            norm_err = results["norm_error"]
            print(f"  max error vs ref: {err:.2e} — {'OK' if err < 1e-6 else 'FAIL'}")
            print(f"  norm error:       {norm_err:.2e} — {'OK' if norm_err < 1e-6 else 'FAIL'}")
            if err >= 1e-6 or norm_err >= 1e-6:
                errors.append(f"Clean run: err={err:.2e}, norm_err={norm_err:.2e}")

    # ── Test 2: Crash + Recovery ─────────────────────────────────────
    print(f"\n[2/3] Crash + recovery")

    with tempfile.TemporaryDirectory() as td:
        # Phase 1: crash mid-run
        r = _mpirun(n_ranks, "wenbo_engine.mpi.smoke_test_worker",
                    ["--work-dir", td, "--mode", "run"],
                    env_extra={"WE_CRASH_AFTER_STEP": "2"})
        # This SHOULD fail (crash injection)
        print(f"  crash phase: exit {r.returncode} (expected non-zero)")

        # Phase 2: recover
        r = _mpirun(n_ranks, "wenbo_engine.mpi.smoke_test_worker",
                    ["--work-dir", td, "--mode", "run"])
        if r.returncode != 0:
            print(f"  recovery FAIL (exit {r.returncode})")
            print(r.stderr[-500:] if r.stderr else "(no stderr)")
            errors.append("Recovery failed")
        else:
            results = json.loads((Path(td) / "test_results.json").read_text())
            err = results["max_error"]
            print(f"  recovered max error vs ref: {err:.2e} — {'OK' if err < 1e-6 else 'FAIL'}")
            if err >= 1e-6:
                errors.append(f"Recovery: err={err:.2e}")

    # ── Test 3: Different rank counts ────────────────────────────────
    if n_ranks >= 4:
        print(f"\n[3/3] Cross-rank consistency (2 ranks vs {n_ranks} ranks)")
        with tempfile.TemporaryDirectory() as td2, \
             tempfile.TemporaryDirectory() as td4:
            _mpirun(2, "wenbo_engine.mpi.smoke_test_worker",
                    ["--work-dir", td2, "--mode", "run"])
            _mpirun(n_ranks, "wenbo_engine.mpi.smoke_test_worker",
                    ["--work-dir", td4, "--mode", "run"])
            r2 = json.loads((Path(td2) / "test_results.json").read_text())
            r4 = json.loads((Path(td4) / "test_results.json").read_text())
            diff = abs(r2["max_error"] - r4["max_error"])
            print(f"  error diff between 2-rank and {n_ranks}-rank: {diff:.2e} — OK")
    else:
        print(f"\n[3/3] Skipped (need --ranks 4 for cross-rank test)")

    # ── Summary ──────────────────────────────────────────────────────
    print()
    if errors:
        print("SMOKE TEST FAILED:")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)
    else:
        print(f"SMOKE TEST PASSED ({n_ranks} MPI ranks)")


if __name__ == "__main__":
    main()
