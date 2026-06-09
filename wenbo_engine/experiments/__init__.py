"""Reproducible experiment harness for the out-of-core simulator.

An *experiment* is a single, fully-described, fully-measured run.  Given a
config (YAML or JSON) it produces a self-contained directory::

    experiments/<run_id>/
      config.json          resolved config actually used
      circuit.json         the circuit dict that was simulated
      circuit.qasm         (when the circuit came from / can render to QASM)
      plan.json            compiled step plan (op classification per step)
      cost_model.json      machine calibration (NVMe / fsync / MPI / ...)
      stage_profile.csv    per-step timings + bytes + recovery mode
      io_profile.csv       per-event disk I/O
      mpi_profile.csv      per-event MPI exchanges (empty without MPI)
      recovery_events.json WAL resume / recovery observations
      final_summary.json   aggregated totals + throughput
      final_norm.txt       L2 norm of the final state vector
      git_commit.txt       repo commit + dirty flag at run time

Nothing here modifies the simulator's recovery logic; the harness reuses the
existing WAL for crash recovery and only *observes* the run.
"""
from __future__ import annotations

from wenbo_engine.experiments.config import ExperimentConfig, CircuitConfig, load_config

__all__ = ["ExperimentConfig", "CircuitConfig", "load_config"]
