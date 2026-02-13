"""Crash recovery for double-buffer design.

With double-buffering, recovery is trivial: just run the circuit again.
The WAL stores (committed_buf, done_steps).  On restart the runner
wipes the destination buffer and redoes the current step from the
intact source buffer.

This module provides a `recover` function for API compatibility,
but it simply delegates to the runner.
"""
from __future__ import annotations

from pathlib import Path


def recover(
    circuit_dict: dict,
    work_dir: str | Path,
    chunk_size: int = 1 << 20,
) -> Path | None:
    """Recover from crash.  Returns path to final state buffer.

    With double-buffering, this is just 'run the circuit again'.
    The runner automatically resumes from the last committed step.
    Returns None if there's nothing to recover (no WAL file).
    """
    wal_path = Path(work_dir) / "wal.json"
    if not wal_path.exists():
        return None

    from wenbo_engine.runner.single_node import run
    return run(circuit_dict, work_dir, chunk_size=chunk_size, use_wal=True)
