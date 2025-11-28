"""
Checkpoint and recovery utilities.
"""
from __future__ import annotations

import csv
from pathlib import Path
import sqlite3

from .gates import Gate
from .state_manager import apply_gate_atomic


def load_latest_checkpoint(db: sqlite3.Connection):
    row = db.execute(
        "SELECT version, path FROM checkpoint ORDER BY version DESC LIMIT 1"
    ).fetchone()
    if row is None:
        return 0, None
    return row[0], row[1]


def restore_state_from_checkpoint(db: sqlite3.Connection, path: str):
    with Path(path).open() as f:
        reader = csv.DictReader(f)
        rows = [(int(r["version"]), int(r["idx"]), float(r["real"]), float(r["imag"])) for r in reader]
    db.execute("DELETE FROM state;")
    db.executemany(
        "INSERT INTO state(version, idx, real, imag) VALUES (?, ?, ?, ?)",
        rows,
    )
    db.commit()


def recover(db: sqlite3.Connection, circuit_gates: list[Gate]):
    checkpoint_version, path = load_latest_checkpoint(db)
    if path:
        restore_state_from_checkpoint(db, path)
        start_version = checkpoint_version
    else:
        start_version = 0
    version = start_version
    for idx, gate in enumerate(circuit_gates):
        if idx < start_version:
            continue
        version = apply_gate_atomic(db, version, gate)
    return version

