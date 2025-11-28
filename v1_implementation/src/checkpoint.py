"""
Checkpointing helpers.
"""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path
import sqlite3


def create_checkpoint(db: sqlite3.Connection, version: int, path: str):
    rows = db.execute(
        "SELECT version, idx, real, imag FROM state WHERE version = ? ORDER BY idx",
        (version,),
    ).fetchall()
    checkpoint_path = Path(path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    with checkpoint_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["version", "idx", "real", "imag"])
        writer.writerows(rows)
    checksum = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    db.execute(
        """
        INSERT OR REPLACE INTO checkpoint(version, path, checksum, timestamp)
        VALUES (?, ?, ?, CURRENT_TIMESTAMP)
        """,
        (version, str(checkpoint_path), checksum),
    )
    db.commit()

