"""
Write-ahead logging helpers.
"""
from __future__ import annotations

import json
import sqlite3

from .gates import Gate


def wal_log_pending(db: sqlite3.Connection, version: int, gate: Gate) -> int:
    qubits_json = json.dumps(gate.qubits)
    params_json = "{}"
    db.execute(
        """
        INSERT INTO wal(version, gate_name, qubits, params, status)
        VALUES (?, ?, ?, ?, 'PENDING')
        """,
        (version, gate.gate_name, qubits_json, params_json),
    )
    wal_id = db.execute("SELECT last_insert_rowid()").fetchone()[0]
    db.commit()
    return wal_id


def wal_mark_committed(db: sqlite3.Connection, wal_id: int):
    db.execute("UPDATE wal SET status='COMMITTED' WHERE wal_id=?", (wal_id,))
    db.commit()

