"""
State initialization and gate application helpers.
"""
from __future__ import annotations

import sqlite3

from .gates import Gate
from .gate_translator import translate_gate
from .wal import wal_log_pending, wal_mark_committed


def initialize_state(db: sqlite3.Connection, n_qubits: int):
    """Initialize |0...0> state; n_qubits included for future use."""
    db.execute("DELETE FROM state;")
    db.execute("INSERT INTO state(version, idx, real, imag) VALUES (0, 0, 1.0, 0.0);")
    db.commit()


def apply_gate_atomic(db: sqlite3.Connection, version: int, gate: Gate) -> int:
    """
    Apply a gate atomically with WAL logging and return the new version.
    """
    wal_id = wal_log_pending(db, version, gate)
    sql = translate_gate(gate, version)
    cur = db.cursor()
    try:
        cur.execute("BEGIN;")
        cur.execute("DELETE FROM state WHERE version = ?", (version + 1,))
        cur.executescript(sql)
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        cur.close()
    wal_mark_committed(db, wal_id)
    return version + 1


def fetch_state(db: sqlite3.Connection, version: int):
    rows = db.execute("SELECT idx, real, imag FROM state WHERE version = ? ORDER BY idx", (version,)).fetchall()
    return rows

