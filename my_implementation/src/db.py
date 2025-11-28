"""
SQLite helper for Phase 1 simulator.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path


def connect(db_path: str | Path) -> sqlite3.Connection:
    path = Path(db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.execute("PRAGMA foreign_keys = ON;")
    return con


def initialize_schema(con: sqlite3.Connection, schema_path: str | Path):
    schema_text = Path(schema_path).read_text()
    con.executescript(schema_text)

