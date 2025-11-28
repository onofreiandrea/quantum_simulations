"""
Utilities to load gate tensors into the SQL gate_matrix table.
"""
from __future__ import annotations

import sqlite3
from typing import Iterable

import numpy as np

from .gates import Gate


def insert_one_qubit_gate(db: sqlite3.Connection, gate: Gate):
    U = gate.tensor
    if U.shape != (2, 2):
        raise ValueError(f"Expected 2x2 tensor for {gate.gate_name}")
    for row in range(2):
        for col in range(2):
            val = U[row, col]
            db.execute(
                """
                INSERT OR REPLACE INTO gate_matrix(gate_name, arity, row, col, real, imag)
                VALUES (?, 1, ?, ?, ?, ?)
                """,
                (gate.gate_name, row, col, float(np.real(val)), float(np.imag(val))),
            )


def insert_two_qubit_gate(db: sqlite3.Connection, gate: Gate):
    T = gate.tensor.reshape(4, 4)
    for row in range(4):
        for col in range(4):
            val = T[row, col]
            db.execute(
                """
                INSERT OR REPLACE INTO gate_matrix(gate_name, arity, row, col, real, imag)
                VALUES (?, 2, ?, ?, ?, ?)
                """,
                (gate.gate_name, row, col, float(np.real(val)), float(np.imag(val))),
            )


def register_gate_types(db: sqlite3.Connection, gates: Iterable[Gate]):
    """Insert every unique gate tensor referenced by the circuit."""
    seen: set[tuple[str, int]] = set()
    for gate in gates:
        key = (gate.gate_name, 2 if gate.two_qubit_gate else 1)
        if key in seen:
            continue
        seen.add(key)
        if gate.two_qubit_gate:
            insert_two_qubit_gate(db, gate)
        else:
            insert_one_qubit_gate(db, gate)
    db.commit()

