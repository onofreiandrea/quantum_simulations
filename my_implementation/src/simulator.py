"""
High-level circuit execution helpers.
"""
from __future__ import annotations

from pathlib import Path
import sqlite3

from .checkpoint import create_checkpoint
from .circuits import (
    generate_ghz_circuit,
    generate_qft_circuit,
)
from .frontend import circuit_dict_to_gates
from .gate_loader import register_gate_types
from .state_manager import apply_gate_atomic, initialize_state


def run_circuit(db: sqlite3.Connection, circuit_dict, checkpoint_dir: str | None = None):
    n_qubits, gates = circuit_dict_to_gates(circuit_dict)
    initialize_state(db, n_qubits)
    register_gate_types(db, gates)
    version = 0
    for gate in gates:
        version = apply_gate_atomic(db, version, gate)
    checkpoint_dir = checkpoint_dir or "checkpoints"
    checkpoint_path = Path(checkpoint_dir) / f"state_v{version}.csv"
    create_checkpoint(db, version, checkpoint_path)
    return version


def run_demo(db: sqlite3.Connection):
    circuits = {
        "ghz3": generate_ghz_circuit(3),
        "qft3": generate_qft_circuit(3),
    }
    versions = {}
    for name, circ in circuits.items():
        versions[name] = run_circuit(db, circ)
    return versions

