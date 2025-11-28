import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from my_implementation.src import db
from my_implementation.src.circuits import generate_ghz_circuit, generate_qft_circuit
from my_implementation.src.frontend import circuit_dict_to_gates
from my_implementation.src.gate_loader import register_gate_types
from my_implementation.src.simulator import run_circuit
from my_implementation.src.state_manager import fetch_state, initialize_state, apply_gate_atomic
from my_implementation.src.recovery import recover


SCHEMA = "/Users/andreaonofrei/Desktop/Quantum Thesis/my_try/my_implementation/sql/schema.sql"


def connect_db():
    conn = db.connect(":memory:")
    db.initialize_schema(conn, SCHEMA)
    return conn


def numpy_simulate(circuit_dict):
    n_qubits, gates = circuit_dict_to_gates(circuit_dict)
    state = np.zeros(2**n_qubits, dtype=complex)
    state[0] = 1.0
    for gate in gates:
        state = apply_gate_numpy(state, gate, n_qubits)
    return state


def apply_gate_numpy(state, gate, n_qubits):
    psi = state.reshape([2] * n_qubits)
    if gate.two_qubit_gate:
        axes = gate.qubits
        tensor = gate.tensor
        contracted = np.tensordot(tensor, psi, axes=([2, 3], axes))
        current_order = list(axes) + [i for i in range(n_qubits) if i not in axes]
        perm = [current_order.index(i) for i in range(n_qubits)]
        psi = np.transpose(contracted, perm)
    else:
        (axis,) = gate.qubits
        tensor = gate.tensor
        contracted = np.tensordot(tensor, psi, axes=([1], [axis]))
        current_order = [axis] + [i for i in range(n_qubits) if i != axis]
        perm = [current_order.index(i) for i in range(n_qubits)]
        psi = np.transpose(contracted, perm)
    return psi.reshape(-1)


def test_ghz_3_qubits():
    conn = connect_db()
    version = run_circuit(conn, generate_ghz_circuit(3))
    rows = fetch_state(conn, version)
    amps = {idx: complex(real, imag) for idx, real, imag in rows if abs(real) > 1e-9 or abs(imag) > 1e-9}
    assert len(amps) == 2
    assert set(amps.keys()) == {0, 7}
    expected = 1 / math.sqrt(2)
    for amp in amps.values():
        assert math.isclose(abs(amp), expected, rel_tol=1e-6)


def test_qft_small_matches_numpy():
    circuit = generate_qft_circuit(3)
    conn = connect_db()
    n_qubits, gates = circuit_dict_to_gates(circuit)
    initialize_state(conn, n_qubits)
    register_gate_types(conn, gates)
    version = 0
    for gate in gates:
        version = apply_gate_atomic(conn, version, gate)
    db_state = np.zeros(2**n_qubits, dtype=complex)
    for idx, real, imag in fetch_state(conn, version):
        db_state[idx] = real + 1j * imag
    np_state = numpy_simulate(circuit)
    assert np.allclose(db_state, np_state, atol=1e-6)


def test_wal_and_checkpoint(tmp_path):
    circuit = generate_ghz_circuit(3)
    conn = connect_db()
    checkpoint_dir = tmp_path / "checkpoints"
    version = run_circuit(conn, circuit, checkpoint_dir=str(checkpoint_dir))

    wal_rows = conn.execute(
        "SELECT wal_id, version, gate_name, status FROM wal ORDER BY wal_id"
    ).fetchall()
    assert len(wal_rows) == len(circuit["gates"])
    assert all(row[3] == "COMMITTED" for row in wal_rows)

    checkpoints = conn.execute("SELECT version, path FROM checkpoint").fetchall()
    assert checkpoints == [(version, str(checkpoint_dir / f"state_v{version}.csv"))]
    assert (checkpoint_dir / f"state_v{version}.csv").exists()

    restored_version = recover(conn, circuit_dict_to_gates(circuit)[1])
    assert restored_version == version

