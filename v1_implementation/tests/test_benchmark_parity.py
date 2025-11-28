import copy
import os
import sqlite3
import sys
from pathlib import Path

import numpy as np
import opt_einsum as oe
import pytest

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "Quantum") not in sys.path:
    sys.path.insert(0, str(ROOT / "Quantum"))

import InfiniQuantumSim.TLtensor as tlt  # noqa: E402

from v1_implementation.src import db  # noqa: E402
from v1_implementation.src.frontend import circuit_dict_to_gates  # noqa: E402
from v1_implementation.src.simulator import run_circuit  # noqa: E402
from v1_implementation.src.state_manager import fetch_state  # noqa: E402

SCHEMA = str(ROOT / "v1_implementation" / "sql" / "schema.sql")
MAX_COMPARE_QUBITS = int(os.environ.get("BENCHMARK_PARITY_MAX_QUBITS", "16"))

BENCHMARK_CIRCUITS = [
    ("GHZ", tlt.generate_ghz_circuit, range(5, 35)),
    ("W", tlt.generate_w_circuit, range(5, 35)),
    ("QFT", tlt.generate_qft_circuit, range(5, 25)),
    ("QPE", tlt.generate_qpe_circuit, range(5, 19)),
]

PARAMS = [
    pytest.param(name, generator, n, id=f"{name}-{n}")
    for name, generator, rng in BENCHMARK_CIRCUITS
    for n in rng
]


@pytest.fixture(scope="module", autouse=True)
def disable_random_greedy_parallel():
    """Ensure opt_einsum stays single-threaded for TLtensor's NumPy backend."""

    original_cls = oe.RandomGreedy

    class SerialRandomGreedy(original_cls):
        def __init__(self, *args, **kwargs):
            kwargs["parallel"] = False
            super().__init__(*args, **kwargs)

    oe.RandomGreedy = SerialRandomGreedy
    yield
    oe.RandomGreedy = original_cls


def run_my_impl(circuit_dict):
    con = sqlite3.connect(":memory:")
    db.initialize_schema(con, SCHEMA)
    version = run_circuit(con, circuit_dict, checkpoint_dir="checkpoints/tests")
    state = np.zeros(2 ** circuit_dict["number_of_qubits"], dtype=complex)
    for idx, real, imag in fetch_state(con, version):
        state[idx] = real + 1j * imag
    return state


def run_reference_impl(circuit_dict):
    qc = tlt.QuantumCircuit(circuit_dict=circuit_dict)
    state = qc.run(contr_method="np")
    return state.reshape(-1)


def prepare_circuit(generator, n_qubits):
    circuit = copy.deepcopy(generator(n_qubits))
    if generator is tlt.generate_qpe_circuit:
        target = circuit["number_of_qubits"] - 1
        cu_idx = 0
        for gate in circuit["gates"]:
            if gate["gate"] == "CU" and not gate.get("qubits"):
                gate["qubits"] = [cu_idx, target]
                cu_idx += 1
    return circuit


@pytest.mark.parametrize("circuit_name,generator,n_qubits", PARAMS)
def test_benchmark_parity_full_range(circuit_name, generator, n_qubits):
    circuit = prepare_circuit(generator, n_qubits)
    total_qubits = circuit["number_of_qubits"]
    if total_qubits > MAX_COMPARE_QUBITS:
        # Ensure our frontend can still build the circuit without executing it.
        n, gates = circuit_dict_to_gates(circuit)
        assert n == total_qubits
        assert len(gates) == len(circuit["gates"])
        pytest.skip(
            f"{circuit_name}({n_qubits}) uses {total_qubits} qubits, "
            f"exceeding BENCHMARK_PARITY_MAX_QUBITS={MAX_COMPARE_QUBITS}."
        )
    my_state = run_my_impl(circuit)
    ref_state = run_reference_impl(circuit)
    assert my_state.shape == ref_state.shape
    assert np.allclose(my_state, ref_state, atol=1e-6)

