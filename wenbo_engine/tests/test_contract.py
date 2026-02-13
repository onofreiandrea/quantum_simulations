"""Tests for circuit dict validation."""
import pytest
import numpy as np
from wenbo_engine.circuit.io import validate_circuit_dict
from wenbo_engine.tests.fixtures.circuits import bell_2q, cr3_encoded, ry_theta


def test_valid_bell():
    d = validate_circuit_dict(bell_2q())
    assert d["number_of_qubits"] == 2
    assert len(d["gates"]) == 2
    assert d["gates"][0]["gate"] == "H"
    assert d["gates"][1]["gate"] == "CNOT"


def test_valid_ry():
    d = validate_circuit_dict(ry_theta())
    assert d["gates"][0]["gate"] == "RY"
    assert abs(d["gates"][0]["params"]["theta"] - np.pi / 3) < 1e-12


def test_name_encoded_cr3():
    d = validate_circuit_dict(cr3_encoded())
    cr = d["gates"][2]
    assert cr["gate"] == "CR"
    assert cr["params"]["k"] == 3


def test_missing_nqubits():
    with pytest.raises(ValueError, match="missing required keys"):
        validate_circuit_dict({"gates": []})


def test_bad_gate_name():
    with pytest.raises(ValueError, match="unsupported gate"):
        validate_circuit_dict({
            "number_of_qubits": 2,
            "gates": [{"qubits": [0], "gate": "FOOBAR"}],
        })


def test_wrong_arity():
    with pytest.raises(ValueError, match="needs 1"):
        validate_circuit_dict({
            "number_of_qubits": 2,
            "gates": [{"qubits": [0, 1], "gate": "H"}],
        })


def test_qubit_out_of_range():
    with pytest.raises(ValueError, match="out of range"):
        validate_circuit_dict({
            "number_of_qubits": 2,
            "gates": [{"qubits": [5], "gate": "X"}],
        })


def test_extra_toplevel_key():
    with pytest.raises(ValueError, match="unknown top-level"):
        validate_circuit_dict({
            "number_of_qubits": 2, "gates": [], "extra": True,
        })


def test_missing_param():
    with pytest.raises(ValueError, match="requires param"):
        validate_circuit_dict({
            "number_of_qubits": 2,
            "gates": [{"qubits": [0], "gate": "RY"}],
        })


def test_unknown_gate_key():
    with pytest.raises(ValueError, match="unknown keys"):
        validate_circuit_dict({
            "number_of_qubits": 2,
            "gates": [{"qubits": [0], "gate": "H", "foo": 1}],
        })
