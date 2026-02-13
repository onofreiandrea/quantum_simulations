"""Known-vector tests for ref_dense."""
import numpy as np
import pytest
from wenbo_engine.kernel.ref_dense import simulate
from wenbo_engine.tests.fixtures.circuits import bell_2q, ghz, qft

S2 = 1.0 / np.sqrt(2.0)


def test_bell_state():
    psi = simulate(bell_2q())
    expected = np.array([S2, 0, 0, S2], dtype=complex)
    np.testing.assert_allclose(psi, expected, atol=1e-12)


def test_ghz3():
    psi = simulate(ghz(3))
    # (|000⟩ + |111⟩)/√2  →  indices 0 and 7
    assert abs(psi[0] - S2) < 1e-12
    assert abs(psi[7] - S2) < 1e-12
    assert abs(np.linalg.norm(psi) - 1.0) < 1e-12


def test_norm_preserved_qft():
    for n in [2, 3, 4, 5]:
        psi = simulate(qft(n))
        assert abs(np.linalg.norm(psi) - 1.0) < 1e-10


def test_hadamard_all():
    """H on all qubits → uniform superposition."""
    n = 4
    cd = {
        "number_of_qubits": n,
        "gates": [{"qubits": [i], "gate": "H"} for i in range(n)],
    }
    psi = simulate(cd)
    expected_amp = 1.0 / (2 ** (n / 2))
    np.testing.assert_allclose(np.abs(psi), expected_amp, atol=1e-12)


@pytest.mark.parametrize("n", [3, 4, 5, 6])
def test_random_circuit_norm(n):
    """Random 1q gates preserve norm."""
    import random
    random.seed(42 + n)
    gates_1q = ["H", "X", "Y", "Z", "S", "T"]
    cd = {
        "number_of_qubits": n,
        "gates": [
            {"qubits": [random.randrange(n)], "gate": random.choice(gates_1q)}
            for _ in range(20)
        ],
    }
    psi = simulate(cd)
    assert abs(np.linalg.norm(psi) - 1.0) < 1e-10
