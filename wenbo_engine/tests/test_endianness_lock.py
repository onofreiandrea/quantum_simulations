"""Lock the endianness convention: LITTLE-ENDIAN.

X on qubit 0 from |000⟩  must put all amplitude at index 1.
"""
import numpy as np
from wenbo_engine.circuit.io import ENDIANNESS
from wenbo_engine.kernel.ref_dense import simulate
from wenbo_engine.tests.fixtures.circuits import x_on_q0_3q


def test_endianness_is_little():
    assert ENDIANNESS == "little"


def test_x_on_q0_amplitude_at_index_1():
    psi = simulate(x_on_q0_3q())
    # |000⟩ → X(q0) → |001⟩  → index 1 in little-endian
    assert abs(psi[1]) > 0.999
    assert abs(np.linalg.norm(psi) - 1.0) < 1e-12
    # all other amplitudes ~0
    for i in range(8):
        if i != 1:
            assert abs(psi[i]) < 1e-12
