"""Compare cpu_scalar and cpu_batched kernels against ref_dense."""
import numpy as np
import pytest
from wenbo_engine.kernel.ref_dense import simulate
from wenbo_engine.kernel import gates as gmod
from wenbo_engine.kernel import cpu_scalar, cpu_batched
from wenbo_engine.circuit.io import validate_circuit_dict
from wenbo_engine.tests.fixtures.circuits import bell_2q, ghz, qft, ry_theta


def _run_kernel(circuit_dict, mod):
    cd = validate_circuit_dict(circuit_dict)
    n = cd["number_of_qubits"]
    psi = np.zeros(1 << n, dtype=np.complex128)
    psi[0] = 1.0
    for g in cd["gates"]:
        U = gmod.gate_matrix(g["gate"], g["params"])
        qs = g["qubits"]
        if len(qs) == 1:
            mod.apply_1q(psi, qs[0], U)
        else:
            mod.apply_2q(psi, qs[0], qs[1], U)
    return psi


@pytest.mark.parametrize("circ_fn", [bell_2q, ry_theta, lambda: ghz(4), lambda: qft(3)])
@pytest.mark.parametrize("mod", [cpu_scalar, cpu_batched])
def test_kernel_matches_ref(circ_fn, mod):
    cd = circ_fn()
    ref = simulate(cd)
    got = _run_kernel(cd, mod)
    np.testing.assert_allclose(got, ref, atol=1e-10)


def test_non_local_raises():
    """Qubit >= log2(chunk_size) must raise NotImplementedError."""
    chunk = np.zeros(4, dtype=np.complex128)  # chunk_size=4 → 2 qubits
    chunk[0] = 1.0
    U = gmod.H()
    with pytest.raises(NotImplementedError, match="non-local"):
        cpu_scalar.apply_1q(chunk, 2, U)
    with pytest.raises(NotImplementedError, match="non-local"):
        cpu_batched.apply_1q(chunk, 2, U)
