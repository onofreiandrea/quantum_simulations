"""End-to-end: single_node runner vs ref_dense."""
import tempfile
import numpy as np
import pytest
from wenbo_engine.kernel.ref_dense import simulate
from wenbo_engine.runner.single_node import run, collect_state
from wenbo_engine.tests.fixtures.circuits import bell_2q, ghz, qft


@pytest.mark.parametrize("circ_fn", [bell_2q, lambda: ghz(3), lambda: qft(3)])
@pytest.mark.parametrize("kernel", ["scalar", "batched"])
def test_single_node_vs_ref(circ_fn, kernel):
    cd = circ_fn()
    ref = simulate(cd)
    n = cd["number_of_qubits"]
    cs = 1 << n  # chunk_size = full state → 1 chunk
    with tempfile.TemporaryDirectory() as td:
        final = run(cd, td, chunk_size=cs, kernel=kernel)
        got = collect_state(final)
    np.testing.assert_allclose(got, ref, atol=1e-6)


@pytest.mark.parametrize("circ_fn", [bell_2q, lambda: ghz(4)])
def test_multi_chunk(circ_fn):
    """Multiple chunks (chunk_size < 2^n) still correct."""
    cd = circ_fn()
    ref = simulate(cd)
    n = cd["number_of_qubits"]
    # Use smallest possible chunk: 2^(max_qubit_in_circuit+1)
    # For chunk-local constraint all qubits must be < log2(chunk_size)
    cs = 1 << n  # must be >= 2^n for all qubits to be local
    with tempfile.TemporaryDirectory() as td:
        final = run(cd, td, chunk_size=cs)
        got = collect_state(final)
    np.testing.assert_allclose(got, ref, atol=1e-6)


def test_pipeline_runner():
    from wenbo_engine.runner.pipeline import run as prun
    cd = bell_2q()
    ref = simulate(cd)
    n = cd["number_of_qubits"]
    with tempfile.TemporaryDirectory() as td:
        final = prun(cd, td, chunk_size=1 << n)
        got = collect_state(final)
    np.testing.assert_allclose(got, ref, atol=1e-6)
