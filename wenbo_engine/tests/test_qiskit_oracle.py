"""Compare ref_dense against Qiskit Statevector on MQT Bench circuits."""
import pytest
import numpy as np

try:
    from qiskit import QuantumCircuit, transpile
    from qiskit.quantum_info import Statevector
    HAS_QISKIT = True
except ImportError:
    HAS_QISKIT = False

try:
    from mqt.bench import get_benchmark
    from mqt.bench.benchmark_generation import BenchmarkLevel
    HAS_MQT = True
except ImportError:
    HAS_MQT = False

from wenbo_engine.kernel.ref_dense import simulate
from wenbo_engine.circuit.import_qiskit import qiskit_to_dict, SUPPORTED_BASIS


def _compare(qc, atol=1e-8):
    """Transpile qc, convert, simulate, compare to Qiskit Statevector."""
    qc_t = transpile(qc, basis_gates=SUPPORTED_BASIS, optimization_level=0)
    cd = qiskit_to_dict(qc_t)
    ours = simulate(cd)
    ref = np.array(Statevector(qc).data)
    # Global phase may differ; compare |amplitudes| or use phase-invariant check
    overlap = np.abs(np.vdot(ref, ours))
    assert overlap > 1.0 - atol, f"overlap={overlap}"


@pytest.mark.skipif(not HAS_QISKIT, reason="qiskit not installed")
class TestQiskitDirect:
    def test_bell(self):
        qc = QuantumCircuit(2)
        qc.h(0); qc.cx(0, 1)
        _compare(qc)

    def test_ghz4(self):
        qc = QuantumCircuit(4)
        qc.h(0)
        for i in range(1, 4):
            qc.cx(i - 1, i)
        _compare(qc)

    @pytest.mark.parametrize("n", [3, 4, 5])
    def test_random_cliffords(self, n):
        qc = QuantumCircuit(n)
        rng = np.random.default_rng(n)
        for _ in range(3 * n):
            q = int(rng.integers(n))
            qc.h(q)
        for _ in range(n):
            a, b = rng.choice(n, size=2, replace=False)
            qc.cx(int(a), int(b))
        _compare(qc)


@pytest.mark.skipif(not HAS_QISKIT or not HAS_MQT, reason="qiskit/mqt.bench not installed")
class TestMQTBench:
    @pytest.mark.parametrize("bench_name", ["ghz", "dj", "graphstate"])
    @pytest.mark.parametrize("n", [3, 4, 5])
    def test_mqt_small(self, bench_name, n):
        qc = get_benchmark(bench_name, circuit_size=n, level=BenchmarkLevel.INDEP)
        qc.remove_final_measurements(inplace=True)
        _compare(qc, atol=1e-6)
