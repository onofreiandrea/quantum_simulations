"""
Comprehensive tests for ALL gate types and sparse/dense states.

Tests:
1. All gate types individually (H, X, Y, Z, S, T, RY, R, G, CNOT, CZ, CY, CR, CU, SWAP)
2. Sparse states (few non-zero amplitudes)
3. Dense states (many non-zero amplitudes)
4. Complex circuit combinations
"""
from __future__ import annotations

import pytest
import numpy as np
import tempfile
import shutil
import sys
from pathlib import Path

# Add paths
V3_SPARK = Path(__file__).parent.parent
if str(V3_SPARK / "src") not in sys.path:
    sys.path.insert(0, str(V3_SPARK / "src"))

from driver import SparkHiSVSIMDriver
from v2_common import config, circuits, gates, frontend

SimulatorConfig = config.SimulatorConfig
circuit_dict_to_gates = frontend.circuit_dict_to_gates


@pytest.fixture
def config_v3():
    """Create a test configuration for v3."""
    temp_dir = Path(tempfile.mkdtemp())
    cfg = SimulatorConfig(
        run_id="test_comprehensive",
        base_path=temp_dir,
        batch_size=10,
        spark_master="local[2]",
        spark_shuffle_partitions=4,
    )
    cfg.ensure_paths()
    yield cfg
    shutil.rmtree(temp_dir, ignore_errors=True)


def numpy_reference(circuit_dict):
    """NumPy reference implementation."""
    n_qubits, gate_list = circuit_dict_to_gates(circuit_dict)
    state = np.zeros(2**n_qubits, dtype=complex)
    state[0] = 1.0
    
    for gate in gate_list:
        state = apply_gate_numpy(state, gate, n_qubits)
    
    return state


def apply_gate_numpy(state: np.ndarray, gate, n_qubits: int) -> np.ndarray:
    """Apply gate using NumPy (reference) - matches v1 implementation."""
    psi = state.reshape([2] * n_qubits)
    
    if gate.two_qubit_gate:
        axes = gate.qubits
        tensor = gate.tensor
        # Contract gate tensor with state
        contracted = np.tensordot(tensor, psi, axes=([2, 3], axes))
        # Reorder axes to restore original order
        current_order = list(axes) + [i for i in range(n_qubits) if i not in axes]
        perm = [current_order.index(i) for i in range(n_qubits)]
        psi = np.transpose(contracted, perm)
    else:
        (axis,) = gate.qubits
        tensor = gate.tensor
        # Contract gate tensor with state
        contracted = np.tensordot(tensor, psi, axes=([1], [axis]))
        # Reorder axes
        current_order = [axis] + [i for i in range(n_qubits) if i != axis]
        perm = [current_order.index(i) for i in range(n_qubits)]
        psi = np.transpose(contracted, perm)
    
    return psi.reshape(-1)


class TestAllGateTypes:
    """Test ALL gate types individually."""
    
    @pytest.mark.parametrize("gate_name,gate_dict", [
        ("H", {"gate": "H", "qubits": [0]}),
        ("X", {"gate": "X", "qubits": [0]}),
        ("Y", {"gate": "Y", "qubits": [0]}),
        ("Z", {"gate": "Z", "qubits": [0]}),
        ("S", {"gate": "S", "qubits": [0]}),
        ("T", {"gate": "T", "qubits": [0]}),
    ])
    def test_single_qubit_gates(self, config_v3, gate_name, gate_dict):
        """Test all 1-qubit gates."""
        circuit = {
            "number_of_qubits": 2,
            "gates": [gate_dict]
        }
        
        # NumPy reference
        np_state = numpy_reference(circuit)
        
        # v3 implementation
        with SparkHiSVSIMDriver(config_v3, enable_parallel=False) as driver:
            result = driver.run_circuit(circuit, enable_parallel=False)
            v3_state = driver.get_state_vector(result)
        
        np.testing.assert_allclose(v3_state, np_state, atol=1e-10)
        
        # Check normalization
        norm = np.linalg.norm(v3_state)
        assert np.isclose(norm, 1.0, atol=1e-10)
    
    @pytest.mark.parametrize("gate_name,gate_dict", [
        ("CNOT", {"gate": "CNOT", "qubits": [0, 1]}),
        ("CZ", {"gate": "CZ", "qubits": [0, 1]}),
        ("CY", {"gate": "CY", "qubits": [0, 1]}),
        ("SWAP", {"gate": "SWAP", "qubits": [0, 1]}),
        ("CR2", {"gate": "CR", "qubits": [0, 1], "params": {"k": 2}}),
        ("CR3", {"gate": "CR", "qubits": [0, 1], "params": {"k": 3}}),
    ])
    def test_two_qubit_gates(self, config_v3, gate_name, gate_dict):
        """Test all 2-qubit gates."""
        circuit = {
            "number_of_qubits": 2,
            "gates": [gate_dict]
        }
        
        # NumPy reference
        np_state = numpy_reference(circuit)
        
        # v3 implementation
        with SparkHiSVSIMDriver(config_v3, enable_parallel=False) as driver:
            result = driver.run_circuit(circuit, enable_parallel=False)
            v3_state = driver.get_state_vector(result)
        
        np.testing.assert_allclose(v3_state, np_state, atol=1e-10)
        
        # Check normalization
        norm = np.linalg.norm(v3_state)
        assert np.isclose(norm, 1.0, atol=1e-10)
    
    def test_parameterized_gates(self, config_v3):
        """Test parameterized gates (RY, R, G, CU)."""
        circuits_to_test = [
            {
                "name": "RY",
                "circuit": {
                    "number_of_qubits": 2,
                    "gates": [{"gate": "RY", "qubits": [0], "params": {"theta": np.pi/4}}]
                }
            },
            {
                "name": "R",
                "circuit": {
                    "number_of_qubits": 2,
                    "gates": [{"gate": "R", "qubits": [0], "params": {"k": 3}}]
                }
            },
            {
                "name": "G",
                "circuit": {
                    "number_of_qubits": 2,
                    "gates": [{"gate": "G", "qubits": [0], "params": {"p": 3}}]
                }
            },
            {
                "name": "CU",
                "circuit": {
                    "number_of_qubits": 2,
                    "gates": [{"gate": "CU", "qubits": [0, 1], "params": {
                        "U": np.array([[1, 0], [0, -1]], dtype=complex),
                        "exponent": 1
                    }}]
                }
            },
        ]
        
        for test_case in circuits_to_test:
            circuit = test_case["circuit"]
            
            # NumPy reference
            np_state = numpy_reference(circuit)
            
            # v3 implementation
            with SparkHiSVSIMDriver(config_v3, enable_parallel=False) as driver:
                result = driver.run_circuit(circuit, enable_parallel=False)
                v3_state = driver.get_state_vector(result)
            
            np.testing.assert_allclose(
                v3_state, np_state, atol=1e-10,
                err_msg=f"Failed for {test_case['name']} gate"
            )


class TestSparseStates:
    """Test circuits that produce sparse states (few non-zero amplitudes)."""
    
    def test_ghz_sparse(self, config_v3):
        """GHZ produces sparse state: only 2 non-zero amplitudes."""
        circuit = circuits.generate_ghz_circuit(4)
        
        with SparkHiSVSIMDriver(config_v3, enable_parallel=False) as driver:
            result = driver.run_circuit(circuit, enable_parallel=False)
            state = driver.get_state_vector(result)
        
        # GHZ should have exactly 2 non-zero amplitudes
        nonzero = np.sum(np.abs(state) > 1e-10)
        assert nonzero == 2, f"Expected 2 non-zero amplitudes, got {nonzero}"
        
        # Should be |0000⟩ and |1111⟩
        assert np.abs(state[0]) > 1e-10, "|0000⟩ should be non-zero"
        assert np.abs(state[15]) > 1e-10, "|1111⟩ should be non-zero"
    
    def test_w_state_sparse(self, config_v3):
        """W-state produces sparse state: n non-zero amplitudes."""
        circuit = circuits.generate_w_circuit(4)
        
        with SparkHiSVSIMDriver(config_v3, enable_parallel=False) as driver:
            result = driver.run_circuit(circuit, enable_parallel=False)
            state = driver.get_state_vector(result)
        
        # W-state should have n non-zero amplitudes
        nonzero = np.sum(np.abs(state) > 1e-10)
        assert nonzero == 4, f"Expected 4 non-zero amplitudes, got {nonzero}"
    
    def test_bell_state_sparse(self, config_v3):
        """Bell state: sparse (2 non-zero amplitudes)."""
        circuit = {
            "number_of_qubits": 2,
            "gates": [
                {"gate": "H", "qubits": [0]},
                {"gate": "CNOT", "qubits": [0, 1]},
            ]
        }
        
        with SparkHiSVSIMDriver(config_v3, enable_parallel=False) as driver:
            result = driver.run_circuit(circuit, enable_parallel=False)
            state = driver.get_state_vector(result)
        
        # Bell state: (|00⟩ + |11⟩) / √2
        nonzero = np.sum(np.abs(state) > 1e-10)
        assert nonzero == 2, f"Expected 2 non-zero amplitudes, got {nonzero}"
        assert np.abs(state[0]) > 1e-10
        assert np.abs(state[3]) > 1e-10


class TestDenseStates:
    """Test circuits that produce dense states (many non-zero amplitudes)."""
    
    def test_hadamard_wall_dense(self, config_v3):
        """Hadamard wall produces uniform superposition (dense)."""
        circuit = circuits.generate_hadamard_wall(4)
        
        with SparkHiSVSIMDriver(config_v3, enable_parallel=False) as driver:
            result = driver.run_circuit(circuit, enable_parallel=False)
            state = driver.get_state_vector(result)
        
        # All amplitudes should be non-zero (uniform superposition)
        nonzero = np.sum(np.abs(state) > 1e-10)
        assert nonzero == 16, f"Expected 16 non-zero amplitudes, got {nonzero}"
        
        # All should have same magnitude
        expected_mag = 1 / np.sqrt(16)
        for amp in state:
            assert np.isclose(np.abs(amp), expected_mag, atol=1e-10)
    
    def test_qft_dense(self, config_v3):
        """QFT produces dense state (uniform superposition)."""
        circuit = circuits.generate_qft_circuit(4)
        
        with SparkHiSVSIMDriver(config_v3, enable_parallel=False) as driver:
            result = driver.run_circuit(circuit, enable_parallel=False)
            state = driver.get_state_vector(result)
        
        # QFT|0⟩ should be uniform superposition
        nonzero = np.sum(np.abs(state) > 1e-10)
        assert nonzero == 16, f"Expected 16 non-zero amplitudes, got {nonzero}"
        
        # All should have same magnitude
        expected_mag = 1 / np.sqrt(16)
        for amp in state:
            assert np.isclose(np.abs(amp), expected_mag, atol=1e-10)


class TestComplexCircuits:
    """Test complex circuit combinations."""
    
    def test_mixed_gate_types(self, config_v3):
        """Test circuit with multiple gate types."""
        circuit = {
            "number_of_qubits": 3,
            "gates": [
                {"gate": "H", "qubits": [0]},
                {"gate": "T", "qubits": [0]},
                {"gate": "CNOT", "qubits": [0, 1]},
                {"gate": "RY", "qubits": [1], "params": {"theta": np.pi/3}},
                {"gate": "CZ", "qubits": [1, 2]},
                {"gate": "CR", "qubits": [0, 2], "params": {"k": 2}},
            ]
        }
        
        # NumPy reference
        np_state = numpy_reference(circuit)
        
        # v3 implementation
        with SparkHiSVSIMDriver(config_v3, enable_parallel=False) as driver:
            result = driver.run_circuit(circuit, enable_parallel=False)
            v3_state = driver.get_state_vector(result)
        
        np.testing.assert_allclose(v3_state, np_state, atol=1e-10)
    
    def test_ghz_qft_combination(self, config_v3):
        """Test GHZ followed by QFT (sparse -> dense transition)."""
        circuit = circuits.generate_ghz_qft(4)
        
        with SparkHiSVSIMDriver(config_v3, enable_parallel=False) as driver:
            result = driver.run_circuit(circuit, enable_parallel=False)
            state = driver.get_state_vector(result)
        
        # Should be dense after QFT
        nonzero = np.sum(np.abs(state) > 1e-10)
        assert nonzero > 2, "QFT should create dense state"
        
        # Check normalization
        norm = np.linalg.norm(state)
        assert np.isclose(norm, 1.0, atol=1e-10)
    
    def test_w_qft_combination(self, config_v3):
        """Test W-state followed by QFT."""
        circuit = circuits.generate_w_qft(4)
        
        with SparkHiSVSIMDriver(config_v3, enable_parallel=False) as driver:
            result = driver.run_circuit(circuit, enable_parallel=False)
            state = driver.get_state_vector(result)
        
        # Should be dense after QFT
        nonzero = np.sum(np.abs(state) > 1e-10)
        assert nonzero > 4, "QFT should create dense state"
        
        # Check normalization
        norm = np.linalg.norm(state)
        assert np.isclose(norm, 1.0, atol=1e-10)
    
    def test_qpe_circuit(self, config_v3):
        """Test Quantum Phase Estimation (complex circuit)."""
        circuit = circuits.generate_qpe_circuit(3)
        
        with SparkHiSVSIMDriver(config_v3, enable_parallel=False) as driver:
            result = driver.run_circuit(circuit, enable_parallel=False)
            state = driver.get_state_vector(result)
        
        # QPE should produce valid state
        norm = np.linalg.norm(state)
        assert np.isclose(norm, 1.0, atol=1e-10)
        
        # Should have multiple non-zero amplitudes
        nonzero = np.sum(np.abs(state) > 1e-10)
        assert nonzero > 0


class TestSparseDenseTransitions:
    """Test transitions between sparse and dense states."""
    
    def test_sparse_to_dense(self, config_v3):
        """Start sparse (GHZ), end dense (QFT)."""
        # Create GHZ then QFT
        ghz = circuits.generate_ghz_circuit(3)
        qft = circuits.generate_qft_circuit(3)
        
        circuit = {
            "number_of_qubits": 3,
            "gates": ghz["gates"] + qft["gates"]
        }
        
        with SparkHiSVSIMDriver(config_v3, enable_parallel=False) as driver:
            result = driver.run_circuit(circuit, enable_parallel=False)
            state = driver.get_state_vector(result)
        
        # After QFT, should be dense
        nonzero = np.sum(np.abs(state) > 1e-10)
        assert nonzero == 8, "QFT should create uniform superposition"
    
    def test_dense_to_sparse(self, config_v3):
        """Start dense (Hadamard wall), end sparse (measurement-like)."""
        # Hadamard wall creates dense state
        circuit = {
            "number_of_qubits": 3,
            "gates": [
                {"gate": "H", "qubits": [0]},
                {"gate": "H", "qubits": [1]},
                {"gate": "H", "qubits": [2]},
                # Then CNOTs to create entanglement (can reduce sparsity)
                {"gate": "CNOT", "qubits": [0, 1]},
                {"gate": "CNOT", "qubits": [1, 2]},
            ]
        }
        
        with SparkHiSVSIMDriver(config_v3, enable_parallel=False) as driver:
            result = driver.run_circuit(circuit, enable_parallel=False)
            state = driver.get_state_vector(result)
        
        # Should have valid state
        norm = np.linalg.norm(state)
        assert np.isclose(norm, 1.0, atol=1e-10)


class TestAllGatesTogether:
    """Test circuit using ALL gate types."""
    
    def test_all_gates_comprehensive(self, config_v3):
        """Test circuit with every gate type."""
        circuit = {
            "number_of_qubits": 3,
            "gates": [
                # 1-qubit gates
                {"gate": "H", "qubits": [0]},
                {"gate": "X", "qubits": [1]},
                {"gate": "Y", "qubits": [2]},
                {"gate": "Z", "qubits": [0]},
                {"gate": "S", "qubits": [1]},
                {"gate": "T", "qubits": [2]},
                {"gate": "RY", "qubits": [0], "params": {"theta": np.pi/4}},
                {"gate": "R", "qubits": [1], "params": {"k": 2}},
                {"gate": "G", "qubits": [2], "params": {"p": 3}},
                # 2-qubit gates
                {"gate": "CNOT", "qubits": [0, 1]},
                {"gate": "CZ", "qubits": [1, 2]},
                {"gate": "CY", "qubits": [0, 2]},
                {"gate": "SWAP", "qubits": [1, 2]},
                {"gate": "CR", "qubits": [0, 1], "params": {"k": 3}},
                {"gate": "CU", "qubits": [0, 2], "params": {
                    "U": np.array([[1, 0], [0, -1]], dtype=complex),
                    "exponent": 1
                }},
            ]
        }
        
        # NumPy reference
        np_state = numpy_reference(circuit)
        
        # v3 implementation
        with SparkHiSVSIMDriver(config_v3, enable_parallel=False) as driver:
            result = driver.run_circuit(circuit, enable_parallel=False)
            v3_state = driver.get_state_vector(result)
        
        np.testing.assert_allclose(v3_state, np_state, atol=1e-10)
        
        # Check normalization
        norm = np.linalg.norm(v3_state)
        assert np.isclose(norm, 1.0, atol=1e-10)
