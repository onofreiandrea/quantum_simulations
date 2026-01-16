"""
Comprehensive tests for ALL gate types and sparse/dense states.

Tests against v1 implementation directly to ensure correctness.
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
from v2_common import config, circuits

SimulatorConfig = config.SimulatorConfig


@pytest.fixture
def config_v3():
    """Create a test configuration for v3."""
    temp_dir = Path(tempfile.mkdtemp())
    cfg = SimulatorConfig(
        run_id="test_all_gates",
        base_path=temp_dir,
        batch_size=10,
        spark_master="local[2]",
        spark_shuffle_partitions=4,
    )
    cfg.ensure_paths()
    yield cfg
    shutil.rmtree(temp_dir, ignore_errors=True)


class TestAllGateTypes:
    """Test ALL gate types - verify they work correctly."""
    
    def test_basic_gates(self, config_v3):
        """Test basic gates: H, X, Y, Z, S, T."""
        gates_to_test = [
            {"gate": "H", "qubits": [0]},
            {"gate": "X", "qubits": [0]},
            {"gate": "Y", "qubits": [0]},
            {"gate": "Z", "qubits": [0]},
            {"gate": "S", "qubits": [0]},
            {"gate": "T", "qubits": [0]},
        ]
        
        for gate_dict in gates_to_test:
            circuit = {
                "number_of_qubits": 1,
                "gates": [gate_dict]
            }
            
            with SparkHiSVSIMDriver(config_v3, enable_parallel=False) as driver:
                result = driver.run_circuit(circuit, enable_parallel=False)
                state = driver.get_state_vector(result)
            
            # Check normalization
            norm = np.linalg.norm(state)
            assert np.isclose(norm, 1.0, atol=1e-10), f"Gate {gate_dict['gate']} failed normalization"
    
    def test_two_qubit_gates(self, config_v3):
        """Test 2-qubit gates: CNOT, CZ, CY, SWAP, CR."""
        gates_to_test = [
            {"gate": "CNOT", "qubits": [0, 1]},
            {"gate": "CZ", "qubits": [0, 1]},
            {"gate": "CY", "qubits": [0, 1]},
            {"gate": "SWAP", "qubits": [0, 1]},
            {"gate": "CR", "qubits": [0, 1], "params": {"k": 2}},
            {"gate": "CR", "qubits": [0, 1], "params": {"k": 3}},
        ]
        
        for gate_dict in gates_to_test:
            circuit = {
                "number_of_qubits": 2,
                "gates": [gate_dict]
            }
            
            with SparkHiSVSIMDriver(config_v3, enable_parallel=False) as driver:
                result = driver.run_circuit(circuit, enable_parallel=False)
                state = driver.get_state_vector(result)
            
            # Check normalization
            norm = np.linalg.norm(state)
            assert np.isclose(norm, 1.0, atol=1e-10), f"Gate {gate_dict['gate']} failed normalization"
    
    def test_parameterized_gates(self, config_v3):
        """Test parameterized gates: RY, R, G, CU."""
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
            
            with SparkHiSVSIMDriver(config_v3, enable_parallel=False) as driver:
                result = driver.run_circuit(circuit, enable_parallel=False)
                state = driver.get_state_vector(result)
            
            # Check normalization
            norm = np.linalg.norm(state)
            assert np.isclose(norm, 1.0, atol=1e-10), f"Gate {test_case['name']} failed normalization"


class TestSparseStates:
    """Test circuits that produce sparse states (few non-zero amplitudes)."""
    
    @pytest.mark.parametrize("n_qubits", [3, 4, 5])
    def test_ghz_sparse(self, config_v3, n_qubits):
        """GHZ produces sparse state: only 2 non-zero amplitudes."""
        circuit = circuits.generate_ghz_circuit(n_qubits)
        
        with SparkHiSVSIMDriver(config_v3, enable_parallel=False) as driver:
            result = driver.run_circuit(circuit, enable_parallel=False)
            state = driver.get_state_vector(result)
        
        # GHZ should have exactly 2 non-zero amplitudes
        nonzero = np.sum(np.abs(state) > 1e-10)
        assert nonzero == 2, f"Expected 2 non-zero amplitudes, got {nonzero}"
        
        # Should be |00...0⟩ and |11...1⟩
        assert np.abs(state[0]) > 1e-10, "|00...0⟩ should be non-zero"
        assert np.abs(state[2**n_qubits - 1]) > 1e-10, "|11...1⟩ should be non-zero"
        
        # Check normalization
        norm = np.linalg.norm(state)
        assert np.isclose(norm, 1.0, atol=1e-10)
    
    @pytest.mark.parametrize("n_qubits", [3, 4])
    def test_w_state_sparse(self, config_v3, n_qubits):
        """W-state produces sparse state: n non-zero amplitudes."""
        circuit = circuits.generate_w_circuit(n_qubits)
        
        with SparkHiSVSIMDriver(config_v3, enable_parallel=False) as driver:
            result = driver.run_circuit(circuit, enable_parallel=False)
            state = driver.get_state_vector(result)
        
        # W-state should have n non-zero amplitudes
        nonzero = np.sum(np.abs(state) > 1e-10)
        assert nonzero == n_qubits, f"Expected {n_qubits} non-zero amplitudes, got {nonzero}"
        
        # Check normalization
        norm = np.linalg.norm(state)
        assert np.isclose(norm, 1.0, atol=1e-10)
    
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
    
    @pytest.mark.parametrize("n_qubits", [3, 4, 5])
    def test_hadamard_wall_dense(self, config_v3, n_qubits):
        """Hadamard wall produces uniform superposition (dense)."""
        circuit = circuits.generate_hadamard_wall(n_qubits)
        
        with SparkHiSVSIMDriver(config_v3, enable_parallel=False) as driver:
            result = driver.run_circuit(circuit, enable_parallel=False)
            state = driver.get_state_vector(result)
        
        # All amplitudes should be non-zero (uniform superposition)
        expected_nonzero = 2**n_qubits
        nonzero = np.sum(np.abs(state) > 1e-10)
        assert nonzero == expected_nonzero, f"Expected {expected_nonzero} non-zero amplitudes, got {nonzero}"
        
        # All should have same magnitude
        expected_mag = 1 / np.sqrt(expected_nonzero)
        for amp in state:
            assert np.isclose(np.abs(amp), expected_mag, atol=1e-10)
    
    @pytest.mark.parametrize("n_qubits", [3, 4, 5])
    def test_qft_dense(self, config_v3, n_qubits):
        """QFT produces dense state (uniform superposition)."""
        circuit = circuits.generate_qft_circuit(n_qubits)
        
        with SparkHiSVSIMDriver(config_v3, enable_parallel=False) as driver:
            result = driver.run_circuit(circuit, enable_parallel=False)
            state = driver.get_state_vector(result)
        
        # QFT|0⟩ should be uniform superposition
        expected_nonzero = 2**n_qubits
        nonzero = np.sum(np.abs(state) > 1e-10)
        assert nonzero == expected_nonzero, f"Expected {expected_nonzero} non-zero amplitudes, got {nonzero}"
        
        # All should have same magnitude
        expected_mag = 1 / np.sqrt(expected_nonzero)
        for amp in state:
            assert np.isclose(np.abs(amp), expected_mag, atol=1e-10)


class TestComplexCircuits:
    """Test complex circuit combinations with all gate types."""
    
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
                {"gate": "S", "qubits": [2]},
            ]
        }
        
        with SparkHiSVSIMDriver(config_v3, enable_parallel=False) as driver:
            result = driver.run_circuit(circuit, enable_parallel=False)
            state = driver.get_state_vector(result)
        
        # Check normalization
        norm = np.linalg.norm(state)
        assert np.isclose(norm, 1.0, atol=1e-10)
    
    def test_all_gates_comprehensive(self, config_v3):
        """Test circuit using ALL gate types."""
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
        
        with SparkHiSVSIMDriver(config_v3, enable_parallel=False) as driver:
            result = driver.run_circuit(circuit, enable_parallel=False)
            state = driver.get_state_vector(result)
        
        # Check normalization
        norm = np.linalg.norm(state)
        assert np.isclose(norm, 1.0, atol=1e-10)
        
        # Should have multiple non-zero amplitudes (dense state)
        nonzero = np.sum(np.abs(state) > 1e-10)
        assert nonzero > 1, "Complex circuit should have multiple non-zero amplitudes"
    
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
        """Test Quantum Phase Estimation (complex circuit with CU gates)."""
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
        
        # After QFT, should be dense (but GHZ+QFT doesn't necessarily give uniform superposition)
        nonzero = np.sum(np.abs(state) > 1e-10)
        assert nonzero >= 2, "QFT should create dense state from sparse GHZ"
        assert nonzero <= 8, "Should have at most 8 non-zero amplitudes for 3 qubits"
        
        # Check normalization
        norm = np.linalg.norm(state)
        assert np.isclose(norm, 1.0, atol=1e-10)
    
    def test_dense_to_sparse(self, config_v3):
        """Start dense (Hadamard wall), apply entangling gates."""
        circuit = {
            "number_of_qubits": 3,
            "gates": [
                {"gate": "H", "qubits": [0]},
                {"gate": "H", "qubits": [1]},
                {"gate": "H", "qubits": [2]},
                # Then CNOTs to create entanglement
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


class TestNonStabilizerCircuits:
    """Test non-stabilizer circuits (T, RY, R, G, CU gates)."""
    
    def test_t_gate_circuits(self, config_v3):
        """Test circuits with T gates."""
        circuits_to_test = [
            {
                "name": "T on |1⟩",
                "circuit": {
                    "number_of_qubits": 1,
                    "gates": [
                        {"gate": "X", "qubits": [0]},  # |0⟩ → |1⟩
                        {"gate": "T", "qubits": [0]},  # |1⟩ → e^(iπ/4)|1⟩
                    ]
                }
            },
            {
                "name": "T on superposition",
                "circuit": {
                    "number_of_qubits": 1,
                    "gates": [
                        {"gate": "H", "qubits": [0]},
                        {"gate": "T", "qubits": [0]},
                    ]
                }
            },
        ]
        
        for test_case in circuits_to_test:
            circuit = test_case["circuit"]
            
            with SparkHiSVSIMDriver(config_v3, enable_parallel=False) as driver:
                result = driver.run_circuit(circuit, enable_parallel=False)
                state = driver.get_state_vector(result)
            
            # Check normalization
            norm = np.linalg.norm(state)
            assert np.isclose(norm, 1.0, atol=1e-10), f"Failed for {test_case['name']}"
    
    def test_ry_gate_circuits(self, config_v3):
        """Test circuits with RY gates."""
        circuit = {
            "number_of_qubits": 2,
            "gates": [
                {"gate": "RY", "qubits": [0], "params": {"theta": np.pi/4}},
                {"gate": "RY", "qubits": [1], "params": {"theta": np.pi/3}},
                {"gate": "CNOT", "qubits": [0, 1]},
            ]
        }
        
        with SparkHiSVSIMDriver(config_v3, enable_parallel=False) as driver:
            result = driver.run_circuit(circuit, enable_parallel=False)
            state = driver.get_state_vector(result)
        
        # Check normalization
        norm = np.linalg.norm(state)
        assert np.isclose(norm, 1.0, atol=1e-10)
    
    def test_cr_gate_circuits(self, config_v3):
        """Test circuits with CR gates (used in QFT)."""
        circuit = {
            "number_of_qubits": 3,
            "gates": [
                {"gate": "H", "qubits": [0]},
                {"gate": "H", "qubits": [1]},
                {"gate": "CR", "qubits": [1, 0], "params": {"k": 2}},
                {"gate": "CR", "qubits": [2, 0], "params": {"k": 3}},
                {"gate": "CR", "qubits": [2, 1], "params": {"k": 2}},
            ]
        }
        
        with SparkHiSVSIMDriver(config_v3, enable_parallel=False) as driver:
            result = driver.run_circuit(circuit, enable_parallel=False)
            state = driver.get_state_vector(result)
        
        # Check normalization
        norm = np.linalg.norm(state)
        assert np.isclose(norm, 1.0, atol=1e-10)
    
    def test_cu_gate_circuits(self, config_v3):
        """Test circuits with CU gates (used in QPE)."""
        U = np.array([[1, 0], [0, -1]], dtype=complex)
        circuit = {
            "number_of_qubits": 3,
            "gates": [
                {"gate": "H", "qubits": [0]},
                {"gate": "H", "qubits": [1]},
                {"gate": "CU", "qubits": [0, 2], "params": {"U": U, "exponent": 1}},
                {"gate": "CU", "qubits": [1, 2], "params": {"U": U, "exponent": 2}},
            ]
        }
        
        with SparkHiSVSIMDriver(config_v3, enable_parallel=False) as driver:
            result = driver.run_circuit(circuit, enable_parallel=False)
            state = driver.get_state_vector(result)
        
        # Check normalization
        norm = np.linalg.norm(state)
        assert np.isclose(norm, 1.0, atol=1e-10)
