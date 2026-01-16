"""
CRITICAL TEST: Verify v3 works correctly and matches expected results.

This test is critical because it verifies:
1. Gate order is preserved correctly
2. Partitioning doesn't break correctness
3. Results match expected quantum states
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
generate_ghz_circuit = circuits.generate_ghz_circuit
generate_qft_circuit = circuits.generate_qft_circuit
generate_w_circuit = circuits.generate_w_circuit


@pytest.fixture
def config_v3():
    """Create a test configuration for v3."""
    temp_dir = Path(tempfile.mkdtemp())
    cfg = SimulatorConfig(
        run_id="test_critical",
        base_path=temp_dir,
        batch_size=10,
        spark_master="local[2]",
        spark_shuffle_partitions=4,
    )
    cfg.ensure_paths()
    yield cfg
    shutil.rmtree(temp_dir, ignore_errors=True)


def numpy_reference_ghz(n_qubits):
    """Reference GHZ state: (|00...0⟩ + |11...1⟩) / √2"""
    state = np.zeros(2**n_qubits, dtype=complex)
    state[0] = 1/np.sqrt(2)
    state[2**n_qubits - 1] = 1/np.sqrt(2)
    return state


def numpy_reference_qft(n_qubits):
    """Reference QFT|0⟩: uniform superposition"""
    state = np.ones(2**n_qubits, dtype=complex) / np.sqrt(2**n_qubits)
    return state


class TestCriticalCorrectness:
    """CRITICAL: These tests verify correctness."""
    
    @pytest.mark.parametrize("n_qubits", [2, 3, 4, 5])
    def test_ghz_matches_reference(self, config_v3, n_qubits):
        """CRITICAL: GHZ must match reference state."""
        circuit = generate_ghz_circuit(n_qubits)
        
        with SparkHiSVSIMDriver(config_v3) as driver:
            result = driver.run_circuit(circuit, n_partitions=2)
            state = driver.get_state_vector(result)
        
        expected = numpy_reference_ghz(n_qubits)
        np.testing.assert_allclose(state, expected, atol=1e-10, rtol=1e-10)
    
    @pytest.mark.parametrize("n_qubits", [2, 3, 4])
    def test_qft_matches_reference(self, config_v3, n_qubits):
        """CRITICAL: QFT|0⟩ must match reference state."""
        circuit = generate_qft_circuit(n_qubits)
        
        with SparkHiSVSIMDriver(config_v3) as driver:
            result = driver.run_circuit(circuit, n_partitions=3)
            state = driver.get_state_vector(result)
        
        expected = numpy_reference_qft(n_qubits)
        np.testing.assert_allclose(state, expected, atol=1e-10, rtol=1e-10)
    
    def test_gate_order_preservation_critical(self, config_v3):
        """CRITICAL: Gate order must be preserved regardless of partitioning."""
        # Create a circuit where gate order matters critically
        circuit = {
            "number_of_qubits": 2,
            "gates": [
                {"gate": "H", "qubits": [0]},
                {"gate": "CNOT", "qubits": [0, 1]},
                {"gate": "H", "qubits": [0]},  # Order matters!
            ]
        }
        
        # Test with different partition counts - all must produce same result
        states = []
        for n_partitions in [1, 2, 3]:
            with SparkHiSVSIMDriver(config_v3) as driver:
                result = driver.run_circuit(circuit, n_partitions=n_partitions)
                state = driver.get_state_vector(result)
                states.append(state)
        
        # All states must be identical
        for i in range(1, len(states)):
            np.testing.assert_allclose(states[0], states[i], atol=1e-10)
    
    def test_duplicate_gates_handled(self, config_v3):
        """CRITICAL: Duplicate gates must be handled correctly."""
        # Circuit with duplicate gates (same type, same qubits)
        circuit = {
            "number_of_qubits": 2,
            "gates": [
                {"gate": "H", "qubits": [0]},
                {"gate": "H", "qubits": [0]},  # H^2 = I
                {"gate": "CNOT", "qubits": [0, 1]},
            ]
        }
        
        # H^2 = I, so this should be equivalent to just CNOT
        with SparkHiSVSIMDriver(config_v3) as driver:
            result = driver.run_circuit(circuit, n_partitions=2)
            state_with_duplicates = driver.get_state_vector(result)
        
        # Compare with circuit without duplicates
        circuit_no_dup = {
            "number_of_qubits": 2,
            "gates": [
                {"gate": "CNOT", "qubits": [0, 1]},
            ]
        }
        
        with SparkHiSVSIMDriver(config_v3) as driver:
            result = driver.run_circuit(circuit_no_dup, n_partitions=1)
            state_no_duplicates = driver.get_state_vector(result)
        
        # Should be identical (H^2 = I)
        np.testing.assert_allclose(state_with_duplicates, state_no_duplicates, atol=1e-10)
    
    def test_partition_independence_critical(self, config_v3):
        """CRITICAL: Different partition counts must produce identical results."""
        circuit = generate_qft_circuit(4)
        
        states = []
        for n_partitions in [1, 2, 3, 4, 5]:
            with SparkHiSVSIMDriver(config_v3) as driver:
                result = driver.run_circuit(circuit, n_partitions=n_partitions)
                state = driver.get_state_vector(result)
                states.append(state)
        
        # All must be identical
        for i in range(1, len(states)):
            np.testing.assert_allclose(states[0], states[i], atol=1e-10, rtol=1e-10)


class TestGateMatchingBug:
    """Test for potential bugs in gate matching logic."""
    
    def test_gate_matching_with_same_gate_type(self, config_v3):
        """Test that gates with same type but different positions are matched correctly."""
        circuit = {
            "number_of_qubits": 3,
            "gates": [
                {"gate": "H", "qubits": [0]},
                {"gate": "H", "qubits": [1]},  # Same gate type, different qubit
                {"gate": "H", "qubits": [2]},  # Same gate type, different qubit
            ]
        }
        
        with SparkHiSVSIMDriver(config_v3) as driver:
            result = driver.run_circuit(circuit, n_partitions=2)
            state = driver.get_state_vector(result)
        
        # Should produce uniform superposition
        expected_mag = 1 / np.sqrt(8)
        for amp in state:
            assert np.isclose(np.abs(amp), expected_mag, atol=1e-10)
    
    def test_gate_matching_with_parameters(self, config_v3):
        """Test that parameterized gates are matched correctly."""
        circuit = {
            "number_of_qubits": 2,
            "gates": [
                {"gate": "CR", "qubits": [0, 1], "params": {"k": 2}},
                {"gate": "CR", "qubits": [0, 1], "params": {"k": 3}},  # Different k
            ]
        }
        
        with SparkHiSVSIMDriver(config_v3) as driver:
            result = driver.run_circuit(circuit, n_partitions=2)
            state = driver.get_state_vector(result)
        
        # Just verify it runs without error and produces normalized state
        norm = np.linalg.norm(state)
        assert np.isclose(norm, 1.0, atol=1e-10)
