"""
Tests specifically for non-stabilizer (non-Clifford) circuits.

Stabilizer circuits (Clifford gates only: H, S, CNOT, CZ, X, Y, Z) can be
efficiently simulated classically via the Gottesman-Knill theorem.

Non-stabilizer circuits require gates like T, arbitrary rotations, or Toffoli.
These are what make quantum computation powerful and hard to simulate.

This test suite ensures the simulator correctly handles non-stabilizer circuits.
"""
from __future__ import annotations

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path

from src.config import SimulatorConfig
from src.driver import SparkQuantumDriver
from src.gates import (
    HadamardGate, XGate, YGate, ZGate, SGate, TGate,
    CNOTGate, CZGate, CRGate, RYGate, GGate
)
from src.circuits import generate_qft_circuit, generate_qpe_circuit, generate_w_circuit
from src.frontend import circuit_dict_to_gates


# NumPy reference implementation
def apply_gate_numpy(state: np.ndarray, gate) -> np.ndarray:
    """Apply a gate to a state vector using NumPy.
    
    Uses the same convention as the Spark implementation:
    - 2-qubit gates use the reshaped (4,4) matrix form
    - Row/col indices are bit0 + 2*bit1 where bit0 is at qubit position q0
    """
    n_qubits = int(np.log2(len(state)))
    
    if gate.two_qubit_gate:
        q0, q1 = gate.qubits
        # Reshape tensor to 4x4 matrix (same as GateApplicator.register_gates)
        matrix = gate.tensor.reshape(4, 4)
        new_state = np.zeros_like(state)
        
        for idx in range(len(state)):
            if state[idx] == 0:
                continue
            
            # Extract bits at qubit positions
            bit0 = (idx >> q0) & 1
            bit1 = (idx >> q1) & 1
            col = bit0 + 2 * bit1
            
            for row in range(4):
                coeff = matrix[row, col]
                if coeff == 0:
                    continue
                
                new_bit0 = row & 1
                new_bit1 = (row >> 1) & 1
                
                # Clear old bits, set new bits
                new_idx = idx
                new_idx = (new_idx & ~(1 << q0)) | (new_bit0 << q0)
                new_idx = (new_idx & ~(1 << q1)) | (new_bit1 << q1)
                
                new_state[new_idx] += coeff * state[idx]
        return new_state
    else:
        qubit = gate.qubits[0]
        matrix = gate.tensor
        new_state = np.zeros_like(state)
        
        for idx in range(len(state)):
            if state[idx] == 0:
                continue
            current_bit = (idx >> qubit) & 1
            
            for new_bit in range(2):
                coeff = matrix[new_bit, current_bit]
                if coeff != 0:
                    new_idx = (idx & ~(1 << qubit)) | (new_bit << qubit)
                    new_state[new_idx] += coeff * state[idx]
        return new_state


def simulate_numpy(circuit_dict: dict) -> np.ndarray:
    """Simulate a circuit using NumPy reference implementation."""
    n_qubits, gates = circuit_dict_to_gates(circuit_dict)
    state = np.zeros(2**n_qubits, dtype=complex)
    state[0] = 1.0
    
    for gate in gates:
        state = apply_gate_numpy(state, gate)
    
    return state


@pytest.fixture
def config():
    """Create a test configuration with unique temporary directory per test."""
    import uuid
    temp_dir = Path(tempfile.mkdtemp())
    cfg = SimulatorConfig(
        run_id=f"test_non_stabilizer_{uuid.uuid4().hex[:8]}",
        base_path=temp_dir,
        batch_size=50,
        spark_master="local[2]",
        spark_shuffle_partitions=4,
    )
    cfg.ensure_paths()
    yield cfg
    shutil.rmtree(temp_dir, ignore_errors=True)


class TestTGateCircuits:
    """Test circuits containing T gates (the canonical non-Clifford gate)."""
    
    def test_single_t_gate(self, config):
        """T gate on |1⟩ should give e^(iπ/4)|1⟩."""
        circuit = {
            "number_of_qubits": 1,
            "gates": [
                {"qubits": [0], "gate": "X"},  # |0⟩ → |1⟩
                {"qubits": [0], "gate": "T"},  # |1⟩ → e^(iπ/4)|1⟩
            ]
        }
        
        expected = simulate_numpy(circuit)
        
        with SparkQuantumDriver(config) as driver:
            result = driver.run_circuit(circuit, resume=False)
            spark_state = driver.get_state_vector(result)
        
        np.testing.assert_allclose(spark_state, expected, atol=1e-10)
        
        # Verify T gate phase: e^(iπ/4) = (1+i)/√2
        expected_phase = np.exp(1j * np.pi / 4)
        assert np.isclose(spark_state[1], expected_phase, atol=1e-10)
    
    def test_t_t_equals_s(self, config):
        """T² = S (phase gate), a fundamental identity."""
        circuit_tt = {
            "number_of_qubits": 1,
            "gates": [
                {"qubits": [0], "gate": "X"},
                {"qubits": [0], "gate": "T"},
                {"qubits": [0], "gate": "T"},
            ]
        }
        circuit_s = {
            "number_of_qubits": 1,
            "gates": [
                {"qubits": [0], "gate": "X"},
                {"qubits": [0], "gate": "S"},
            ]
        }
        
        with SparkQuantumDriver(config) as driver:
            result_tt = driver.run_circuit(circuit_tt, resume=False)
            state_tt = driver.get_state_vector(result_tt)
        
        # Need fresh config for second run
        config2 = SimulatorConfig(
            run_id="test_non_stabilizer_2",
            base_path=config.base_path.parent / "test2",
            batch_size=50,
            spark_master="local[2]",
        )
        config2.ensure_paths()
        
        with SparkQuantumDriver(config2) as driver:
            result_s = driver.run_circuit(circuit_s, resume=False)
            state_s = driver.get_state_vector(result_s)
        
        np.testing.assert_allclose(state_tt, state_s, atol=1e-10)
        shutil.rmtree(config2.base_path, ignore_errors=True)
    
    def test_t_gate_superposition(self, config):
        """T gate on superposition: H|0⟩ then T."""
        circuit = {
            "number_of_qubits": 1,
            "gates": [
                {"qubits": [0], "gate": "H"},
                {"qubits": [0], "gate": "T"},
            ]
        }
        
        expected = simulate_numpy(circuit)
        
        with SparkQuantumDriver(config) as driver:
            result = driver.run_circuit(circuit, resume=False)
            spark_state = driver.get_state_vector(result)
        
        np.testing.assert_allclose(spark_state, expected, atol=1e-10)
        
        # Verify: (|0⟩ + e^(iπ/4)|1⟩)/√2
        sqrt2_inv = 1 / np.sqrt(2)
        assert np.isclose(spark_state[0], sqrt2_inv, atol=1e-10)
        assert np.isclose(spark_state[1], sqrt2_inv * np.exp(1j * np.pi / 4), atol=1e-10)
    
    def test_t_dagger_circuit(self, config):
        """T†T = I (T-dagger times T equals identity)."""
        # T† has phase e^(-iπ/4), so 7 T gates = T† (since T⁸ = I)
        circuit = {
            "number_of_qubits": 1,
            "gates": [
                {"qubits": [0], "gate": "X"},
                {"qubits": [0], "gate": "T"},
                {"qubits": [0], "gate": "T"},
                {"qubits": [0], "gate": "T"},
                {"qubits": [0], "gate": "T"},
                {"qubits": [0], "gate": "T"},
                {"qubits": [0], "gate": "T"},
                {"qubits": [0], "gate": "T"},
                {"qubits": [0], "gate": "T"},  # 8 T gates = identity
            ]
        }
        
        with SparkQuantumDriver(config) as driver:
            result = driver.run_circuit(circuit, resume=False)
            spark_state = driver.get_state_vector(result)
        
        # Should be back to |1⟩
        expected = np.array([0, 1], dtype=complex)
        np.testing.assert_allclose(spark_state, expected, atol=1e-10)


class TestQFTCircuits:
    """Test QFT circuits which use controlled-rotation (CR) gates."""
    
    @pytest.mark.parametrize("n_qubits", [2, 3, 4, 5])
    def test_qft_normalization(self, config, n_qubits):
        """QFT should preserve normalization."""
        circuit = generate_qft_circuit(n_qubits)
        
        with SparkQuantumDriver(config) as driver:
            result = driver.run_circuit(circuit, resume=False)
            spark_state = driver.get_state_vector(result)
        
        norm = np.linalg.norm(spark_state)
        assert np.isclose(norm, 1.0, atol=1e-10), f"Norm should be 1, got {norm}"
    
    @pytest.mark.parametrize("n_qubits", [2, 3, 4])
    def test_qft_matches_numpy(self, config, n_qubits):
        """QFT result should match NumPy reference."""
        circuit = generate_qft_circuit(n_qubits)
        
        expected = simulate_numpy(circuit)
        
        with SparkQuantumDriver(config) as driver:
            result = driver.run_circuit(circuit, resume=False)
            spark_state = driver.get_state_vector(result)
        
        np.testing.assert_allclose(spark_state, expected, atol=1e-10)
    
    def test_qft_on_basis_state(self, config):
        """QFT|0⟩ should give uniform superposition."""
        n_qubits = 3
        circuit = generate_qft_circuit(n_qubits)
        
        with SparkQuantumDriver(config) as driver:
            result = driver.run_circuit(circuit, resume=False)
            spark_state = driver.get_state_vector(result)
        
        # QFT|0⟩ = (1/√N) Σ|k⟩ (uniform superposition)
        expected_amp = 1 / np.sqrt(2**n_qubits)
        for amp in spark_state:
            assert np.isclose(np.abs(amp), expected_amp, atol=1e-10)


class TestWStateCircuits:
    """Test W-state circuits which use G gates (arbitrary rotations)."""
    
    @pytest.mark.parametrize("n_qubits", [3, 4, 5])
    def test_w_state_normalization(self, config, n_qubits):
        """W-state should have unit norm."""
        circuit = generate_w_circuit(n_qubits)
        
        with SparkQuantumDriver(config) as driver:
            result = driver.run_circuit(circuit, resume=False)
            spark_state = driver.get_state_vector(result)
        
        norm = np.linalg.norm(spark_state)
        assert np.isclose(norm, 1.0, atol=1e-10)
    
    @pytest.mark.parametrize("n_qubits", [3, 4])
    def test_w_state_structure(self, config, n_qubits):
        """W-state should be (|100...0⟩ + |010...0⟩ + ... + |000...1⟩)/√n."""
        circuit = generate_w_circuit(n_qubits)
        
        with SparkQuantumDriver(config) as driver:
            result = driver.run_circuit(circuit, resume=False)
            spark_state = driver.get_state_vector(result)
        
        # W-state has exactly n non-zero amplitudes (one 1 in each position)
        expected_amp = 1 / np.sqrt(n_qubits)
        
        non_zero_count = 0
        for idx, amp in enumerate(spark_state):
            # Check if this is a single-excitation basis state
            if bin(idx).count('1') == 1:
                assert np.isclose(np.abs(amp), expected_amp, atol=1e-8), \
                    f"W-state amplitude at idx={idx} should be ~{expected_amp}, got {amp}"
                non_zero_count += 1
            else:
                assert np.isclose(amp, 0, atol=1e-10), \
                    f"W-state amplitude at idx={idx} should be 0, got {amp}"
        
        assert non_zero_count == n_qubits


class TestArbitraryRotations:
    """Test circuits with arbitrary rotation angles (non-Clifford)."""
    
    @pytest.mark.parametrize("theta", [0.1, 0.5, 1.0, np.pi/7, np.pi/3])
    def test_ry_rotation(self, config, theta):
        """RY gate with arbitrary angle."""
        # Create circuit with explicit gate
        n_qubits = 1
        gates = [RYGate(0, theta)]
        
        # NumPy reference
        state = np.array([1, 0], dtype=complex)
        state = apply_gate_numpy(state, gates[0])
        expected = state
        
        # Spark execution
        from src.gate_applicator import GateApplicator
        from src.state_manager import StateManager
        from src.spark_session import get_or_create_spark_session
        
        spark = get_or_create_spark_session(config)
        state_manager = StateManager(spark, config)
        applicator = GateApplicator(spark, num_partitions=4)
        applicator.register_gates(gates)
        
        state_df = state_manager.initialize_state(n_qubits)
        state_df = applicator.apply_gate(state_df, gates[0])
        
        spark_state = state_manager.get_state_as_array(state_df, n_qubits)
        applicator.cleanup()
        spark.stop()
        
        np.testing.assert_allclose(spark_state, expected, atol=1e-10)
        
        # Verify RY matrix structure
        assert np.isclose(spark_state[0], np.cos(theta/2), atol=1e-10)
        assert np.isclose(spark_state[1], np.sin(theta/2), atol=1e-10)
    
    @pytest.mark.parametrize("k", [2, 3, 4, 5])
    def test_cr_gate(self, config, k):
        """Controlled-rotation with different k values."""
        circuit = {
            "number_of_qubits": 2,
            "gates": [
                {"qubits": [0], "gate": "X"},  # Control = |1⟩
                {"qubits": [1], "gate": "X"},  # Target = |1⟩
                {"qubits": [0, 1], "gate": "CR", "params": {"k": k}},
            ]
        }
        
        expected = simulate_numpy(circuit)
        
        with SparkQuantumDriver(config) as driver:
            result = driver.run_circuit(circuit, resume=False)
            spark_state = driver.get_state_vector(result)
        
        np.testing.assert_allclose(spark_state, expected, atol=1e-10)
        
        # CR applies phase e^(2πi/2^k) when both qubits are |1⟩
        # Final state is |11⟩ with phase
        expected_phase = np.exp(2j * np.pi / (2**k))
        assert np.isclose(spark_state[3], expected_phase, atol=1e-10)


class TestMixedCircuits:
    """Test circuits mixing Clifford and non-Clifford gates."""
    
    def test_clifford_then_t(self, config):
        """Clifford circuit followed by T gates."""
        circuit = {
            "number_of_qubits": 2,
            "gates": [
                # Clifford part: create Bell state
                {"qubits": [0], "gate": "H"},
                {"qubits": [0, 1], "gate": "CNOT"},
                # Non-Clifford: apply T to both qubits
                {"qubits": [0], "gate": "T"},
                {"qubits": [1], "gate": "T"},
            ]
        }
        
        expected = simulate_numpy(circuit)
        
        with SparkQuantumDriver(config) as driver:
            result = driver.run_circuit(circuit, resume=False)
            spark_state = driver.get_state_vector(result)
        
        np.testing.assert_allclose(spark_state, expected, atol=1e-10)
    
    def test_deep_non_stabilizer_circuit(self, config):
        """Deep circuit alternating between Clifford and T gates."""
        gates = []
        n_qubits = 3
        
        for layer in range(5):
            # Clifford layer
            for q in range(n_qubits):
                gates.append({"qubits": [q], "gate": "H"})
            for q in range(n_qubits - 1):
                gates.append({"qubits": [q, q+1], "gate": "CNOT"})
            
            # Non-Clifford layer
            for q in range(n_qubits):
                gates.append({"qubits": [q], "gate": "T"})
        
        circuit = {"number_of_qubits": n_qubits, "gates": gates}
        
        expected = simulate_numpy(circuit)
        
        with SparkQuantumDriver(config) as driver:
            result = driver.run_circuit(circuit, resume=False)
            spark_state = driver.get_state_vector(result)
        
        np.testing.assert_allclose(spark_state, expected, atol=1e-9)
        
        # Verify it's a dense state (non-stabilizer circuits typically give dense states)
        non_zero = np.sum(np.abs(spark_state) > 1e-10)
        assert non_zero > n_qubits, f"Expected dense state, got only {non_zero} non-zero amplitudes"


class TestMagicStatePreparation:
    """Test magic state preparation (key for fault-tolerant quantum computing)."""
    
    def test_t_magic_state(self, config):
        """Prepare the T-magic state |T⟩ = (|0⟩ + e^(iπ/4)|1⟩)/√2."""
        circuit = {
            "number_of_qubits": 1,
            "gates": [
                {"qubits": [0], "gate": "H"},
                {"qubits": [0], "gate": "T"},
            ]
        }
        
        with SparkQuantumDriver(config) as driver:
            result = driver.run_circuit(circuit, resume=False)
            spark_state = driver.get_state_vector(result)
        
        # |T⟩ = (|0⟩ + e^(iπ/4)|1⟩)/√2
        sqrt2_inv = 1 / np.sqrt(2)
        expected = np.array([sqrt2_inv, sqrt2_inv * np.exp(1j * np.pi / 4)], dtype=complex)
        
        np.testing.assert_allclose(spark_state, expected, atol=1e-10)
    
    def test_multiple_magic_states(self, config):
        """Prepare multiple magic states in parallel."""
        n_qubits = 3
        circuit = {
            "number_of_qubits": n_qubits,
            "gates": []
        }
        for q in range(n_qubits):
            circuit["gates"].append({"qubits": [q], "gate": "H"})
            circuit["gates"].append({"qubits": [q], "gate": "T"})
        
        with SparkQuantumDriver(config) as driver:
            result = driver.run_circuit(circuit, resume=False)
            spark_state = driver.get_state_vector(result)
        
        expected = simulate_numpy(circuit)
        np.testing.assert_allclose(spark_state, expected, atol=1e-10)
        
        # Verify all amplitudes have equal magnitude
        expected_mag = 1 / np.sqrt(2**n_qubits)
        for amp in spark_state:
            assert np.isclose(np.abs(amp), expected_mag, atol=1e-10)
