"""
Critical verification tests for the Spark quantum simulator.

These tests are designed to catch subtle bugs in:
1. Index computation (XOR vs mask approach)
2. Complex number arithmetic
3. Sparse state handling
4. Recovery logic
5. Off-by-one errors
6. Gate matrix correctness for all gate types
"""
from __future__ import annotations

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.gates import (
    HadamardGate, XGate, YGate, ZGate, SGate, TGate,
    CNOTGate, CZGate, CYGate, CRGate, SWAPGate, RYGate, GGate
)
from src.circuits import generate_ghz_circuit, generate_qft_circuit, generate_w_circuit
from src.frontend import circuit_dict_to_gates
from tests.test_unit import apply_gate_numpy, simulate_circuit_numpy


# =============================================================================
# Index Computation Verification: XOR vs MASK
# =============================================================================

class TestIndexComputationParity:
    """
    Verify that the XOR approach used in Spark produces 
    identical results to the mask approach used in v1.
    """
    
    def test_one_qubit_index_computation_parity(self):
        """
        Compare: 
        - v1: (idx & ~(1 << q)) | (row << q)
        - v2: (idx ^ (qubit_bit << q)) | (row << q)
        
        These should be equivalent.
        """
        for n_qubits in [1, 2, 3, 4, 5]:
            for idx in range(2 ** n_qubits):
                for qubit in range(n_qubits):
                    qubit_bit = (idx >> qubit) & 1
                    
                    for row in [0, 1]:
                        # v1 approach (mask and OR)
                        v1_result = (idx & ~(1 << qubit)) | (row << qubit)
                        
                        # v2 approach (XOR then OR)
                        v2_result = (idx ^ (qubit_bit << qubit)) | (row << qubit)
                        
                        assert v1_result == v2_result, \
                            f"Mismatch: idx={idx}, qubit={qubit}, row={row}"
    
    def test_two_qubit_index_computation_parity(self):
        """
        Compare 2-qubit gate index computation between v1 and v2 approaches.
        """
        for n_qubits in [2, 3, 4, 5]:
            for idx in range(2 ** n_qubits):
                for q0 in range(n_qubits):
                    for q1 in range(n_qubits):
                        if q0 == q1:
                            continue
                        
                        bit0 = (idx >> q0) & 1
                        bit1 = (idx >> q1) & 1
                        
                        for row in range(4):
                            new_bit0 = row & 1
                            new_bit1 = (row >> 1) & 1
                            
                            # v1 approach
                            v1_result = (
                                (idx & ~((1 << q0) | (1 << q1)))
                                | (new_bit0 << q0)
                                | (new_bit1 << q1)
                            )
                            
                            # v2 approach
                            cleared = idx ^ (bit0 << q0) ^ (bit1 << q1)
                            v2_result = cleared | (new_bit0 << q0) | (new_bit1 << q1)
                            
                            assert v1_result == v2_result, \
                                f"Mismatch: idx={idx}, q0={q0}, q1={q1}, row={row}"


# =============================================================================
# Complex Number Arithmetic Verification
# =============================================================================

class TestComplexArithmetic:
    """Test that complex number operations are correct."""
    
    def test_complex_multiplication_formula(self):
        """
        Verify: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
        """
        for _ in range(100):
            a, b = np.random.randn(2)
            c, d = np.random.randn(2)
            
            # Python complex
            py_result = complex(a, b) * complex(c, d)
            
            # Our formula
            real = a * c - b * d
            imag = a * d + b * c
            our_result = complex(real, imag)
            
            np.testing.assert_allclose(our_result.real, py_result.real, atol=1e-10)
            np.testing.assert_allclose(our_result.imag, py_result.imag, atol=1e-10)
    
    def test_gate_with_complex_phases(self):
        """Test gates with complex phases (S, T, Y gates)."""
        # S gate: [[1, 0], [0, i]]
        state = np.array([0.6, 0.8], dtype=complex)
        gate = SGate(0)
        result = apply_gate_numpy(state, gate)
        
        expected = np.array([0.6, 0.8j], dtype=complex)
        np.testing.assert_allclose(result, expected, atol=1e-10)
    
    def test_t_gate_phase(self):
        """T gate should have exp(i*pi/4) phase."""
        state = np.array([0, 1], dtype=complex)
        gate = TGate(0)
        result = apply_gate_numpy(state, gate)
        
        # exp(i*pi/4) = (1 + i) / sqrt(2)
        expected_phase = np.exp(1j * np.pi / 4)
        expected = np.array([0, expected_phase], dtype=complex)
        np.testing.assert_allclose(result, expected, atol=1e-10)
    
    def test_y_gate_imaginary(self):
        """Y gate: [[0, -i], [i, 0]]."""
        state = np.array([1, 0], dtype=complex)
        gate = YGate(0)
        result = apply_gate_numpy(state, gate)
        
        # Y|0⟩ = i|1⟩
        expected = np.array([0, 1j], dtype=complex)
        np.testing.assert_allclose(result, expected, atol=1e-10)


# =============================================================================
# All Gate Types Verification
# =============================================================================

class TestAllGateTypes:
    """Test that all gate types work correctly."""
    
    @pytest.mark.parametrize("gate_class,expected_name", [
        (lambda: HadamardGate(0), "H"),
        (lambda: XGate(0), "X"),
        (lambda: YGate(0), "Y"),
        (lambda: ZGate(0), "Z"),
        (lambda: SGate(0), "S"),
        (lambda: TGate(0), "T"),
    ])
    def test_single_qubit_gates(self, gate_class, expected_name):
        """Test each 1-qubit gate type."""
        gate = gate_class()
        assert gate.gate_name == expected_name
        assert not gate.two_qubit_gate
        
        # Apply to |0⟩
        state = np.array([1, 0], dtype=complex)
        result = apply_gate_numpy(state, gate)
        
        # Check normalization
        norm = np.linalg.norm(result)
        np.testing.assert_allclose(norm, 1.0, atol=1e-10)
    
    @pytest.mark.parametrize("gate_class,expected_name", [
        (lambda: CNOTGate(0, 1), "CNOT"),
        (lambda: CZGate(0, 1), "CZ"),
        (lambda: CYGate(0, 1), "CY"),
        (lambda: SWAPGate(0, 1), "SWAP"),
    ])
    def test_two_qubit_gates(self, gate_class, expected_name):
        """Test each 2-qubit gate type."""
        gate = gate_class()
        assert gate.gate_name == expected_name
        assert gate.two_qubit_gate
        
        # Apply to random state
        np.random.seed(42)
        state = np.random.randn(4) + 1j * np.random.randn(4)
        state /= np.linalg.norm(state)
        
        result = apply_gate_numpy(state, gate)
        
        # Check normalization
        norm = np.linalg.norm(result)
        np.testing.assert_allclose(norm, 1.0, atol=1e-10)
    
    def test_cr_gate_parametric(self):
        """Test parametric CR gate."""
        for k in [2, 3, 4, 5]:
            gate = CRGate(0, 1, k)
            assert gate.gate_name == f"CR{k}"
            
            # CR|11⟩ should add phase exp(2πi/2^k)
            state = np.zeros(4, dtype=complex)
            state[0b11] = 1.0
            
            result = apply_gate_numpy(state, gate)
            expected_phase = np.exp(2j * np.pi / (2 ** k))
            
            np.testing.assert_allclose(result[0b11], expected_phase, atol=1e-10)
    
    def test_ry_gate_parametric(self):
        """Test parametric RY gate."""
        for theta in [np.pi/4, np.pi/2, np.pi, 3*np.pi/2]:
            gate = RYGate(0, theta)
            
            state = np.array([1, 0], dtype=complex)
            result = apply_gate_numpy(state, gate)
            
            # RY(θ)|0⟩ = cos(θ/2)|0⟩ + sin(θ/2)|1⟩
            expected = np.array([np.cos(theta/2), np.sin(theta/2)], dtype=complex)
            np.testing.assert_allclose(result, expected, atol=1e-10)


# =============================================================================
# Sparse State Handling
# =============================================================================

class TestSparseStateHandling:
    """Test that sparse state representation is handled correctly."""
    
    def test_interference_to_zero(self):
        """Test when amplitudes cancel to exactly zero."""
        # H|+⟩ = H * H|0⟩ = |0⟩
        # But intermediately we have a superposition
        state = np.zeros(2, dtype=complex)
        state[0] = 1.0
        
        h = HadamardGate(0)
        
        # After first H: equal superposition
        state = apply_gate_numpy(state, h)
        assert np.abs(state[0]) > 1e-10
        assert np.abs(state[1]) > 1e-10
        
        # After second H: back to |0⟩
        state = apply_gate_numpy(state, h)
        np.testing.assert_allclose(state[0], 1.0, atol=1e-10)
        np.testing.assert_allclose(state[1], 0.0, atol=1e-10)
    
    def test_sparse_to_dense_and_back(self):
        """Test state that goes from sparse to dense and back."""
        # Start with |0⟩ (sparse)
        circuit = {
            "number_of_qubits": 3,
            "gates": [
                {"qubits": [0], "gate": "H"},  # Creates superposition
                {"qubits": [1], "gate": "H"},  # More superposition
                {"qubits": [2], "gate": "H"},  # Full superposition (all 8 states)
                {"qubits": [2], "gate": "H"},  # Back to 4 states
                {"qubits": [1], "gate": "H"},  # Back to 2 states
                {"qubits": [0], "gate": "H"},  # Back to |0⟩
            ]
        }
        
        state = simulate_circuit_numpy(circuit)
        
        # Should be back to |000⟩
        expected = np.zeros(8, dtype=complex)
        expected[0] = 1.0
        np.testing.assert_allclose(state, expected, atol=1e-10)
    
    def test_destructive_interference_ghz_inverse(self):
        """Test GHZ circuit and its inverse (should return to |0⟩)."""
        n_qubits = 4
        
        # Forward GHZ
        circuit = generate_ghz_circuit(n_qubits, reverse=False)
        
        # Add reversed GHZ (inverse)
        reversed_circuit = generate_ghz_circuit(n_qubits, reverse=True)
        circuit["gates"].extend(reversed_circuit["gates"])
        
        state = simulate_circuit_numpy(circuit)
        
        # Should be back to |0000⟩
        expected = np.zeros(2 ** n_qubits, dtype=complex)
        expected[0] = 1.0
        np.testing.assert_allclose(state, expected, atol=1e-10)


# =============================================================================
# Off-by-one Error Detection
# =============================================================================

class TestOffByOneErrors:
    """Tests designed to catch off-by-one errors."""
    
    def test_qubit_indexing_edge_cases(self):
        """Test gates on qubit 0 and highest qubit."""
        for n_qubits in [2, 3, 4, 5]:
            state = np.zeros(2 ** n_qubits, dtype=complex)
            state[0] = 1.0
            
            # Apply H on qubit 0
            state = apply_gate_numpy(state, HadamardGate(0))
            
            # Verify qubit 0 is in superposition
            assert np.abs(state[0]) > 0.5
            assert np.abs(state[1]) > 0.5
            
            # Reset
            state = np.zeros(2 ** n_qubits, dtype=complex)
            state[0] = 1.0
            
            # Apply H on highest qubit
            highest = n_qubits - 1
            state = apply_gate_numpy(state, HadamardGate(highest))
            
            # Verify highest qubit is in superposition
            assert np.abs(state[0]) > 0.5
            assert np.abs(state[1 << highest]) > 0.5
    
    def test_batch_boundary_gates(self):
        """Test that gates at batch boundaries are handled correctly."""
        # Create circuit with exactly batch_size gates
        from src.gate_batcher import GateBatcher
        
        for batch_size in [1, 2, 3, 5, 10]:
            batcher = GateBatcher(batch_size=batch_size)
            
            # Create gates that span exactly 2 batches
            n_gates = batch_size * 2
            gates = [HadamardGate(i % 3) for i in range(n_gates)]
            
            batches = batcher.create_batches(gates)
            
            assert len(batches) == 2
            assert batches[0].end_seq == batch_size
            assert batches[1].start_seq == batch_size
            
            # Verify all gates are accounted for
            total_gates = sum(b.size for b in batches)
            assert total_gates == n_gates
    
    def test_gate_sequence_numbers(self):
        """Test that gate sequence numbers are correct."""
        circuit = generate_ghz_circuit(5)
        n_qubits, gates = circuit_dict_to_gates(circuit)
        
        # Gates should be: H, CNOT, CNOT, CNOT, CNOT
        assert len(gates) == 5
        
        # Verify gate types in correct order
        assert gates[0].gate_name == "H"
        for i in range(1, 5):
            assert gates[i].gate_name == "CNOT"


# =============================================================================
# Circuit Correctness (comparing with known results)
# =============================================================================

class TestKnownCircuitResults:
    """Test circuits with analytically known results."""
    
    def test_bell_state_exact(self):
        """Bell state should be exactly (|00⟩ + |11⟩)/√2."""
        circuit = generate_ghz_circuit(2)
        state = simulate_circuit_numpy(circuit)
        
        sqrt2_inv = 1 / np.sqrt(2)
        
        # Exact values
        np.testing.assert_allclose(state[0b00], sqrt2_inv, atol=1e-15)
        np.testing.assert_allclose(state[0b11], sqrt2_inv, atol=1e-15)
        np.testing.assert_allclose(state[0b01], 0.0, atol=1e-15)
        np.testing.assert_allclose(state[0b10], 0.0, atol=1e-15)
    
    def test_qft_on_computational_basis(self):
        """QFT|j⟩ should have equal amplitudes with specific phases."""
        n = 2
        
        # QFT|0⟩ = uniform superposition with no phases
        circuit = generate_qft_circuit(n)
        state = simulate_circuit_numpy(circuit)
        
        # All amplitudes should have equal magnitude
        expected_mag = 1 / 2  # 1/√4 = 0.5
        for amp in state:
            np.testing.assert_allclose(np.abs(amp), expected_mag, atol=1e-10)
    
    def test_x_gate_effect(self):
        """X|0⟩ = |1⟩, X|1⟩ = |0⟩."""
        # X|0⟩ = |1⟩
        state0 = np.array([1, 0], dtype=complex)
        result0 = apply_gate_numpy(state0, XGate(0))
        np.testing.assert_allclose(result0, [0, 1], atol=1e-15)
        
        # X|1⟩ = |0⟩
        state1 = np.array([0, 1], dtype=complex)
        result1 = apply_gate_numpy(state1, XGate(0))
        np.testing.assert_allclose(result1, [1, 0], atol=1e-15)
    
    def test_cnot_truth_table(self):
        """Test all 4 cases of CNOT."""
        test_cases = [
            # (input_idx, expected_output_idx)
            (0b00, 0b00),  # |00⟩ → |00⟩ (control=0, no flip)
            (0b01, 0b11),  # |01⟩ → |11⟩ (control=1, flip target)
            (0b10, 0b10),  # |10⟩ → |10⟩ (control=0, no flip)
            (0b11, 0b01),  # |11⟩ → |01⟩ (control=1, flip target)
        ]
        
        gate = CNOTGate(0, 1)  # Control: qubit 0, Target: qubit 1
        
        for input_idx, expected_idx in test_cases:
            state = np.zeros(4, dtype=complex)
            state[input_idx] = 1.0
            
            result = apply_gate_numpy(state, gate)
            
            assert np.abs(result[expected_idx]) > 0.99, \
                f"CNOT failed: |{input_idx:02b}⟩ should go to |{expected_idx:02b}⟩"


# =============================================================================
# V1 vs V2 Direct Parity
# =============================================================================

class TestV1V2Parity:
    """Direct parity tests comparing v2 Pandas logic with NumPy reference."""
    
    @pytest.mark.parametrize("n_qubits", [2, 3, 4, 5, 6])
    def test_ghz_parity_extended(self, n_qubits):
        """Extended GHZ parity test."""
        from tests.test_spark_logic_verification import (
            simulate_circuit_pandas, pandas_state_to_numpy
        )
        
        circuit = generate_ghz_circuit(n_qubits)
        
        # NumPy reference
        numpy_state = simulate_circuit_numpy(circuit)
        
        # Pandas/Spark logic
        pandas_df = simulate_circuit_pandas(circuit)
        pandas_state = pandas_state_to_numpy(pandas_df, n_qubits)
        
        np.testing.assert_allclose(pandas_state, numpy_state, atol=1e-10)
    
    @pytest.mark.parametrize("n_qubits", [3, 4, 5])
    def test_w_state_parity(self, n_qubits):
        """W-state parity test."""
        from tests.test_spark_logic_verification import (
            simulate_circuit_pandas, pandas_state_to_numpy
        )
        
        circuit = generate_w_circuit(n_qubits)
        
        numpy_state = simulate_circuit_numpy(circuit)
        pandas_df = simulate_circuit_pandas(circuit)
        pandas_state = pandas_state_to_numpy(pandas_df, circuit["number_of_qubits"])
        
        np.testing.assert_allclose(pandas_state, numpy_state, atol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
