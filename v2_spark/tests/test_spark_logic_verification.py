"""
Verification tests for Spark DataFrame gate application logic.

Since Spark requires Java, we verify the LOGIC by implementing
the exact same transformations in pure Python/pandas and comparing
with our reference NumPy implementation.

This ensures the DataFrame operations are mathematically correct.
"""
from __future__ import annotations

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.gates import (
    Gate, HadamardGate, XGate, YGate, ZGate,
    CNOTGate, CZGate, CRGate, SWAPGate
)
from src.circuits import generate_ghz_circuit, generate_qft_circuit
from src.frontend import circuit_dict_to_gates
from tests.test_unit import apply_gate_numpy, simulate_circuit_numpy


# =============================================================================
# Pandas-based simulation mimicking Spark DataFrame logic
# =============================================================================

def register_gates_pandas(gates: List[Gate]) -> pd.DataFrame:
    """
    Create gate matrix DataFrame in Pandas (mirrors Spark logic).
    """
    seen = set()
    rows = []
    
    for gate in gates:
        arity = 2 if gate.two_qubit_gate else 1
        key = (gate.gate_name, arity)
        
        if key in seen:
            continue
        seen.add(key)
        
        if gate.two_qubit_gate:
            tensor = gate.tensor.reshape(4, 4)
            for row in range(4):
                for col in range(4):
                    val = tensor[row, col]
                    rows.append({
                        'gate_name': gate.gate_name,
                        'arity': 2,
                        'row': row,
                        'col': col,
                        'real': float(np.real(val)),
                        'imag': float(np.imag(val)),
                    })
        else:
            tensor = gate.tensor
            for row in range(2):
                for col in range(2):
                    val = tensor[row, col]
                    rows.append({
                        'gate_name': gate.gate_name,
                        'arity': 1,
                        'row': row,
                        'col': col,
                        'real': float(np.real(val)),
                        'imag': float(np.imag(val)),
                    })
    
    return pd.DataFrame(rows)


def apply_one_qubit_gate_pandas(
    state_df: pd.DataFrame,
    gate_matrix_df: pd.DataFrame,
    gate_name: str,
    qubit: int
) -> pd.DataFrame:
    """
    Apply 1-qubit gate using Pandas operations that mirror Spark logic.
    
    This implements the EXACT same algorithm as GateApplicator.apply_one_qubit_gate
    """
    # Filter gate matrix for this gate
    gm = gate_matrix_df[
        (gate_matrix_df['gate_name'] == gate_name) & 
        (gate_matrix_df['arity'] == 1)
    ][['row', 'col', 'real', 'imag']].copy()
    gm.columns = ['g_row', 'g_col', 'g_real', 'g_imag']
    
    # Add qubit_bit column: (idx >> qubit) & 1
    # Use integer division and modulo for pandas compatibility
    state_df = state_df.copy()
    state_df['qubit_bit'] = (state_df['idx'] // (1 << qubit)) % 2
    
    # Join on qubit_bit == g_col
    merged = state_df.merge(gm, left_on='qubit_bit', right_on='g_col')
    
    # Compute new_idx using XOR approach (same as Spark implementation)
    # XOR clears the bit, then OR sets the new bit
    # Use numpy operations for bitwise ops on pandas columns
    merged['new_idx'] = (
        np.bitwise_xor(merged['idx'].values.astype(int), 
                       (merged['qubit_bit'].values.astype(int) * (1 << qubit))) |
        (merged['g_row'].values.astype(int) * (1 << qubit))
    )
    
    # Complex multiplication: (a+bi)(c+di) = (ac-bd) + (ad+bc)i
    merged['new_real'] = (
        merged['g_real'] * merged['real'] - 
        merged['g_imag'] * merged['imag']
    )
    merged['new_imag'] = (
        merged['g_real'] * merged['imag'] + 
        merged['g_imag'] * merged['real']
    )
    
    # Group by new_idx and sum
    result = merged.groupby('new_idx').agg({
        'new_real': 'sum',
        'new_imag': 'sum'
    }).reset_index()
    
    result.columns = ['idx', 'real', 'imag']
    
    # Filter out zeros
    result = result[
        (result['real'].abs() > 1e-15) | 
        (result['imag'].abs() > 1e-15)
    ].copy()
    
    return result


def apply_two_qubit_gate_pandas(
    state_df: pd.DataFrame,
    gate_matrix_df: pd.DataFrame,
    gate_name: str,
    qubit0: int,
    qubit1: int
) -> pd.DataFrame:
    """
    Apply 2-qubit gate using Pandas operations that mirror Spark logic.
    """
    # Filter gate matrix for this gate
    gm = gate_matrix_df[
        (gate_matrix_df['gate_name'] == gate_name) & 
        (gate_matrix_df['arity'] == 2)
    ][['row', 'col', 'real', 'imag']].copy()
    gm.columns = ['g_row', 'g_col', 'g_real', 'g_imag']
    
    # Add qubit bit columns using division/modulo for pandas compatibility
    state_df = state_df.copy()
    state_df['bit0'] = (state_df['idx'] // (1 << qubit0)) % 2
    state_df['bit1'] = (state_df['idx'] // (1 << qubit1)) % 2
    state_df['qubit_bits'] = state_df['bit0'] + state_df['bit1'] * 2
    
    # Join on qubit_bits == g_col
    merged = state_df.merge(gm, left_on='qubit_bits', right_on='g_col')
    
    # Compute new_idx using XOR approach with numpy for bitwise ops
    idx_arr = merged['idx'].values.astype(int)
    bit0_arr = merged['bit0'].values.astype(int)
    bit1_arr = merged['bit1'].values.astype(int)
    g_row_arr = merged['g_row'].values.astype(int)
    
    # XOR to clear both bits, then OR to set new values
    cleared = np.bitwise_xor(
        np.bitwise_xor(idx_arr, bit0_arr * (1 << qubit0)),
        bit1_arr * (1 << qubit1)
    )
    new_bit0 = g_row_arr & 1
    new_bit1 = (g_row_arr >> 1) & 1
    merged['new_idx'] = cleared | (new_bit0 * (1 << qubit0)) | (new_bit1 * (1 << qubit1))
    
    # Complex multiplication
    merged['new_real'] = (
        merged['g_real'] * merged['real'] - 
        merged['g_imag'] * merged['imag']
    )
    merged['new_imag'] = (
        merged['g_real'] * merged['imag'] + 
        merged['g_imag'] * merged['real']
    )
    
    # Group by new_idx and sum
    result = merged.groupby('new_idx').agg({
        'new_real': 'sum',
        'new_imag': 'sum'
    }).reset_index()
    
    result.columns = ['idx', 'real', 'imag']
    
    # Filter out zeros
    result = result[
        (result['real'].abs() > 1e-15) | 
        (result['imag'].abs() > 1e-15)
    ].copy()
    
    return result


def apply_gate_pandas(
    state_df: pd.DataFrame,
    gate_matrix_df: pd.DataFrame,
    gate: Gate
) -> pd.DataFrame:
    """Apply a gate using Pandas."""
    if gate.two_qubit_gate:
        q0, q1 = gate.qubits
        return apply_two_qubit_gate_pandas(
            state_df, gate_matrix_df, gate.gate_name, q0, q1
        )
    else:
        (q,) = gate.qubits
        return apply_one_qubit_gate_pandas(
            state_df, gate_matrix_df, gate.gate_name, q
        )


def simulate_circuit_pandas(circuit_dict: Dict) -> pd.DataFrame:
    """
    Simulate circuit using Pandas DataFrame operations.
    
    This mirrors the Spark implementation logic exactly.
    """
    n_qubits, gates = circuit_dict_to_gates(circuit_dict)
    
    # Initialize state |0...0⟩
    state_df = pd.DataFrame([{'idx': 0, 'real': 1.0, 'imag': 0.0}])
    
    # Register gate matrices
    gate_matrix_df = register_gates_pandas(gates)
    
    # Apply all gates
    for gate in gates:
        state_df = apply_gate_pandas(state_df, gate_matrix_df, gate)
    
    return state_df


def pandas_state_to_numpy(state_df: pd.DataFrame, n_qubits: int) -> np.ndarray:
    """Convert pandas state DataFrame to numpy array."""
    size = 2 ** n_qubits
    arr = np.zeros(size, dtype=complex)
    
    for _, row in state_df.iterrows():
        idx = int(row['idx'])
        if idx < size:
            arr[idx] = complex(row['real'], row['imag'])
    
    return arr


# =============================================================================
# Verification Tests
# =============================================================================

class TestSparkLogicVerification:
    """
    Test that our Pandas implementation (mirroring Spark logic)
    matches the reference NumPy implementation.
    """
    
    def test_hadamard_single_qubit(self):
        """Verify H gate application."""
        circuit = {
            "number_of_qubits": 1,
            "gates": [{"qubits": [0], "gate": "H"}]
        }
        
        # NumPy reference
        numpy_state = simulate_circuit_numpy(circuit)
        
        # Pandas/Spark logic
        pandas_df = simulate_circuit_pandas(circuit)
        pandas_state = pandas_state_to_numpy(pandas_df, 1)
        
        np.testing.assert_allclose(pandas_state, numpy_state, atol=1e-10)
    
    def test_x_gate(self):
        """Verify X gate application."""
        circuit = {
            "number_of_qubits": 2,
            "gates": [{"qubits": [1], "gate": "X"}]
        }
        
        numpy_state = simulate_circuit_numpy(circuit)
        pandas_df = simulate_circuit_pandas(circuit)
        pandas_state = pandas_state_to_numpy(pandas_df, 2)
        
        np.testing.assert_allclose(pandas_state, numpy_state, atol=1e-10)
    
    def test_cnot_gate(self):
        """Verify CNOT gate application."""
        circuit = {
            "number_of_qubits": 2,
            "gates": [
                {"qubits": [0], "gate": "H"},
                {"qubits": [0, 1], "gate": "CNOT"}
            ]
        }
        
        numpy_state = simulate_circuit_numpy(circuit)
        pandas_df = simulate_circuit_pandas(circuit)
        pandas_state = pandas_state_to_numpy(pandas_df, 2)
        
        np.testing.assert_allclose(pandas_state, numpy_state, atol=1e-10)
    
    @pytest.mark.parametrize("n_qubits", [2, 3, 4, 5])
    def test_ghz_circuit(self, n_qubits):
        """Verify GHZ circuit matches reference."""
        circuit = generate_ghz_circuit(n_qubits)
        
        numpy_state = simulate_circuit_numpy(circuit)
        pandas_df = simulate_circuit_pandas(circuit)
        pandas_state = pandas_state_to_numpy(pandas_df, n_qubits)
        
        np.testing.assert_allclose(pandas_state, numpy_state, atol=1e-10)
    
    @pytest.mark.parametrize("n_qubits", [2, 3, 4])
    def test_qft_circuit(self, n_qubits):
        """Verify QFT circuit matches reference."""
        circuit = generate_qft_circuit(n_qubits)
        
        numpy_state = simulate_circuit_numpy(circuit)
        pandas_df = simulate_circuit_pandas(circuit)
        pandas_state = pandas_state_to_numpy(pandas_df, n_qubits)
        
        np.testing.assert_allclose(pandas_state, numpy_state, atol=1e-10)
    
    def test_multiple_gates_sequence(self):
        """Verify sequence of multiple gates."""
        circuit = {
            "number_of_qubits": 3,
            "gates": [
                {"qubits": [0], "gate": "H"},
                {"qubits": [1], "gate": "H"},
                {"qubits": [2], "gate": "X"},
                {"qubits": [0, 1], "gate": "CNOT"},
                {"qubits": [1, 2], "gate": "CNOT"},
                {"qubits": [0], "gate": "H"},
            ]
        }
        
        numpy_state = simulate_circuit_numpy(circuit)
        pandas_df = simulate_circuit_pandas(circuit)
        pandas_state = pandas_state_to_numpy(pandas_df, 3)
        
        np.testing.assert_allclose(pandas_state, numpy_state, atol=1e-10)
    
    def test_non_adjacent_qubits(self):
        """Verify 2-qubit gates on non-adjacent qubits."""
        circuit = {
            "number_of_qubits": 4,
            "gates": [
                {"qubits": [0], "gate": "H"},
                {"qubits": [0, 3], "gate": "CNOT"},  # Non-adjacent
                {"qubits": [1, 2], "gate": "CZ"},
            ]
        }
        
        numpy_state = simulate_circuit_numpy(circuit)
        pandas_df = simulate_circuit_pandas(circuit)
        pandas_state = pandas_state_to_numpy(pandas_df, 4)
        
        np.testing.assert_allclose(pandas_state, numpy_state, atol=1e-10)
    
    def test_reversed_qubit_order(self):
        """Verify CNOT with control > target."""
        circuit = {
            "number_of_qubits": 3,
            "gates": [
                {"qubits": [2], "gate": "H"},
                {"qubits": [2, 0], "gate": "CNOT"},  # Control > target
            ]
        }
        
        numpy_state = simulate_circuit_numpy(circuit)
        pandas_df = simulate_circuit_pandas(circuit)
        pandas_state = pandas_state_to_numpy(pandas_df, 3)
        
        np.testing.assert_allclose(pandas_state, numpy_state, atol=1e-10)


class TestIndexManipulation:
    """
    Detailed tests for the index manipulation logic used in Spark.
    """
    
    def test_xor_clear_bit_logic(self):
        """
        Verify the XOR approach for clearing bits works correctly.
        
        The Spark code uses: idx ^ (qubit_bit << qubit)
        This XORs the current bit value, effectively clearing it.
        """
        for idx in range(16):
            for qubit in range(4):
                qubit_bit = (idx >> qubit) & 1
                
                # Clear using XOR
                cleared = idx ^ (qubit_bit << qubit)
                
                # Verify the bit is now 0
                assert (cleared >> qubit) & 1 == 0
                
                # Verify other bits unchanged
                for other in range(4):
                    if other != qubit:
                        assert (cleared >> other) & 1 == (idx >> other) & 1
    
    def test_xor_clear_then_or_set(self):
        """
        Verify clearing with XOR then setting with OR produces correct results.
        """
        for idx in range(16):
            for qubit in range(4):
                qubit_bit = (idx >> qubit) & 1
                
                for new_bit in [0, 1]:
                    # Our logic: XOR to clear, OR to set
                    result = (idx ^ (qubit_bit << qubit)) | (new_bit << qubit)
                    
                    # Alternative (reference): mask and set
                    reference = (idx & ~(1 << qubit)) | (new_bit << qubit)
                    
                    assert result == reference
    
    def test_two_qubit_xor_logic(self):
        """
        Verify the XOR approach works for 2-qubit operations.
        """
        for idx in range(32):
            for q0 in range(4):
                for q1 in range(q0 + 1, 5):
                    bit0 = (idx >> q0) & 1
                    bit1 = (idx >> q1) & 1
                    
                    # Clear both bits using double XOR
                    cleared = idx ^ (bit0 << q0) ^ (bit1 << q1)
                    
                    # Verify both bits are 0
                    assert (cleared >> q0) & 1 == 0
                    assert (cleared >> q1) & 1 == 0
                    
                    # Test setting new values
                    for new_row in range(4):
                        new_bit0 = new_row & 1
                        new_bit1 = (new_row >> 1) & 1
                        
                        result = cleared | (new_bit0 << q0) | (new_bit1 << q1)
                        
                        assert (result >> q0) & 1 == new_bit0
                        assert (result >> q1) & 1 == new_bit1


class TestGateMatrixRegistration:
    """Test gate matrix registration logic."""
    
    def test_unique_gates_only(self):
        """Verify duplicate gates aren't registered twice."""
        gates = [
            HadamardGate(0),
            HadamardGate(1),  # Same gate, different qubit
            HadamardGate(2),
            XGate(0),
            XGate(1),
        ]
        
        gm_df = register_gates_pandas(gates)
        
        # H should appear once (4 entries for 2x2 matrix)
        h_entries = len(gm_df[gm_df['gate_name'] == 'H'])
        assert h_entries == 4
        
        # X should appear once
        x_entries = len(gm_df[gm_df['gate_name'] == 'X'])
        assert x_entries == 4
    
    def test_two_qubit_gates_16_entries(self):
        """2-qubit gates should have 16 matrix entries."""
        gates = [CNOTGate(0, 1)]
        gm_df = register_gates_pandas(gates)
        
        cnot_entries = len(gm_df[gm_df['gate_name'] == 'CNOT'])
        assert cnot_entries == 16  # 4x4 matrix
    
    def test_gate_matrix_values_correct(self):
        """Verify gate matrix values are stored correctly."""
        gates = [HadamardGate(0)]
        gm_df = register_gates_pandas(gates)
        
        # H = 1/√2 [[1, 1], [1, -1]]
        sqrt2_inv = 1 / np.sqrt(2)
        
        # Check (0,0) entry
        entry_00 = gm_df[(gm_df['row'] == 0) & (gm_df['col'] == 0)]
        np.testing.assert_allclose(entry_00['real'].values[0], sqrt2_inv, atol=1e-10)
        np.testing.assert_allclose(entry_00['imag'].values[0], 0.0, atol=1e-10)
        
        # Check (1,1) entry (should be -1/√2)
        entry_11 = gm_df[(gm_df['row'] == 1) & (gm_df['col'] == 1)]
        np.testing.assert_allclose(entry_11['real'].values[0], -sqrt2_inv, atol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
