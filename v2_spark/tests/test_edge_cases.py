"""
Edge case and stress tests for the quantum simulator.

Tests critical edge cases:
- Bitwise operation correctness at various qubit positions
- Large qubit indices
- Empty/trivial circuits
- Recovery edge cases
- Numerical precision
"""
from __future__ import annotations

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import SimulatorConfig
from src.gates import (
    HadamardGate, XGate, ZGate, CNOTGate, CZGate, CRGate, SWAPGate
)
from src.circuits import generate_ghz_circuit, generate_qft_circuit
from src.frontend import circuit_dict_to_gates
from src.gate_batcher import GateBatcher
from src.metadata_store import MetadataStore

# Import the pure Python simulation from test_unit
from tests.test_unit import apply_gate_numpy, simulate_circuit_numpy


# =============================================================================
# Bitwise Operation Verification
# =============================================================================

class TestBitwiseOperations:
    """
    Test that our bitwise logic for index manipulation is correct.
    
    This is critical because the Spark implementation uses the same logic.
    """
    
    def test_one_qubit_bit_extraction(self):
        """Test extracting single qubit bit from index."""
        for idx in range(16):
            for qubit in range(4):
                bit = (idx >> qubit) & 1
                expected = (idx >> qubit) % 2
                assert bit == expected, f"idx={idx}, qubit={qubit}"
    
    def test_one_qubit_bit_clear_and_set(self):
        """Test clearing and setting a qubit bit."""
        for idx in range(16):
            for qubit in range(4):
                # Clear bit at position qubit
                cleared = idx ^ ((idx >> qubit) & 1) << qubit
                assert (cleared >> qubit) & 1 == 0, f"Failed to clear bit: idx={idx}, qubit={qubit}"
                
                # Set bit to 1
                set_one = cleared | (1 << qubit)
                assert (set_one >> qubit) & 1 == 1
                
                # Set bit to 0 (should remain 0)
                set_zero = cleared | (0 << qubit)
                assert (set_zero >> qubit) & 1 == 0
    
    def test_two_qubit_bit_extraction(self):
        """Test extracting two qubit bits from index."""
        for idx in range(16):
            for q0 in range(4):
                for q1 in range(4):
                    if q0 == q1:
                        continue
                    
                    bit0 = (idx >> q0) & 1
                    bit1 = (idx >> q1) & 1
                    combined = bit0 | (bit1 << 1)
                    
                    expected = ((idx >> q0) & 1) | (((idx >> q1) & 1) << 1)
                    assert combined == expected
    
    def test_two_qubit_bit_clear_and_set(self):
        """Test clearing and setting two qubit bits."""
        for idx in range(16):
            for q0 in range(3):
                for q1 in range(q0 + 1, 4):
                    bit0 = (idx >> q0) & 1
                    bit1 = (idx >> q1) & 1
                    
                    # Clear both bits using XOR
                    cleared = idx ^ (bit0 << q0) ^ (bit1 << q1)
                    
                    assert (cleared >> q0) & 1 == 0
                    assert (cleared >> q1) & 1 == 0
                    
                    # Set new bits
                    for new_row in range(4):
                        new_bit0 = new_row & 1
                        new_bit1 = (new_row >> 1) & 1
                        
                        new_idx = cleared | (new_bit0 << q0) | (new_bit1 << q1)
                        
                        assert (new_idx >> q0) & 1 == new_bit0
                        assert (new_idx >> q1) & 1 == new_bit1


# =============================================================================
# Gate Application at Various Qubit Positions
# =============================================================================

class TestGatePositions:
    """Test gate application at different qubit positions."""
    
    @pytest.mark.parametrize("qubit", [0, 1, 2, 3, 4, 5])
    def test_hadamard_at_position(self, qubit):
        """Test Hadamard at various qubit positions."""
        n_qubits = max(qubit + 1, 3)
        state = np.zeros(2 ** n_qubits, dtype=complex)
        state[0] = 1.0
        
        gate = HadamardGate(qubit)
        result = apply_gate_numpy(state, gate)
        
        # Check normalization
        norm = np.linalg.norm(result)
        np.testing.assert_allclose(norm, 1.0, atol=1e-10)
        
        # Check that only two states have amplitude
        non_zero = np.abs(result) > 1e-10
        assert np.sum(non_zero) == 2
        
        # Check correct states: |0...0⟩ and |0...1...0⟩ (1 at position qubit)
        assert np.abs(result[0]) > 0.5
        assert np.abs(result[1 << qubit]) > 0.5
    
    @pytest.mark.parametrize("control,target", [(0, 1), (1, 0), (0, 2), (2, 0), (1, 3), (3, 1)])
    def test_cnot_at_positions(self, control, target):
        """Test CNOT at various control/target positions."""
        n_qubits = max(control, target) + 1
        
        # Start with |11...1⟩
        state = np.zeros(2 ** n_qubits, dtype=complex)
        state[(1 << n_qubits) - 1] = 1.0  # All ones
        
        gate = CNOTGate(control, target)
        result = apply_gate_numpy(state, gate)
        
        # Control is 1, so target should flip
        # Result should have target bit = 0
        expected_idx = ((1 << n_qubits) - 1) ^ (1 << target)  # Flip target bit
        
        np.testing.assert_allclose(np.abs(result[expected_idx]), 1.0, atol=1e-10)
    
    @pytest.mark.parametrize("q0,q1", [(0, 1), (1, 2), (0, 3), (2, 4)])
    def test_cz_symmetry(self, q0, q1):
        """CZ gate should be symmetric - CZ(a,b) = CZ(b,a)."""
        n_qubits = max(q0, q1) + 1
        
        # Random state
        np.random.seed(42)
        state = np.random.randn(2 ** n_qubits) + 1j * np.random.randn(2 ** n_qubits)
        state /= np.linalg.norm(state)
        
        # Apply CZ(q0, q1)
        result1 = apply_gate_numpy(state.copy(), CZGate(q0, q1))
        
        # Apply CZ(q1, q0)
        result2 = apply_gate_numpy(state.copy(), CZGate(q1, q0))
        
        np.testing.assert_allclose(result1, result2, atol=1e-10)


# =============================================================================
# Numerical Precision Tests
# =============================================================================

class TestNumericalPrecision:
    """Test numerical precision of gate operations."""
    
    def test_hadamard_self_inverse(self):
        """H² = I."""
        for n_qubits in [1, 2, 3]:
            for qubit in range(n_qubits):
                # Random state
                np.random.seed(123 + qubit)
                state = np.random.randn(2 ** n_qubits) + 1j * np.random.randn(2 ** n_qubits)
                state /= np.linalg.norm(state)
                
                # Apply H twice
                gate = HadamardGate(qubit)
                result = apply_gate_numpy(state, gate)
                result = apply_gate_numpy(result, gate)
                
                np.testing.assert_allclose(result, state, atol=1e-10)
    
    def test_x_self_inverse(self):
        """X² = I."""
        for n_qubits in [1, 2, 3]:
            for qubit in range(n_qubits):
                np.random.seed(456 + qubit)
                state = np.random.randn(2 ** n_qubits) + 1j * np.random.randn(2 ** n_qubits)
                state /= np.linalg.norm(state)
                
                gate = XGate(qubit)
                result = apply_gate_numpy(state, gate)
                result = apply_gate_numpy(result, gate)
                
                np.testing.assert_allclose(result, state, atol=1e-10)
    
    def test_cnot_self_inverse(self):
        """CNOT² = I."""
        for n_qubits in [2, 3, 4]:
            for control in range(n_qubits):
                for target in range(n_qubits):
                    if control == target:
                        continue
                    
                    np.random.seed(789 + control * 10 + target)
                    state = np.random.randn(2 ** n_qubits) + 1j * np.random.randn(2 ** n_qubits)
                    state /= np.linalg.norm(state)
                    
                    gate = CNOTGate(control, target)
                    result = apply_gate_numpy(state, gate)
                    result = apply_gate_numpy(result, gate)
                    
                    np.testing.assert_allclose(result, state, atol=1e-10)
    
    def test_deep_circuit_normalization(self):
        """Test that normalization is preserved through deep circuits."""
        n_qubits = 4
        depth = 100
        
        np.random.seed(42)
        state = np.zeros(2 ** n_qubits, dtype=complex)
        state[0] = 1.0
        
        # Apply random gates
        for i in range(depth):
            qubit = i % n_qubits
            gate = HadamardGate(qubit)
            state = apply_gate_numpy(state, gate)
        
        norm = np.linalg.norm(state)
        np.testing.assert_allclose(norm, 1.0, atol=1e-8)
    
    def test_zero_amplitude_cleanup(self):
        """Test that near-zero amplitudes are handled correctly."""
        # Create a state, apply H, then apply X twice (should return to original)
        state = np.zeros(4, dtype=complex)
        state[0] = 1.0
        
        # Apply H, then H again
        state = apply_gate_numpy(state, HadamardGate(0))
        state = apply_gate_numpy(state, HadamardGate(0))
        
        # Should be back to |00⟩ with only numerical noise elsewhere
        assert np.abs(state[0] - 1.0) < 1e-10
        for i in range(1, 4):
            assert np.abs(state[i]) < 1e-10


# =============================================================================
# Edge Cases for Circuits
# =============================================================================

class TestCircuitEdgeCases:
    """Test edge cases in circuit generation and simulation."""
    
    def test_empty_circuit(self):
        """Test circuit with no gates."""
        circuit = {
            "number_of_qubits": 3,
            "gates": []
        }
        
        state = simulate_circuit_numpy(circuit)
        
        expected = np.zeros(8, dtype=complex)
        expected[0] = 1.0
        np.testing.assert_allclose(state, expected, atol=1e-10)
    
    def test_single_gate_circuit(self):
        """Test circuit with single gate."""
        circuit = {
            "number_of_qubits": 2,
            "gates": [{"qubits": [0], "gate": "X"}]
        }
        
        state = simulate_circuit_numpy(circuit)
        
        # X|00⟩ = |01⟩ (idx 1 in little-endian)
        expected = np.zeros(4, dtype=complex)
        expected[1] = 1.0
        np.testing.assert_allclose(state, expected, atol=1e-10)
    
    def test_identity_sequence(self):
        """XX, HH, etc. should be identity."""
        circuit = {
            "number_of_qubits": 2,
            "gates": [
                {"qubits": [0], "gate": "X"},
                {"qubits": [0], "gate": "X"},
                {"qubits": [1], "gate": "H"},
                {"qubits": [1], "gate": "H"},
            ]
        }
        
        state = simulate_circuit_numpy(circuit)
        
        expected = np.zeros(4, dtype=complex)
        expected[0] = 1.0
        np.testing.assert_allclose(state, expected, atol=1e-10)
    
    def test_large_qubit_count(self):
        """Test with larger qubit counts (up to 10)."""
        for n_qubits in [6, 8, 10]:
            circuit = generate_ghz_circuit(n_qubits)
            state = simulate_circuit_numpy(circuit)
            
            # GHZ should have 2 non-zero amplitudes
            non_zero = np.sum(np.abs(state) > 1e-10)
            assert non_zero == 2
            
            # Check normalization
            norm = np.linalg.norm(state)
            np.testing.assert_allclose(norm, 1.0, atol=1e-10)
    
    def test_qft_correctness_small(self):
        """Verify QFT output for small cases."""
        # QFT|0⟩ should give uniform superposition
        circuit = generate_qft_circuit(2)
        state = simulate_circuit_numpy(circuit)
        
        expected_amp = 0.5  # 1/√4 = 0.5
        np.testing.assert_allclose(np.abs(state), expected_amp, atol=1e-10)


# =============================================================================
# WAL and Recovery Edge Cases
# =============================================================================

@pytest.fixture
def temp_config():
    """Create config with temporary directory."""
    temp_dir = Path(tempfile.mkdtemp())
    config = SimulatorConfig(
        run_id="edge_test_run",
        base_path=temp_dir,
    )
    config.ensure_paths()
    yield config
    shutil.rmtree(temp_dir, ignore_errors=True)


class TestWALEdgeCases:
    """Test WAL edge cases."""
    
    def test_multiple_pending_same_run(self, temp_config):
        """Test multiple PENDING entries for same run."""
        store = MetadataStore(temp_config)
        
        # Create multiple pending entries
        ids = []
        for i in range(5):
            wal_id = store.wal_create_pending(
                run_id="edge_test_run",
                gate_start=i * 10,
                gate_end=(i + 1) * 10,
                state_version_in=i,
                state_version_out=i + 1,
            )
            ids.append(wal_id)
        
        pending = store.wal_get_pending("edge_test_run")
        assert len(pending) == 5
        
        # Mark some as committed, some as failed
        store.wal_mark_committed(ids[0])
        store.wal_mark_failed(ids[1])
        store.wal_mark_committed(ids[2])
        
        pending = store.wal_get_pending("edge_test_run")
        assert len(pending) == 2  # ids[3] and ids[4]
        
        store.close()
    
    def test_different_runs_isolation(self, temp_config):
        """Test that different run_ids are isolated."""
        store = MetadataStore(temp_config)
        
        # Create entries for two different runs
        store.wal_create_pending(
            run_id="run_a",
            gate_start=0, gate_end=10,
            state_version_in=0, state_version_out=1,
        )
        store.wal_create_pending(
            run_id="run_b",
            gate_start=0, gate_end=10,
            state_version_in=0, state_version_out=1,
        )
        
        # Check isolation
        pending_a = store.wal_get_pending("run_a")
        pending_b = store.wal_get_pending("run_b")
        
        assert len(pending_a) == 1
        assert len(pending_b) == 1
        assert pending_a[0].run_id == "run_a"
        assert pending_b[0].run_id == "run_b"
        
        store.close()
    
    def test_checkpoint_overwrite(self, temp_config):
        """Test that checkpoints can be overwritten."""
        store = MetadataStore(temp_config)
        
        # Create checkpoint
        store.checkpoint_create(
            run_id="edge_test_run",
            state_version=1,
            last_gate_seq=9,
            state_path="/path/v1",
        )
        
        # Overwrite with new data
        store.checkpoint_create(
            run_id="edge_test_run",
            state_version=1,
            last_gate_seq=19,  # Different
            state_path="/path/v1_updated",
        )
        
        # Should have only one checkpoint with updated values
        latest = store.checkpoint_get_latest("edge_test_run")
        assert latest.last_gate_seq == 19
        assert latest.state_path == "/path/v1_updated"
        
        checkpoints = store.checkpoint_list("edge_test_run")
        assert len(checkpoints) == 1
        
        store.close()


# =============================================================================
# Batcher Edge Cases
# =============================================================================

class TestBatcherEdgeCases:
    """Test gate batcher edge cases."""
    
    def test_batch_size_one(self):
        """Test with batch size of 1."""
        batcher = GateBatcher(batch_size=1)
        gates = [HadamardGate(i % 4) for i in range(5)]
        
        batches = batcher.create_batches(gates)
        
        assert len(batches) == 5
        for batch in batches:
            assert batch.size == 1
    
    def test_batch_size_larger_than_gates(self):
        """Test batch size larger than total gates."""
        batcher = GateBatcher(batch_size=100)
        gates = [HadamardGate(i % 4) for i in range(5)]
        
        batches = batcher.create_batches(gates)
        
        assert len(batches) == 1
        assert batches[0].size == 5
    
    def test_batch_sequence_continuity(self):
        """Test that batch sequences are continuous."""
        batcher = GateBatcher(batch_size=3)
        gates = [HadamardGate(i % 4) for i in range(10)]
        
        batches = batcher.create_batches(gates, start_seq=100)
        
        # Check continuity
        for i in range(len(batches) - 1):
            assert batches[i].end_seq == batches[i + 1].start_seq
        
        # Check first and last
        assert batches[0].start_seq == 100
        assert batches[-1].end_seq == 110


# =============================================================================
# Stress Tests
# =============================================================================

class TestStress:
    """Stress tests for performance and correctness under load."""
    
    def test_many_single_qubit_gates(self):
        """Test circuit with many single-qubit gates."""
        n_qubits = 3
        depth = 500
        
        gates = []
        for i in range(depth):
            qubit = i % n_qubits
            gate_type = ["H", "X", "Z"][i % 3]
            gates.append({"qubits": [qubit], "gate": gate_type})
        
        circuit = {
            "number_of_qubits": n_qubits,
            "gates": gates
        }
        
        state = simulate_circuit_numpy(circuit)
        
        # Just check normalization
        norm = np.linalg.norm(state)
        np.testing.assert_allclose(norm, 1.0, atol=1e-8)
    
    def test_alternating_two_qubit_gates(self):
        """Test circuit with alternating 2-qubit gates."""
        n_qubits = 4
        depth = 100
        
        gates = []
        for i in range(depth):
            control = i % n_qubits
            target = (i + 1) % n_qubits
            if control == target:
                target = (target + 1) % n_qubits
            gates.append({"qubits": [control, target], "gate": "CNOT"})
        
        circuit = {
            "number_of_qubits": n_qubits,
            "gates": gates
        }
        
        state = simulate_circuit_numpy(circuit)
        
        # Check normalization
        norm = np.linalg.norm(state)
        np.testing.assert_allclose(norm, 1.0, atol=1e-8)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
