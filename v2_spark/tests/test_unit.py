"""
Unit tests for quantum simulator components that don't require Spark.

Tests:
- Gate definitions and matrix correctness
- Gate batching logic
- Metadata store (WAL, checkpoints)
- Pure Python gate application verification
- Configuration
"""
from __future__ import annotations

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil
from typing import Dict

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import SimulatorConfig
from src.gates import (
    Gate, HadamardGate, XGate, YGate, ZGate, 
    CNOTGate, CZGate, CRGate, SWAPGate
)
from src.circuits import (
    generate_ghz_circuit, generate_qft_circuit, 
    generate_w_circuit, generate_qpe_circuit
)
from src.frontend import circuit_dict_to_gates
from src.gate_batcher import GateBatcher, GateBatch
from src.metadata_store import MetadataStore, WALEntry, CheckpointRecord


# =============================================================================
# Gate Definition Tests
# =============================================================================

class TestGateDefinitions:
    """Test that gate matrices are correct."""
    
    def test_hadamard_is_unitary(self):
        """Hadamard gate should be unitary: H†H = I."""
        h = HadamardGate(0)
        H = h.tensor
        
        # Check unitarity
        product = H @ H.conj().T
        expected = np.eye(2)
        np.testing.assert_allclose(product, expected, atol=1e-10)
    
    def test_hadamard_values(self):
        """Hadamard gate should have correct values."""
        h = HadamardGate(0)
        H = h.tensor
        
        sqrt2_inv = 1 / np.sqrt(2)
        expected = sqrt2_inv * np.array([[1, 1], [1, -1]])
        np.testing.assert_allclose(H, expected, atol=1e-10)
    
    def test_x_gate(self):
        """X gate should flip |0⟩ to |1⟩."""
        x = XGate(0)
        X = x.tensor
        
        expected = np.array([[0, 1], [1, 0]])
        np.testing.assert_allclose(X, expected, atol=1e-10)
        
        # Check unitarity
        product = X @ X.conj().T
        np.testing.assert_allclose(product, np.eye(2), atol=1e-10)
    
    def test_z_gate(self):
        """Z gate should have correct phase."""
        z = ZGate(0)
        Z = z.tensor
        
        expected = np.array([[1, 0], [0, -1]])
        np.testing.assert_allclose(Z, expected, atol=1e-10)
    
    def test_cnot_is_unitary(self):
        """CNOT gate should be unitary."""
        cnot = CNOTGate(0, 1)
        CNOT = cnot.tensor.reshape(4, 4)
        
        product = CNOT @ CNOT.conj().T
        np.testing.assert_allclose(product, np.eye(4), atol=1e-10)
    
    def test_cz_is_symmetric(self):
        """CZ gate should be symmetric in control/target."""
        cz = CZGate(0, 1)
        CZ = cz.tensor.reshape(4, 4)
        
        # CZ is symmetric
        np.testing.assert_allclose(CZ, CZ.T, atol=1e-10)
    
    def test_swap_swaps_qubits(self):
        """SWAP gate should swap two qubits."""
        swap = SWAPGate(0, 1)
        SWAP = swap.tensor.reshape(4, 4)
        
        # SWAP|01⟩ = |10⟩, SWAP|10⟩ = |01⟩
        # In little-endian: |01⟩ = idx 1, |10⟩ = idx 2
        state_01 = np.array([0, 1, 0, 0], dtype=complex)
        state_10 = np.array([0, 0, 1, 0], dtype=complex)
        
        result_01 = SWAP @ state_01
        result_10 = SWAP @ state_10
        
        np.testing.assert_allclose(result_01, state_10, atol=1e-10)
        np.testing.assert_allclose(result_10, state_01, atol=1e-10)


# =============================================================================
# Pure Python Gate Application (for verification)
# =============================================================================

def apply_gate_numpy(state: np.ndarray, gate: Gate) -> np.ndarray:
    """
    Apply a gate to a state vector using NumPy.
    This is the reference implementation for testing.
    """
    n_qubits = int(np.log2(len(state)))
    
    if gate.two_qubit_gate:
        return _apply_two_qubit_gate_numpy(state, gate, n_qubits)
    else:
        return _apply_one_qubit_gate_numpy(state, gate, n_qubits)


def _apply_one_qubit_gate_numpy(
    state: np.ndarray, 
    gate: Gate, 
    n_qubits: int
) -> np.ndarray:
    """Apply 1-qubit gate using NumPy."""
    qubit = gate.qubits[0]
    U = gate.tensor  # 2x2 matrix
    
    new_state = np.zeros_like(state)
    
    for idx in range(len(state)):
        if state[idx] == 0:
            continue
            
        # Get the qubit bit
        qubit_bit = (idx >> qubit) & 1
        
        # For each row in the gate matrix
        for row in range(2):
            # Compute new index
            new_idx = (idx & ~(1 << qubit)) | (row << qubit)
            # Add contribution
            new_state[new_idx] += U[row, qubit_bit] * state[idx]
    
    return new_state


def _apply_two_qubit_gate_numpy(
    state: np.ndarray, 
    gate: Gate, 
    n_qubits: int
) -> np.ndarray:
    """Apply 2-qubit gate using NumPy."""
    q0, q1 = gate.qubits
    U = gate.tensor.reshape(4, 4)
    
    new_state = np.zeros_like(state)
    
    for idx in range(len(state)):
        if state[idx] == 0:
            continue
            
        # Get the qubit bits
        bit0 = (idx >> q0) & 1
        bit1 = (idx >> q1) & 1
        col = bit0 | (bit1 << 1)
        
        # For each row in the gate matrix
        for row in range(4):
            # Compute new index
            new_bit0 = row & 1
            new_bit1 = (row >> 1) & 1
            new_idx = (idx & ~((1 << q0) | (1 << q1))) | (new_bit0 << q0) | (new_bit1 << q1)
            # Add contribution
            new_state[new_idx] += U[row, col] * state[idx]
    
    return new_state


class TestPureGateApplication:
    """Test pure Python gate application for correctness."""
    
    def test_hadamard_on_zero(self):
        """H|0⟩ = (|0⟩ + |1⟩) / √2."""
        state = np.array([1, 0], dtype=complex)
        gate = HadamardGate(0)
        
        result = apply_gate_numpy(state, gate)
        
        sqrt2_inv = 1 / np.sqrt(2)
        expected = np.array([sqrt2_inv, sqrt2_inv], dtype=complex)
        np.testing.assert_allclose(result, expected, atol=1e-10)
    
    def test_hadamard_on_one(self):
        """H|1⟩ = (|0⟩ - |1⟩) / √2."""
        state = np.array([0, 1], dtype=complex)
        gate = HadamardGate(0)
        
        result = apply_gate_numpy(state, gate)
        
        sqrt2_inv = 1 / np.sqrt(2)
        expected = np.array([sqrt2_inv, -sqrt2_inv], dtype=complex)
        np.testing.assert_allclose(result, expected, atol=1e-10)
    
    def test_x_flips_zero(self):
        """X|0⟩ = |1⟩."""
        state = np.array([1, 0], dtype=complex)
        gate = XGate(0)
        
        result = apply_gate_numpy(state, gate)
        
        expected = np.array([0, 1], dtype=complex)
        np.testing.assert_allclose(result, expected, atol=1e-10)
    
    def test_double_x_is_identity(self):
        """XX|ψ⟩ = |ψ⟩."""
        state = np.array([0.6, 0.8], dtype=complex)
        gate = XGate(0)
        
        result = apply_gate_numpy(state, gate)
        result = apply_gate_numpy(result, gate)
        
        np.testing.assert_allclose(result, state, atol=1e-10)
    
    def test_cnot_creates_bell_state(self):
        """H on q0, then CNOT(0,1) creates Bell state."""
        # Start with |00⟩
        state = np.array([1, 0, 0, 0], dtype=complex)
        
        # Apply H to qubit 0
        h = HadamardGate(0)
        state = apply_gate_numpy(state, h)
        
        # Apply CNOT(0, 1)
        cnot = CNOTGate(0, 1)
        state = apply_gate_numpy(state, cnot)
        
        # Expected: (|00⟩ + |11⟩) / √2
        sqrt2_inv = 1 / np.sqrt(2)
        expected = np.array([sqrt2_inv, 0, 0, sqrt2_inv], dtype=complex)
        np.testing.assert_allclose(state, expected, atol=1e-10)
    
    def test_ghz_3_qubits(self):
        """GHZ(3) = (|000⟩ + |111⟩) / √2."""
        # Start with |000⟩
        state = np.zeros(8, dtype=complex)
        state[0] = 1
        
        # Apply H to qubit 0
        state = apply_gate_numpy(state, HadamardGate(0))
        
        # Apply CNOT(0, 1)
        state = apply_gate_numpy(state, CNOTGate(0, 1))
        
        # Apply CNOT(1, 2)
        state = apply_gate_numpy(state, CNOTGate(1, 2))
        
        # Expected: (|000⟩ + |111⟩) / √2
        sqrt2_inv = 1 / np.sqrt(2)
        expected = np.zeros(8, dtype=complex)
        expected[0b000] = sqrt2_inv
        expected[0b111] = sqrt2_inv
        
        np.testing.assert_allclose(state, expected, atol=1e-10)
    
    def test_normalization_preserved(self):
        """Gate application should preserve normalization."""
        # Random normalized state
        np.random.seed(42)
        state = np.random.randn(4) + 1j * np.random.randn(4)
        state /= np.linalg.norm(state)
        
        # Apply various gates
        state = apply_gate_numpy(state, HadamardGate(0))
        state = apply_gate_numpy(state, XGate(1))
        state = apply_gate_numpy(state, CNOTGate(0, 1))
        
        # Should still be normalized
        norm = np.linalg.norm(state)
        np.testing.assert_allclose(norm, 1.0, atol=1e-10)


# =============================================================================
# Circuit Generator Tests
# =============================================================================

class TestCircuitGenerators:
    """Test circuit generators produce valid circuits."""
    
    def test_ghz_circuit_structure(self):
        """GHZ circuit should have H + (n-1) CNOTs."""
        n = 5
        circuit = generate_ghz_circuit(n)
        
        assert circuit["number_of_qubits"] == n
        assert len(circuit["gates"]) == n  # 1 H + (n-1) CNOTs
        
        # First gate is H on qubit 0
        assert circuit["gates"][0]["gate"] == "H"
        assert circuit["gates"][0]["qubits"] == [0]
        
        # Rest are CNOTs
        for i in range(1, n):
            assert circuit["gates"][i]["gate"] == "CNOT"
    
    def test_qft_circuit_structure(self):
        """QFT circuit should have correct gate count."""
        n = 3
        circuit = generate_qft_circuit(n)
        
        assert circuit["number_of_qubits"] == n
        
        # Count gates: n H gates + n(n-1)/2 CR gates
        expected_gates = n + (n * (n - 1)) // 2
        assert len(circuit["gates"]) == expected_gates
    
    def test_circuit_parsing(self):
        """Circuit dict should parse to Gate objects correctly."""
        circuit = generate_ghz_circuit(3)
        n_qubits, gates = circuit_dict_to_gates(circuit)
        
        assert n_qubits == 3
        assert len(gates) == 3
        
        assert isinstance(gates[0], HadamardGate)
        assert isinstance(gates[1], CNOTGate)
        assert isinstance(gates[2], CNOTGate)


# =============================================================================
# Gate Batcher Tests
# =============================================================================

class TestGateBatcher:
    """Test gate batching logic."""
    
    def test_exact_batch_size(self):
        """Test batching with exact multiples."""
        batcher = GateBatcher(batch_size=3)
        gates = [HadamardGate(i % 4) for i in range(9)]
        
        batches = batcher.create_batches(gates)
        
        assert len(batches) == 3
        assert all(b.size == 3 for b in batches)
    
    def test_partial_last_batch(self):
        """Test batching with partial last batch."""
        batcher = GateBatcher(batch_size=3)
        gates = [HadamardGate(i % 4) for i in range(7)]
        
        batches = batcher.create_batches(gates)
        
        assert len(batches) == 3
        assert batches[0].size == 3
        assert batches[1].size == 3
        assert batches[2].size == 1  # Partial batch
    
    def test_batch_sequence_numbers(self):
        """Test that batch sequence numbers are correct."""
        batcher = GateBatcher(batch_size=2)
        gates = [HadamardGate(i % 4) for i in range(5)]
        
        batches = batcher.create_batches(gates, start_seq=10)
        
        assert batches[0].start_seq == 10
        assert batches[0].end_seq == 12
        assert batches[1].start_seq == 12
        assert batches[1].end_seq == 14
        assert batches[2].start_seq == 14
        assert batches[2].end_seq == 15
    
    def test_batch_from_seq(self):
        """Test getting batches from a specific sequence."""
        batcher = GateBatcher(batch_size=2)
        gates = [HadamardGate(i % 4) for i in range(6)]
        
        # Skip first 3 gates
        batches = batcher.get_batches_from_seq(gates, from_seq=3)
        
        assert len(batches) == 2  # 3 remaining gates = 2 batches
        assert batches[0].start_seq == 3
        assert len(batches[0].gates) == 2
    
    def test_empty_gates(self):
        """Test batching empty gate list."""
        batcher = GateBatcher(batch_size=5)
        batches = batcher.create_batches([])
        
        assert len(batches) == 0
    
    def test_single_gate(self):
        """Test batching single gate."""
        batcher = GateBatcher(batch_size=5)
        gates = [HadamardGate(0)]
        
        batches = batcher.create_batches(gates)
        
        assert len(batches) == 1
        assert batches[0].size == 1


# =============================================================================
# Metadata Store Tests
# =============================================================================

@pytest.fixture
def temp_config():
    """Create config with temporary directory."""
    temp_dir = Path(tempfile.mkdtemp())
    config = SimulatorConfig(
        run_id="test_run",
        base_path=temp_dir,
    )
    config.ensure_paths()
    yield config
    shutil.rmtree(temp_dir, ignore_errors=True)


class TestMetadataStore:
    """Test metadata store (WAL and checkpoints)."""
    
    def test_wal_create_pending(self, temp_config):
        """Test creating PENDING WAL entry."""
        store = MetadataStore(temp_config)
        
        wal_id = store.wal_create_pending(
            run_id="test_run",
            gate_start=0,
            gate_end=10,
            state_version_in=0,
            state_version_out=1,
        )
        
        assert wal_id > 0
        
        # Check pending entries
        pending = store.wal_get_pending("test_run")
        assert len(pending) == 1
        assert pending[0].status == "PENDING"
        
        store.close()
    
    def test_wal_mark_committed(self, temp_config):
        """Test marking WAL entry as COMMITTED."""
        store = MetadataStore(temp_config)
        
        wal_id = store.wal_create_pending(
            run_id="test_run",
            gate_start=0,
            gate_end=10,
            state_version_in=0,
            state_version_out=1,
        )
        
        store.wal_mark_committed(wal_id)
        
        # Should have no pending entries
        pending = store.wal_get_pending("test_run")
        assert len(pending) == 0
        
        # Should have committed entry
        last_committed = store.wal_get_last_committed("test_run")
        assert last_committed is not None
        assert last_committed.status == "COMMITTED"
        
        store.close()
    
    def test_wal_mark_failed(self, temp_config):
        """Test marking WAL entry as FAILED."""
        store = MetadataStore(temp_config)
        
        wal_id = store.wal_create_pending(
            run_id="test_run",
            gate_start=0,
            gate_end=10,
            state_version_in=0,
            state_version_out=1,
        )
        
        store.wal_mark_failed(wal_id)
        
        # Should have no pending entries
        pending = store.wal_get_pending("test_run")
        assert len(pending) == 0
        
        # Should have no committed entries
        last_committed = store.wal_get_last_committed("test_run")
        assert last_committed is None
        
        store.close()
    
    def test_checkpoint_create_and_get(self, temp_config):
        """Test creating and retrieving checkpoints."""
        store = MetadataStore(temp_config)
        
        cp_id = store.checkpoint_create(
            run_id="test_run",
            state_version=1,
            last_gate_seq=9,
            state_path="/data/state/v1",
            checksum="abc123",
        )
        
        assert cp_id > 0
        
        # Get latest checkpoint
        latest = store.checkpoint_get_latest("test_run")
        assert latest is not None
        assert latest.state_version == 1
        assert latest.last_gate_seq == 9
        
        store.close()
    
    def test_checkpoint_multiple_versions(self, temp_config):
        """Test that latest checkpoint is correctly identified."""
        store = MetadataStore(temp_config)
        
        # Create multiple checkpoints
        for v in [1, 2, 3]:
            store.checkpoint_create(
                run_id="test_run",
                state_version=v,
                last_gate_seq=v * 10 - 1,
                state_path=f"/data/state/v{v}",
            )
        
        # Latest should be version 3
        latest = store.checkpoint_get_latest("test_run")
        assert latest.state_version == 3
        assert latest.last_gate_seq == 29
        
        # Get specific version
        v2 = store.checkpoint_get_by_version("test_run", 2)
        assert v2.state_version == 2
        
        store.close()
    
    def test_wal_multiple_entries(self, temp_config):
        """Test WAL with multiple entries."""
        store = MetadataStore(temp_config)
        
        # Create multiple WAL entries
        ids = []
        for i in range(3):
            wal_id = store.wal_create_pending(
                run_id="test_run",
                gate_start=i * 10,
                gate_end=(i + 1) * 10,
                state_version_in=i,
                state_version_out=i + 1,
            )
            ids.append(wal_id)
        
        # Commit first two
        store.wal_mark_committed(ids[0])
        store.wal_mark_committed(ids[1])
        
        # One should be pending
        pending = store.wal_get_pending("test_run")
        assert len(pending) == 1
        assert pending[0].gate_start == 20
        
        # Last committed should be second entry
        last = store.wal_get_last_committed("test_run")
        assert last.gate_end == 20
        
        store.close()


# =============================================================================
# Configuration Tests
# =============================================================================

class TestConfiguration:
    """Test configuration handling."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = SimulatorConfig()
        
        assert config.batch_size == 10
        assert config.spark_master == "local[*]"
        assert len(config.run_id) == 8
    
    def test_custom_config(self):
        """Test custom configuration."""
        config = SimulatorConfig(
            run_id="my_run",
            batch_size=50,
            base_path=Path("/tmp/test"),
        )
        
        assert config.run_id == "my_run"
        assert config.batch_size == 50
    
    def test_path_generation(self):
        """Test path generation methods."""
        config = SimulatorConfig(
            run_id="test_run",
            base_path=Path("/data"),
        )
        
        state_path = config.state_version_path(5)
        assert "run_id=test_run" in str(state_path)
        assert "state_version=5" in str(state_path)


# =============================================================================
# Integration Tests (Pure Python Simulation)
# =============================================================================

def simulate_circuit_numpy(circuit_dict: Dict) -> np.ndarray:
    """
    Simulate a circuit using pure NumPy.
    This is the reference implementation for validating Spark results.
    """
    n_qubits, gates = circuit_dict_to_gates(circuit_dict)
    
    # Initialize |0...0⟩
    state = np.zeros(2 ** n_qubits, dtype=complex)
    state[0] = 1.0
    
    # Apply all gates
    for gate in gates:
        state = apply_gate_numpy(state, gate)
    
    return state


class TestCircuitSimulation:
    """End-to-end circuit simulation tests using pure Python."""
    
    @pytest.mark.parametrize("n_qubits", [2, 3, 4, 5])
    def test_ghz_circuit(self, n_qubits):
        """Test GHZ circuit produces correct state."""
        circuit = generate_ghz_circuit(n_qubits)
        state = simulate_circuit_numpy(circuit)
        
        # GHZ state: (|00...0⟩ + |11...1⟩) / √2
        sqrt2_inv = 1 / np.sqrt(2)
        
        # Check only two non-zero amplitudes
        non_zero = np.abs(state) > 1e-10
        assert np.sum(non_zero) == 2
        
        # Check correct indices
        assert np.abs(state[0]) > 0.5  # |00...0⟩
        assert np.abs(state[(1 << n_qubits) - 1]) > 0.5  # |11...1⟩
        
        # Check amplitudes
        np.testing.assert_allclose(np.abs(state[0]), sqrt2_inv, atol=1e-10)
        np.testing.assert_allclose(
            np.abs(state[(1 << n_qubits) - 1]), 
            sqrt2_inv, 
            atol=1e-10
        )
    
    @pytest.mark.parametrize("n_qubits", [2, 3, 4])
    def test_qft_normalization(self, n_qubits):
        """Test QFT produces normalized state."""
        circuit = generate_qft_circuit(n_qubits)
        state = simulate_circuit_numpy(circuit)
        
        # Check normalization
        norm = np.linalg.norm(state)
        np.testing.assert_allclose(norm, 1.0, atol=1e-10)
    
    def test_qft_on_zero(self):
        """QFT|0⟩ should produce uniform superposition."""
        n = 3
        circuit = generate_qft_circuit(n)
        state = simulate_circuit_numpy(circuit)
        
        # QFT|0⟩ = uniform superposition
        expected_amp = 1 / np.sqrt(2 ** n)
        np.testing.assert_allclose(np.abs(state), expected_amp, atol=1e-10)
    
    def test_w_state_normalization(self, ):
        """Test W state is normalized."""
        circuit = generate_w_circuit(4)
        state = simulate_circuit_numpy(circuit)
        
        norm = np.linalg.norm(state)
        np.testing.assert_allclose(norm, 1.0, atol=1e-10)
    
    def test_w_state_structure(self):
        """W state should have n non-zero amplitudes."""
        n = 3
        circuit = generate_w_circuit(n)
        state = simulate_circuit_numpy(circuit)
        
        # W state has exactly n non-zero amplitudes
        non_zero = np.abs(state) > 1e-10
        assert np.sum(non_zero) == n
        
        # Each non-zero amplitude should have magnitude 1/√n
        expected_amp = 1 / np.sqrt(n)
        for amp in state[non_zero]:
            np.testing.assert_allclose(np.abs(amp), expected_amp, atol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
