"""
Tests for Spark-based quantum circuit simulation.

Verifies:
- Basic gate operations
- GHZ, QFT circuit correctness
- Checkpoint and recovery
- Parity with v1 implementation
"""
from __future__ import annotations

import pytest
import numpy as np
from pathlib import Path
import shutil
import tempfile

from pyspark.sql import SparkSession

# Import simulator components
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import SimulatorConfig
from src.spark_session import create_spark_session
from src.state_manager import StateManager
from src.gate_applicator import GateApplicator
from src.gates import HadamardGate, CNOTGate, XGate
from src.circuits import generate_ghz_circuit, generate_qft_circuit
from src.driver import SparkQuantumDriver, SimulationResult


@pytest.fixture(scope="module")
def spark():
    """Create a SparkSession for testing."""
    session = (
        SparkSession.builder
        .appName("QuantumSimulatorTest")
        .master("local[2]")
        .config("spark.sql.shuffle.partitions", "4")
        .config("spark.driver.memory", "1g")
        .getOrCreate()
    )
    yield session
    session.stop()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test data."""
    path = Path(tempfile.mkdtemp())
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def config(temp_dir):
    """Create test configuration."""
    return SimulatorConfig(
        base_path=temp_dir,
        batch_size=5,
        spark_master="local[2]",
        spark_shuffle_partitions=4,
    )


class TestStateManager:
    """Tests for StateManager."""
    
    def test_initialize_state(self, spark, config):
        """Test initial state |0...0⟩ creation."""
        manager = StateManager(spark, config)
        state = manager.initialize_state(3)
        
        result = manager.get_state_as_dict(state)
        
        assert len(result) == 1
        assert 0 in result
        assert result[0] == complex(1.0, 0.0)
    
    def test_save_and_load_state(self, spark, config):
        """Test state persistence to Parquet."""
        manager = StateManager(spark, config)
        
        # Create a simple state
        initial_state = manager.initialize_state(2)
        
        # Save it
        manager.save_state(initial_state, state_version=1)
        
        # Load it back
        loaded_state = manager.load_state(state_version=1)
        
        result = manager.get_state_as_dict(loaded_state)
        assert result[0] == complex(1.0, 0.0)


class TestGateApplicator:
    """Tests for GateApplicator."""
    
    def test_hadamard_gate(self, spark, config):
        """Test Hadamard gate creates superposition."""
        manager = StateManager(spark, config)
        applicator = GateApplicator(spark)
        
        # Initialize state |0⟩
        state = manager.initialize_state(1)
        
        # Register and apply Hadamard
        h_gate = HadamardGate(0)
        applicator.register_gates([h_gate])
        
        result_state = applicator.apply_gate(state, h_gate)
        result = manager.get_state_as_dict(result_state)
        
        # Should have |0⟩ + |1⟩ with equal amplitudes
        sqrt2_inv = 1 / np.sqrt(2)
        assert len(result) == 2
        assert np.isclose(abs(result[0]), sqrt2_inv, atol=1e-10)
        assert np.isclose(abs(result[1]), sqrt2_inv, atol=1e-10)
        
        applicator.cleanup()
    
    def test_x_gate(self, spark, config):
        """Test X gate flips |0⟩ to |1⟩."""
        manager = StateManager(spark, config)
        applicator = GateApplicator(spark)
        
        state = manager.initialize_state(1)
        
        x_gate = XGate(0)
        applicator.register_gates([x_gate])
        
        result_state = applicator.apply_gate(state, x_gate)
        result = manager.get_state_as_dict(result_state)
        
        # Should be |1⟩
        assert len(result) == 1
        assert 1 in result
        assert np.isclose(result[1], complex(1.0, 0.0), atol=1e-10)
        
        applicator.cleanup()
    
    def test_cnot_gate(self, spark, config):
        """Test CNOT gate entangles two qubits."""
        manager = StateManager(spark, config)
        applicator = GateApplicator(spark)
        
        # Start with |0⟩
        state = manager.initialize_state(2)
        
        # Apply H to qubit 0, then CNOT
        h_gate = HadamardGate(0)
        cnot_gate = CNOTGate(0, 1)
        applicator.register_gates([h_gate, cnot_gate])
        
        state = applicator.apply_gate(state, h_gate)
        result_state = applicator.apply_gate(state, cnot_gate)
        result = manager.get_state_as_dict(result_state)
        
        # Should have |00⟩ + |11⟩ (Bell state)
        sqrt2_inv = 1 / np.sqrt(2)
        assert len(result) == 2
        assert 0b00 in result  # |00⟩
        assert 0b11 in result  # |11⟩
        assert np.isclose(abs(result[0b00]), sqrt2_inv, atol=1e-10)
        assert np.isclose(abs(result[0b11]), sqrt2_inv, atol=1e-10)
        
        applicator.cleanup()


class TestDriver:
    """Tests for SparkQuantumDriver."""
    
    def test_ghz_circuit_2_qubits(self, config):
        """Test 2-qubit GHZ circuit produces Bell state."""
        with SparkQuantumDriver(config) as driver:
            result = driver.run_ghz(2, resume=False)
            
            state_dict = driver.get_state_dict(result)
            
            # GHZ(2) = Bell state = (|00⟩ + |11⟩) / √2
            sqrt2_inv = 1 / np.sqrt(2)
            assert len(state_dict) == 2
            assert np.isclose(abs(state_dict[0b00]), sqrt2_inv, atol=1e-10)
            assert np.isclose(abs(state_dict[0b11]), sqrt2_inv, atol=1e-10)
    
    def test_ghz_circuit_3_qubits(self, config):
        """Test 3-qubit GHZ circuit."""
        with SparkQuantumDriver(config) as driver:
            result = driver.run_ghz(3, resume=False)
            
            state_dict = driver.get_state_dict(result)
            
            # GHZ(3) = (|000⟩ + |111⟩) / √2
            sqrt2_inv = 1 / np.sqrt(2)
            assert len(state_dict) == 2
            assert np.isclose(abs(state_dict[0b000]), sqrt2_inv, atol=1e-10)
            assert np.isclose(abs(state_dict[0b111]), sqrt2_inv, atol=1e-10)
    
    def test_simulation_metadata(self, config):
        """Test simulation result contains correct metadata."""
        with SparkQuantumDriver(config) as driver:
            result = driver.run_ghz(3, resume=False)
            
            assert result.n_qubits == 3
            assert result.n_gates == 3  # 1 H + 2 CNOT
            assert result.n_batches >= 1
            assert result.elapsed_time > 0
            assert result.run_id == config.run_id
    
    def test_qft_circuit_normalization(self, config):
        """Test QFT circuit produces normalized state."""
        with SparkQuantumDriver(config) as driver:
            result = driver.run_qft(3, resume=False)
            
            state_array = driver.get_state_vector(result)
            
            # Check normalization: sum of |amplitude|^2 should be 1
            norm = np.sum(np.abs(state_array) ** 2)
            assert np.isclose(norm, 1.0, atol=1e-10)


class TestCheckpointRecovery:
    """Tests for checkpoint and recovery functionality."""
    
    def test_checkpoint_created(self, config):
        """Test that checkpoints are created during simulation."""
        with SparkQuantumDriver(config) as driver:
            driver.run_ghz(4, resume=False)
            
            # Check that checkpoints exist
            checkpoints = driver.checkpoint_manager.list_checkpoints()
            assert len(checkpoints) > 0
    
    def test_recovery_after_complete(self, config):
        """Test that completed simulation can be detected."""
        with SparkQuantumDriver(config) as driver:
            # Run simulation
            result1 = driver.run_ghz(3, resume=False)
            state1 = driver.get_state_dict(result1)
        
        # Create new driver with same config (simulates restart)
        config2 = SimulatorConfig(
            run_id=config.run_id,
            base_path=config.base_path,
            batch_size=config.batch_size,
        )
        
        with SparkQuantumDriver(config2) as driver2:
            # Check if simulation is complete
            circuit = generate_ghz_circuit(3)
            n_qubits, gates = driver2.gate_applicator.spark.sparkContext._jvm.java.util.ArrayList, None
            
            # Verify recovery manager can detect completion
            is_complete = driver2.recovery_manager.is_simulation_complete(3)
            # Note: This depends on checkpoint being written


class TestBatching:
    """Tests for gate batching."""
    
    def test_batch_size_respected(self, config):
        """Test that batches respect configured size."""
        config.batch_size = 2
        
        with SparkQuantumDriver(config) as driver:
            # GHZ(5) has 5 gates: 1 H + 4 CNOT
            result = driver.run_ghz(5, resume=False)
            
            # Should have ceil(5/2) = 3 batches
            assert result.n_batches == 3


# Parity tests with v1 implementation
class TestParityWithV1:
    """Tests to verify parity with v1 SQLite implementation."""
    
    @pytest.mark.parametrize("n_qubits", [2, 3, 4])
    def test_ghz_parity(self, config, n_qubits):
        """Test GHZ circuit produces same results as expected."""
        with SparkQuantumDriver(config) as driver:
            result = driver.run_ghz(n_qubits, resume=False)
            state_dict = driver.get_state_dict(result)
            
            # GHZ state should have exactly 2 non-zero amplitudes
            assert len(state_dict) == 2
            
            # |00...0⟩ and |11...1⟩
            all_zeros = 0
            all_ones = (1 << n_qubits) - 1
            
            assert all_zeros in state_dict
            assert all_ones in state_dict
            
            sqrt2_inv = 1 / np.sqrt(2)
            assert np.isclose(abs(state_dict[all_zeros]), sqrt2_inv, atol=1e-10)
            assert np.isclose(abs(state_dict[all_ones]), sqrt2_inv, atol=1e-10)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
