"""
Critical tests for recovery logic edge cases.

These tests verify:
1. Recovery after crash during batch processing
2. Resume from checkpoint
3. PENDING WAL entries are properly handled
4. State consistency after recovery
"""
from __future__ import annotations

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil

from pyspark.sql import SparkSession

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import SimulatorConfig
from src.state_manager import StateManager
from src.gate_applicator import GateApplicator
from src.gates import HadamardGate, CNOTGate
from src.circuits import generate_ghz_circuit
from src.frontend import circuit_dict_to_gates
from src.metadata_store import MetadataStore
from src.checkpoint_manager import CheckpointManager
from src.recovery_manager import RecoveryManager, RecoveryState
from src.driver import SparkQuantumDriver


@pytest.fixture(scope="module")
def spark():
    """Create a SparkSession for testing."""
    session = (
        SparkSession.builder
        .appName("RecoveryTest")
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
        run_id="recovery_test",
        base_path=temp_dir,
        batch_size=2,
        spark_master="local[2]",
        spark_shuffle_partitions=4,
    )


class TestRecoveryFromCheckpoint:
    """Test recovery scenarios."""
    
    def test_recovery_no_prior_state(self, spark, config):
        """Test recovery when there's no prior state."""
        state_manager = StateManager(spark, config)
        metadata_store = MetadataStore(config)
        checkpoint_manager = CheckpointManager(
            spark, config, state_manager, metadata_store
        )
        recovery_manager = RecoveryManager(
            spark, config, state_manager, metadata_store, checkpoint_manager
        )
        
        # Recover with no prior state
        recovery_state = recovery_manager.recover(n_qubits=3)
        
        # Should have initial state
        assert recovery_state.state_version == 0
        assert recovery_state.last_gate_seq == -1
        assert recovery_state.checkpoint_record is None
        
        # State should be |000⟩
        state_dict = state_manager.get_state_as_dict(recovery_state.state_df)
        assert len(state_dict) == 1
        assert state_dict[0] == complex(1.0, 0.0)
        
        metadata_store.close()
    
    def test_recovery_with_checkpoint(self, spark, config):
        """Test recovery from existing checkpoint."""
        state_manager = StateManager(spark, config)
        metadata_store = MetadataStore(config)
        checkpoint_manager = CheckpointManager(
            spark, config, state_manager, metadata_store
        )
        applicator = GateApplicator(spark)
        
        # Create initial state and apply some gates
        state = state_manager.initialize_state(2)
        h_gate = HadamardGate(0)
        applicator.register_gates([h_gate])
        state = applicator.apply_gate(state, h_gate)
        
        # Save state and create checkpoint
        state_path = state_manager.save_state(state, state_version=1)
        checkpoint_manager.create_checkpoint(
            state_version=1,
            last_gate_seq=0,
            state_path=state_path,
        )
        
        # Create new recovery manager (simulates restart)
        recovery_manager = RecoveryManager(
            spark, config, state_manager, metadata_store, checkpoint_manager
        )
        
        recovery_state = recovery_manager.recover(n_qubits=2)
        
        assert recovery_state.state_version == 1
        assert recovery_state.last_gate_seq == 0
        assert recovery_state.checkpoint_record is not None
        
        # State should be H|00⟩
        state_dict = state_manager.get_state_as_dict(recovery_state.state_df)
        assert len(state_dict) == 2  # Superposition
        
        applicator.cleanup()
        metadata_store.close()
    
    def test_recovery_with_pending_wal(self, spark, config):
        """Test recovery with PENDING WAL entries."""
        state_manager = StateManager(spark, config)
        metadata_store = MetadataStore(config)
        checkpoint_manager = CheckpointManager(
            spark, config, state_manager, metadata_store
        )
        
        # Create initial state and checkpoint
        state = state_manager.initialize_state(2)
        state_path = state_manager.save_state(state, state_version=0)
        checkpoint_manager.create_checkpoint(
            state_version=0,
            last_gate_seq=-1,
            state_path=state_path,
        )
        
        # Simulate crash: create PENDING WAL entry but don't complete
        wal_id = metadata_store.wal_create_pending(
            run_id=config.run_id,
            gate_start=0,
            gate_end=2,
            state_version_in=0,
            state_version_out=1,
        )
        
        # Recovery should find and handle this PENDING entry
        recovery_manager = RecoveryManager(
            spark, config, state_manager, metadata_store, checkpoint_manager
        )
        recovery_state = recovery_manager.recover(n_qubits=2)
        
        # Should have marked PENDING as FAILED
        pending = metadata_store.wal_get_pending(config.run_id)
        assert len(pending) == 0
        
        # State should be at checkpoint
        assert recovery_state.state_version == 0
        assert recovery_state.last_gate_seq == -1
        
        metadata_store.close()
    
    def test_resume_partial_circuit(self, config):
        """Test resuming a partially completed circuit."""
        # Run first part of circuit
        with SparkQuantumDriver(config) as driver:
            circuit = generate_ghz_circuit(4)
            result1 = driver.run_circuit(circuit, resume=False)
            
            first_state = driver.get_state_dict(result1)
            first_version = result1.final_version
        
        # Create new driver with same run_id (simulates restart)
        config2 = SimulatorConfig(
            run_id=config.run_id,
            base_path=config.base_path,
            batch_size=config.batch_size,
        )
        
        with SparkQuantumDriver(config2) as driver2:
            # Resume should detect completed simulation
            is_complete = driver2.recovery_manager.is_simulation_complete(4)
            assert is_complete, "Simulation should be detected as complete"
    
    def test_consistent_state_after_recovery(self, config):
        """Verify state is consistent after recovery."""
        # Run circuit completely
        with SparkQuantumDriver(config) as driver:
            result = driver.run_ghz(3, resume=False)
            expected_state = driver.get_state_dict(result)
        
        # Create new driver and verify state
        config2 = SimulatorConfig(
            run_id=config.run_id,
            base_path=config.base_path,
            batch_size=config.batch_size,
        )
        
        with SparkQuantumDriver(config2) as driver2:
            # Recover
            recovery_state = driver2.recovery_manager.recover(n_qubits=3)
            recovered_state = driver2.state_manager.get_state_as_dict(
                recovery_state.state_df
            )
            
            # States should match
            assert set(recovered_state.keys()) == set(expected_state.keys())
            for idx in recovered_state:
                np.testing.assert_allclose(
                    abs(recovered_state[idx]),
                    abs(expected_state[idx]),
                    atol=1e-10
                )


class TestCheckpointSequence:
    """Test checkpoint sequencing."""
    
    def test_multiple_checkpoints_created(self, config):
        """Test that multiple checkpoints are created during long circuit."""
        # Use small batch size to create multiple batches
        config.batch_size = 2
        
        with SparkQuantumDriver(config) as driver:
            # GHZ(5) has 5 gates = ceil(5/2) = 3 batches = 3 checkpoints
            result = driver.run_ghz(5, resume=False)
            
            checkpoints = driver.checkpoint_manager.list_checkpoints()
            assert len(checkpoints) == 3
            
            # Verify checkpoint sequence
            for i, cp in enumerate(checkpoints):
                assert cp.state_version == i + 1
    
    def test_checkpoint_versions_sequential(self, config):
        """Verify checkpoint versions are sequential."""
        config.batch_size = 1  # One checkpoint per gate
        
        with SparkQuantumDriver(config) as driver:
            result = driver.run_ghz(4, resume=False)
            
            checkpoints = driver.checkpoint_manager.list_checkpoints()
            
            # Versions should be 1, 2, 3, 4
            versions = [cp.state_version for cp in checkpoints]
            assert versions == [1, 2, 3, 4]
            
            # Gate sequences should be correct
            gate_seqs = [cp.last_gate_seq for cp in checkpoints]
            assert gate_seqs == [0, 1, 2, 3]


class TestWALConsistency:
    """Test WAL consistency."""
    
    def test_all_wal_committed_after_success(self, config):
        """After successful run, all WAL entries should be COMMITTED."""
        with SparkQuantumDriver(config) as driver:
            driver.run_ghz(4, resume=False)
            
            pending = driver.metadata_store.wal_get_pending(config.run_id)
            assert len(pending) == 0, "No PENDING entries after successful run"
    
    def test_wal_failed_on_exception(self, spark, config):
        """WAL should be marked FAILED if batch processing fails."""
        state_manager = StateManager(spark, config)
        metadata_store = MetadataStore(config)
        
        # Create PENDING entry
        wal_id = metadata_store.wal_create_pending(
            run_id=config.run_id,
            gate_start=0,
            gate_end=5,
            state_version_in=0,
            state_version_out=1,
        )
        
        # Mark as FAILED (simulating exception handling)
        metadata_store.wal_mark_failed(wal_id)
        
        # Verify
        pending = metadata_store.wal_get_pending(config.run_id)
        assert len(pending) == 0
        
        last_committed = metadata_store.wal_get_last_committed(config.run_id)
        assert last_committed is None
        
        metadata_store.close()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
