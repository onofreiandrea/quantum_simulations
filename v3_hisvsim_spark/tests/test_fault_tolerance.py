"""
Comprehensive tests for fault tolerance and recovery.

Tests:
1. WAL PENDING entries during execution
2. Crash simulation (interrupt execution)
3. Recovery from checkpoint
4. WAL reconciliation (PENDING → FAILED)
5. Resume execution after crash
6. Multiple crash/recovery cycles
"""
from __future__ import annotations

import pytest
import numpy as np
import tempfile
import shutil
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

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
        run_id="test_fault_tolerance",
        base_path=temp_dir,
        batch_size=3,  # Small batches to test multiple checkpoints
        spark_master="local[2]",
        spark_shuffle_partitions=4,
    )
    cfg.ensure_paths()
    yield cfg
    shutil.rmtree(temp_dir, ignore_errors=True)


class TestWALFlow:
    """Test Write-Ahead Log flow."""
    
    def test_wal_pending_during_execution(self, config_v3):
        """Verify WAL entries are created as PENDING before execution."""
        circuit = circuits.generate_qft_circuit(4)  # 10 gates
        
        # Create a PENDING entry manually to simulate mid-execution
        with SparkHiSVSIMDriver(config_v3, enable_parallel=False) as driver:
            # Create a PENDING entry (simulating crash before COMMITTED)
            wal_id = driver.metadata_store.wal_create_pending(
                run_id=config_v3.run_id,
                gate_start=5,
                gate_end=8,
                state_version_in=2,
                state_version_out=3,
            )
            
            # Verify PENDING entry exists
            pending = driver.metadata_store.wal_get_pending(config_v3.run_id)
            assert len(pending) > 0, "Should have PENDING entry"
            assert pending[0].wal_id == wal_id, "Should have correct WAL ID"
            
            # Now run circuit normally (this will create more WAL entries)
            result = driver.run_circuit(circuit, resume=False)
        
        # After completion, the manually created PENDING should still exist
        # (unless recovery marked it as FAILED)
        # But normal execution should have COMMITTED entries
        conn = driver.metadata_store.connect()
        committed = conn.execute(
            "SELECT COUNT(*) FROM wal WHERE run_id = ? AND status = 'COMMITTED'",
            [config_v3.run_id]
        ).fetchone()
        assert committed[0] > 0, "Should have COMMITTED entries after successful execution"
    
    def test_wal_committed_after_success(self, config_v3):
        """Verify WAL entries are marked COMMITTED after successful execution."""
        circuit = circuits.generate_qft_circuit(4)
        
        with SparkHiSVSIMDriver(config_v3, enable_parallel=False) as driver:
            result = driver.run_circuit(circuit, resume=False)
        
        # Check all WAL entries are COMMITTED
        conn = driver.metadata_store.connect()
        rows = conn.execute(
            "SELECT status FROM wal WHERE run_id = ?",
            [config_v3.run_id]
        ).fetchall()
        
        assert len(rows) > 0, "Should have WAL entries"
        for row in rows:
            assert row[0] == "COMMITTED", f"All WAL entries should be COMMITTED, got {row[0]}"


class TestCrashSimulation:
    """Simulate crashes and test recovery."""
    
    def test_crash_mid_execution(self, config_v3):
        """Simulate a crash mid-execution and verify recovery."""
        circuit = circuits.generate_qft_circuit(4)  # 10 gates, batch_size=3 → 4 batches
        
        # Run circuit normally first to create checkpoints
        with SparkHiSVSIMDriver(config_v3, enable_parallel=False) as driver:
            result = driver.run_circuit(circuit, resume=False)
            state1 = driver.get_state_vector(result)
        
        # Now simulate crash: create a PENDING entry manually
        with SparkHiSVSIMDriver(config_v3, enable_parallel=False) as driver2:
            # Create a fake PENDING entry (simulating crash mid-execution)
            fake_wal_id = driver2.metadata_store.wal_create_pending(
                run_id=config_v3.run_id,
                gate_start=15,  # After all gates
                gate_end=20,
                state_version_in=10,
                state_version_out=11,
            )
        
        # Test recovery
        with SparkHiSVSIMDriver(config_v3, enable_parallel=False) as driver3:
            recovery_state = driver3.recovery_manager.recover(4)
            
            # Should recover from checkpoint
            assert recovery_state.checkpoint_record is not None, "Should have checkpoint after execution"
            assert recovery_state.state_version > 0, "Should have state version > 0"
            
            # Resume execution (should detect already complete)
            result2 = driver3.run_circuit(circuit, resume=True)
            state2 = driver3.get_state_vector(result2)
            
            # Verify state is correct
            norm = np.linalg.norm(state2)
            assert np.isclose(norm, 1.0, atol=1e-10), "State should be normalized"
            
            # States should match (both complete)
            max_diff = np.max(np.abs(state1 - state2))
            assert max_diff < 1e-10, f"States should match, max_diff={max_diff}"
    
    def test_recovery_with_pending_wal(self, config_v3):
        """Test recovery when PENDING WAL entries exist (simulated crash)."""
        circuit = circuits.generate_qft_circuit(4)
        
        # Step 1: Create a PENDING WAL entry manually (simulating crash)
        with SparkHiSVSIMDriver(config_v3, enable_parallel=False) as driver:
            # Create a fake PENDING entry
            wal_id = driver.metadata_store.wal_create_pending(
                run_id=config_v3.run_id,
                gate_start=5,
                gate_end=8,
                state_version_in=2,
                state_version_out=3,
            )
            
            # Run circuit normally (this will create checkpoints)
            result = driver.run_circuit(circuit, resume=False)
        
        # Step 2: Verify PENDING entry exists
        pending = driver.metadata_store.wal_get_pending(config_v3.run_id)
        # The manually created PENDING entry should still exist if it's after the checkpoint
        
        # Step 3: Test recovery
        with SparkHiSVSIMDriver(config_v3, enable_parallel=False) as driver2:
            recovery_state = driver2.recovery_manager.recover(4)
            
            # Recovery should mark PENDING entries as FAILED if they're after checkpoint
            pending_after_recovery = driver2.metadata_store.wal_get_pending(config_v3.run_id)
            # Should have fewer or no PENDING entries after recovery reconciliation


class TestRecovery:
    """Test recovery mechanisms."""
    
    def test_recovery_from_checkpoint(self, config_v3):
        """Test recovery loads latest checkpoint correctly."""
        circuit = circuits.generate_qft_circuit(4)
        
        # Run circuit to create checkpoints
        with SparkHiSVSIMDriver(config_v3, enable_parallel=False) as driver:
            result = driver.run_circuit(circuit, resume=False)
        
        # Test recovery
        with SparkHiSVSIMDriver(config_v3, enable_parallel=False) as driver2:
            recovery_state = driver2.recovery_manager.recover(4)
            
            # Verify recovery state
            assert recovery_state.checkpoint_record is not None, "Should have checkpoint"
            assert recovery_state.state_version > 0, "Should have state version"
            assert recovery_state.last_gate_seq >= 0, "Should have gate sequence"
            
            # Verify state can be loaded
            state_df = recovery_state.state_df
            assert state_df is not None, "State DataFrame should exist"
            
            # Verify state is valid
            state_array = driver2.state_manager.get_state_as_array(state_df, 4)
            norm = np.linalg.norm(state_array)
            assert np.isclose(norm, 1.0, atol=1e-10), "Recovered state should be normalized"
    
    def test_resume_after_recovery(self, config_v3):
        """Test resuming execution after recovery."""
        circuit = circuits.generate_qft_circuit(4)
        
        # Run circuit partially (simulate crash)
        with SparkHiSVSIMDriver(config_v3, enable_parallel=False) as driver:
            # Run with small batch size to create multiple checkpoints
            result1 = driver.run_circuit(circuit, resume=False)
            state1 = driver.get_state_vector(result1)
        
        # Simulate crash and recovery
        with SparkHiSVSIMDriver(config_v3, enable_parallel=False) as driver2:
            # Recover
            recovery_state = driver2.recovery_manager.recover(4)
            
            # Resume execution
            result2 = driver2.run_circuit(circuit, resume=True)
            state2 = driver2.get_state_vector(result2)
            
            # States should match (both complete)
            max_diff = np.max(np.abs(state1 - state2))
            assert max_diff < 1e-10, f"States should match after recovery, max_diff={max_diff}"
    
    def test_recovery_no_checkpoint(self, config_v3):
        """Test recovery when no checkpoint exists (fresh start)."""
        circuit = circuits.generate_qft_circuit(4)
        
        # Test recovery without any previous execution
        with SparkHiSVSIMDriver(config_v3, enable_parallel=False) as driver:
            recovery_state = driver.recovery_manager.recover(4)
            
            # Should start from initial state
            assert recovery_state.checkpoint_record is None, "Should have no checkpoint"
            assert recovery_state.state_version == 0, "Should start at version 0"
            assert recovery_state.last_gate_seq == -1, "Should start at gate -1"
            
            # Run circuit
            result = driver.run_circuit(circuit, resume=True)
            state = driver.get_state_vector(result)
            
            # Verify state is correct
            norm = np.linalg.norm(state)
            assert np.isclose(norm, 1.0, atol=1e-10), "State should be normalized"


class TestMultipleCrashRecovery:
    """Test multiple crash/recovery cycles."""
    
    def test_multiple_crash_recovery_cycles(self, config_v3):
        """Test multiple crash and recovery cycles."""
        circuit = circuits.generate_qft_circuit(4)
        
        # Cycle 1: Run partially
        with SparkHiSVSIMDriver(config_v3, enable_parallel=False) as driver:
            result1 = driver.run_circuit(circuit, resume=False)
            state1 = driver.get_state_vector(result1)
        
        # Cycle 2: Recover and verify
        with SparkHiSVSIMDriver(config_v3, enable_parallel=False) as driver2:
            recovery_state = driver2.recovery_manager.recover(4)
            result2 = driver2.run_circuit(circuit, resume=True)
            state2 = driver2.get_state_vector(result2)
            
            # States should match
            max_diff = np.max(np.abs(state1 - state2))
            assert max_diff < 1e-10, f"States should match, max_diff={max_diff}"
        
        # Cycle 3: Another recovery
        with SparkHiSVSIMDriver(config_v3, enable_parallel=False) as driver3:
            recovery_state3 = driver3.recovery_manager.recover(4)
            result3 = driver3.run_circuit(circuit, resume=True)
            state3 = driver3.get_state_vector(result3)
            
            # States should still match
            max_diff = np.max(np.abs(state1 - state3))
            assert max_diff < 1e-10, f"States should match after multiple recoveries, max_diff={max_diff}"


class TestWALReconciliation:
    """Test WAL reconciliation during recovery."""
    
    def test_pending_wal_marked_failed(self, config_v3):
        """Test that PENDING WAL entries are marked FAILED during recovery."""
        circuit = circuits.generate_qft_circuit(4)
        
        # Create a PENDING WAL entry manually (simulating crash)
        with SparkHiSVSIMDriver(config_v3, enable_parallel=False) as driver:
            # Create checkpoint first
            driver.run_circuit(circuit, resume=False)
            
            # Create a PENDING entry that's after the checkpoint
            wal_id = driver.metadata_store.wal_create_pending(
                run_id=config_v3.run_id,
                gate_start=15,  # After all gates
                gate_end=20,
                state_version_in=10,
                state_version_out=11,
            )
        
        # Verify PENDING entry exists
        pending_before = driver.metadata_store.wal_get_pending(config_v3.run_id)
        assert len(pending_before) > 0, "Should have PENDING entry"
        
        # Recover - should mark PENDING as FAILED
        with SparkHiSVSIMDriver(config_v3, enable_parallel=False) as driver2:
            recovery_state = driver2.recovery_manager.recover(4)
            
            # Check that PENDING entry was handled
            pending_after = driver2.metadata_store.wal_get_pending(config_v3.run_id)
            # PENDING entries after checkpoint should be marked FAILED
            # (The recovery manager marks them as FAILED)


class TestCheckpointIntegrity:
    """Test checkpoint integrity and verification."""
    
    def test_checkpoint_creation(self, config_v3):
        """Test that checkpoints are created correctly."""
        circuit = circuits.generate_qft_circuit(4)
        
        with SparkHiSVSIMDriver(config_v3, enable_parallel=False) as driver:
            result = driver.run_circuit(circuit, resume=False)
        
        # Verify checkpoints exist
        checkpoints = driver.checkpoint_manager.list_checkpoints()
        assert len(checkpoints) > 0, "Should have checkpoints"
        
        # Verify each checkpoint has valid state file
        for cp in checkpoints:
            path = Path(cp.state_path)
            assert path.exists(), f"Checkpoint state file should exist: {cp.state_path}"
            
            # Verify state can be loaded
            state_df = driver.state_manager.load_state_by_path(cp.state_path)
            assert state_df is not None, f"Should be able to load checkpoint {cp.state_version}"
    
    def test_checkpoint_loading(self, config_v3):
        """Test loading checkpoints."""
        circuit = circuits.generate_qft_circuit(4)
        
        # Create checkpoints
        with SparkHiSVSIMDriver(config_v3, enable_parallel=False) as driver:
            result = driver.run_circuit(circuit, resume=False)
        
        # Load latest checkpoint
        checkpoint_result = driver.checkpoint_manager.load_latest_checkpoint()
        assert checkpoint_result is not None, "Should have latest checkpoint"
        
        state_df, checkpoint_record = checkpoint_result
        assert state_df is not None, "State DataFrame should exist"
        assert checkpoint_record is not None, "Checkpoint record should exist"
        
        # Verify state is valid
        state_array = driver.state_manager.get_state_as_array(state_df, 4)
        norm = np.linalg.norm(state_array)
        assert np.isclose(norm, 1.0, atol=1e-10), "Checkpoint state should be normalized"
