"""
Recovery manager for crash recovery and simulation resumption.

Handles:
- Loading latest checkpoint
- Reconciling PENDING WAL entries
- Resuming execution from last committed gate
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Tuple

from pyspark.sql import SparkSession, DataFrame

from .config import SimulatorConfig
from .gates import Gate
from .state_manager import StateManager
from .metadata_store import MetadataStore, WALEntry, CheckpointRecord
from .checkpoint_manager import CheckpointManager


@dataclass
class RecoveryState:
    """
    State recovered from checkpoint and WAL reconciliation.
    
    Attributes:
        state_df: Recovered state DataFrame.
        state_version: Version of the recovered state.
        last_gate_seq: Last successfully applied gate sequence number.
        checkpoint_record: Checkpoint record used (if any).
        pending_wal_entries: PENDING WAL entries found (for inspection).
    """
    state_df: DataFrame
    state_version: int
    last_gate_seq: int
    checkpoint_record: Optional[CheckpointRecord]
    pending_wal_entries: List[WALEntry]


class RecoveryManager:
    """
    Manages crash recovery for the quantum simulator.
    
    Recovery process:
    1. Load latest checkpoint (if exists)
    2. Check for PENDING WAL entries
    3. Mark PENDING entries as FAILED (incomplete batches)
    4. Return state ready for resumption from last_gate_seq + 1
    """
    
    def __init__(
        self,
        spark: SparkSession,
        config: SimulatorConfig,
        state_manager: StateManager,
        metadata_store: MetadataStore,
        checkpoint_manager: CheckpointManager,
    ):
        self.spark = spark
        self.config = config
        self.state_manager = state_manager
        self.metadata_store = metadata_store
        self.checkpoint_manager = checkpoint_manager
    
    def recover(self, n_qubits: int) -> RecoveryState:
        """
        Perform recovery and return state ready for resumption.
        
        Args:
            n_qubits: Number of qubits (needed if no checkpoint exists).
            
        Returns:
            RecoveryState with recovered state and metadata.
        """
        # Step 1: Load latest checkpoint
        checkpoint_result = self.checkpoint_manager.load_latest_checkpoint()
        
        if checkpoint_result is not None:
            state_df, checkpoint_record = checkpoint_result
            state_version = checkpoint_record.state_version
            last_gate_seq = checkpoint_record.last_gate_seq
        else:
            # No checkpoint - start from initial state
            state_df = self.state_manager.initialize_state(n_qubits)
            state_version = 0
            last_gate_seq = -1  # No gates applied yet
            checkpoint_record = None
        
        # Step 2: Check for PENDING WAL entries
        pending_entries = self.metadata_store.wal_get_pending(self.config.run_id)
        
        # Step 3: Handle PENDING entries
        for entry in pending_entries:
            # Check if PENDING entry is after our checkpoint
            if entry.gate_start > last_gate_seq:
                # This batch was never completed - mark as FAILED
                self.metadata_store.wal_mark_failed(entry.wal_id)
                
                # Clean up any partial state that might have been written
                if self.state_manager.state_exists(entry.state_version_out):
                    self.state_manager.delete_state(entry.state_version_out)
        
        # Step 4: Check WAL for any COMMITTED entries after checkpoint
        last_committed = self.metadata_store.wal_get_last_committed(self.config.run_id)
        
        if last_committed is not None and last_committed.gate_end > last_gate_seq + 1:
            # There's a COMMITTED batch after our checkpoint
            # Load that state instead
            if self.state_manager.state_exists(last_committed.state_version_out):
                state_df = self.state_manager.load_state(last_committed.state_version_out)
                state_version = last_committed.state_version_out
                last_gate_seq = last_committed.gate_end - 1
        
        return RecoveryState(
            state_df=state_df,
            state_version=state_version,
            last_gate_seq=last_gate_seq,
            checkpoint_record=checkpoint_record,
            pending_wal_entries=pending_entries,
        )
    
    def get_resume_point(self, total_gates: int) -> Tuple[int, int]:
        """
        Get the point from which to resume execution.
        
        Args:
            total_gates: Total number of gates in the circuit.
            
        Returns:
            Tuple of (start_gate_seq, start_version).
        """
        # Check latest checkpoint
        checkpoint_record = self.metadata_store.checkpoint_get_latest(self.config.run_id)
        
        if checkpoint_record is not None:
            start_gate_seq = checkpoint_record.last_gate_seq + 1
            start_version = checkpoint_record.state_version
        else:
            start_gate_seq = 0
            start_version = 0
        
        # Check for COMMITTED WAL entries after checkpoint
        last_committed = self.metadata_store.wal_get_last_committed(self.config.run_id)
        
        if last_committed is not None:
            if last_committed.gate_end > start_gate_seq:
                start_gate_seq = last_committed.gate_end
                start_version = last_committed.state_version_out
        
        return start_gate_seq, start_version
    
    def is_simulation_complete(self, total_gates: int) -> bool:
        """
        Check if the simulation has already completed.
        
        Args:
            total_gates: Total number of gates in the circuit.
            
        Returns:
            True if all gates have been applied.
        """
        start_gate_seq, _ = self.get_resume_point(total_gates)
        return start_gate_seq >= total_gates
    
    def cleanup_orphaned_states(self, keep_versions: List[int]):
        """
        Clean up state versions that are not in the keep list.
        
        Useful for cleaning up after failed runs or old checkpoints.
        
        Args:
            keep_versions: List of state versions to keep.
        """
        all_versions = self.state_manager.list_state_versions()
        
        for version in all_versions:
            if version not in keep_versions:
                self.state_manager.delete_state(version)
