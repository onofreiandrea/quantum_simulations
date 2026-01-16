"""
Checkpoint management for quantum state snapshots.

Checkpoints are durable snapshots of quantum state that enable:
- Recovery after crashes
- Resuming long-running simulations
- State inspection at specific points
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional, Tuple

from pyspark.sql import SparkSession, DataFrame

from v2_common.config import SimulatorConfig
from v2_common.state_manager import StateManager
from v2_common.metadata_store import MetadataStore, CheckpointRecord


class CheckpointManager:
    """
    Manages checkpoint creation and loading for quantum state.
    
    Checkpoints are stored as:
    - State data: Parquet files in /data/state/run_id=R/state_version=V/
    - Metadata: Record in checkpoints table (version, path, checksum, etc.)
    """
    
    def __init__(
        self,
        spark: SparkSession,
        config: SimulatorConfig,
        state_manager: StateManager,
        metadata_store: MetadataStore,
    ):
        self.spark = spark
        self.config = config
        self.state_manager = state_manager
        self.metadata_store = metadata_store
    
    def create_checkpoint(
        self,
        state_version: int,
        last_gate_seq: int,
        state_path: Optional[Path] = None,
        compute_checksum: bool = False,
    ) -> CheckpointRecord:
        """
        Create a checkpoint record for an already-saved state.
        
        NOTE: The state must already be saved to Parquet before calling this.
        This method only records the checkpoint metadata.
        
        Args:
            state_version: Version number for this state.
            last_gate_seq: Last gate sequence number that was applied.
            state_path: Path where state is saved (defaults to config path).
            compute_checksum: Whether to compute SHA256 checksum (expensive).
            
        Returns:
            CheckpointRecord with checkpoint metadata.
        """
        # Use provided path or default
        if state_path is None:
            state_path = self.config.state_version_path(state_version)
        
        # Compute checksum if requested
        checksum = None
        if compute_checksum:
            checksum = self._compute_checksum(state_path)
        
        # Record checkpoint in metadata store
        checkpoint_id = self.metadata_store.checkpoint_create(
            run_id=self.config.run_id,
            state_version=state_version,
            last_gate_seq=last_gate_seq,
            state_path=str(state_path),
            checksum=checksum,
        )
        
        from datetime import datetime
        return CheckpointRecord(
            checkpoint_id=checkpoint_id,
            run_id=self.config.run_id,
            state_version=state_version,
            last_gate_seq=last_gate_seq,
            state_path=str(state_path),
            checksum=checksum,
            created_at=datetime.now(),
        )
    
    def load_latest_checkpoint(self) -> Optional[Tuple[DataFrame, CheckpointRecord]]:
        """
        Load the latest checkpoint for the current run.
        
        Returns:
            Tuple of (state_df, checkpoint_record) or None if no checkpoint exists.
        """
        record = self.metadata_store.checkpoint_get_latest(self.config.run_id)
        if record is None:
            return None
        
        state_df = self.state_manager.load_state_by_path(record.state_path)
        return state_df, record
    
    def load_checkpoint(
        self, 
        state_version: int
    ) -> Optional[Tuple[DataFrame, CheckpointRecord]]:
        """
        Load a specific checkpoint by version.
        
        Args:
            state_version: Version number of checkpoint to load.
            
        Returns:
            Tuple of (state_df, checkpoint_record) or None if not found.
        """
        record = self.metadata_store.checkpoint_get_by_version(
            self.config.run_id, 
            state_version
        )
        if record is None:
            return None
        
        state_df = self.state_manager.load_state_by_path(record.state_path)
        return state_df, record
    
    def verify_checkpoint(self, record: CheckpointRecord) -> bool:
        """
        Verify a checkpoint's integrity using stored checksum.
        
        Args:
            record: Checkpoint record to verify.
            
        Returns:
            True if checksum matches or no checksum stored, False otherwise.
        """
        if record.checksum is None:
            return True  # No checksum to verify
        
        current_checksum = self._compute_checksum(Path(record.state_path))
        return current_checksum == record.checksum
    
    def list_checkpoints(self):
        """List all checkpoints for the current run."""
        return self.metadata_store.checkpoint_list(self.config.run_id)
    
    def _compute_checksum(self, state_path: Path) -> str:
        """
        Compute SHA256 checksum of state Parquet files.
        
        Note: This is expensive for large states. Consider skipping
        for performance-critical scenarios.
        """
        hasher = hashlib.sha256()
        
        # Hash all Parquet files in the state directory
        if state_path.is_dir():
            for parquet_file in sorted(state_path.glob("*.parquet")):
                hasher.update(parquet_file.read_bytes())
        else:
            hasher.update(state_path.read_bytes())
        
        return hasher.hexdigest()
