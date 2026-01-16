"""
Configuration for the Spark quantum simulator.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import uuid


@dataclass
class SimulatorConfig:
    """Configuration for the quantum circuit simulator."""
    
    # Run identification
    run_id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    
    # Storage paths
    base_path: Path = field(default_factory=lambda: Path("data"))
    
    # Batch processing
    batch_size: int = 10  # Number of gates per batch
    
    # Spark configuration
    spark_app_name: str = "QuantumSimulator"
    spark_master: str = "local[*]"
    spark_shuffle_partitions: int = 200
    spark_driver_memory: str = "4g"
    spark_executor_memory: str = "4g"
    
    # Checkpoint settings
    checkpoint_every_n_batches: int = 1  # Checkpoint after every N batches
    checkpoint_every_n_gates: int = 10  # Checkpoint after N gates (adaptive)
    checkpoint_threshold_size: int = 1_000_000  # Checkpoint if state has > N rows
    checkpoint_min_interval_seconds: float = 60.0  # Minimum time between checkpoints
    
    @property
    def state_path(self) -> Path:
        """Path to state snapshots: /data/state/run_id=R/state_version=V/"""
        return self.base_path / "state"
    
    @property
    def gate_matrix_path(self) -> Path:
        """Path to gate matrices: /data/gate_matrix/"""
        return self.base_path / "gate_matrix"
    
    @property
    def wal_path(self) -> Path:
        """Path to WAL: /data/wal/"""
        return self.base_path / "wal"
    
    @property
    def checkpoint_path(self) -> Path:
        """Path to checkpoint metadata: /data/checkpoints/"""
        return self.base_path / "checkpoints"
    
    @property
    def metadata_db_path(self) -> Path:
        """Path to SQLite metadata database."""
        return self.base_path / "metadata.duckdb"
    
    def state_version_path(self, state_version: int) -> Path:
        """Full path for a specific state version."""
        return self.state_path / f"run_id={self.run_id}" / f"state_version={state_version}"
    
    def ensure_paths(self):
        """Create all required directories."""
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.state_path.mkdir(parents=True, exist_ok=True)
        self.gate_matrix_path.mkdir(parents=True, exist_ok=True)
        self.wal_path.mkdir(parents=True, exist_ok=True)
        self.checkpoint_path.mkdir(parents=True, exist_ok=True)


# Default configuration instance
DEFAULT_CONFIG = SimulatorConfig()
