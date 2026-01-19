"""
State management for quantum simulation using Spark DataFrames and Parquet storage.

State is stored as sparse vectors:
- idx: basis state index (e.g., 0b101 = |101⟩)
- real: real part of amplitude
- imag: imaginary part of amplitude
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional, List, Tuple

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.types import StructType, StructField, LongType, DoubleType
import pyspark.sql.functions as F

from .config import SimulatorConfig


# Schema for quantum state DataFrame
STATE_SCHEMA = StructType([
    StructField("idx", LongType(), nullable=False),
    StructField("real", DoubleType(), nullable=False),
    StructField("imag", DoubleType(), nullable=False),
])


class StateManager:
    """
    Manages quantum state storage and retrieval using Spark DataFrames.
    
    State is stored as Parquet files partitioned by run_id and state_version:
    /data/state/run_id=R/state_version=V/part-*.parquet
    """
    
    def __init__(self, spark: SparkSession, config: SimulatorConfig):
        self.spark = spark
        self.config = config
    
    def initialize_state(self, n_qubits: int) -> DataFrame:
        """
        Create initial state |0...0⟩ as a DataFrame.
        
        Args:
            n_qubits: Number of qubits in the system.
            
        Returns:
            DataFrame with single row: (idx=0, real=1.0, imag=0.0)
        """
        # Initial state is |0...0⟩ with amplitude 1
        initial_data = [(0, 1.0, 0.0)]
        return self.spark.createDataFrame(initial_data, schema=STATE_SCHEMA)
    
    def save_state(self, state_df: DataFrame, state_version: int) -> Path:
        """
        Save state DataFrame to Parquet storage.
        
        Args:
            state_df: DataFrame containing (idx, real, imag) columns.
            state_version: Version number for this state snapshot.
            
        Returns:
            Path where state was saved.
        """
        path = self.config.state_version_path(state_version)
        
        # Write as Parquet, overwriting if exists
        state_df.write.mode("overwrite").parquet(str(path))
        
        return path
    
    def load_state(self, state_version: int) -> DataFrame:
        """
        Load state DataFrame from Parquet storage.
        
        Args:
            state_version: Version number of state to load.
            
        Returns:
            DataFrame containing (idx, real, imag) columns.
        """
        path = self.config.state_version_path(state_version)
        return self.spark.read.schema(STATE_SCHEMA).parquet(str(path))
    
    def state_exists(self, state_version: int) -> bool:
        """Check if a state version exists in storage."""
        path = self.config.state_version_path(state_version)
        return path.exists()
    
    def load_state_by_path(self, path: str | Path) -> DataFrame:
        """Load state from a specific path."""
        return self.spark.read.schema(STATE_SCHEMA).parquet(str(path))
    
    def get_state_as_dict(self, state_df: DataFrame) -> dict[int, complex]:
        """
        Convert state DataFrame to a dictionary for inspection/testing.
        
        Args:
            state_df: State DataFrame.
            
        Returns:
            Dictionary mapping basis state index to complex amplitude.
        """
        rows = state_df.collect()
        return {row.idx: complex(row.real, row.imag) for row in rows}
    
    def get_state_as_array(self, state_df: DataFrame, n_qubits: int):
        """
        Convert sparse state DataFrame to dense numpy array.
        
        Args:
            state_df: State DataFrame.
            n_qubits: Number of qubits (determines array size).
            
        Returns:
            Complex numpy array of shape (2^n_qubits,).
        """
        import numpy as np
        
        size = 2 ** n_qubits
        arr = np.zeros(size, dtype=complex)
        
        for row in state_df.collect():
            if row.idx < size:
                arr[row.idx] = complex(row.real, row.imag)
        
        return arr
    
    def delete_state(self, state_version: int):
        """Delete a state version from storage."""
        import shutil
        path = self.config.state_version_path(state_version)
        if path.exists():
            shutil.rmtree(path)
    
    def list_state_versions(self) -> List[int]:
        """List all available state versions for current run_id."""
        run_path = self.config.state_path / f"run_id={self.config.run_id}"
        if not run_path.exists():
            return []
        
        versions = []
        for p in run_path.iterdir():
            if p.is_dir() and p.name.startswith("state_version="):
                try:
                    version = int(p.name.split("=")[1])
                    versions.append(version)
                except ValueError:
                    pass
        
        return sorted(versions)
