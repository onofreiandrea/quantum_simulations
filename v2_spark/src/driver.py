"""
Spark Quantum Simulator Driver.

Orchestrates the complete simulation flow:
1. Load circuit → parse gates
2. Load/create checkpoint
3. For each batch:
   - Write WAL PENDING
   - Build DataFrame plan (lazy)
   - Trigger ACTION (write state)
   - Write checkpoint + mark WAL COMMITTED
4. Return final state
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict, Any
import time

from pyspark.sql import SparkSession, DataFrame

from .config import SimulatorConfig
from .spark_session import get_or_create_spark_session
from .gates import Gate
from .circuits import (
    generate_ghz_circuit,
    generate_qft_circuit,
    generate_qpe_circuit,
    generate_w_circuit,
)
from .frontend import circuit_dict_to_gates
from .state_manager import StateManager
from .gate_applicator import GateApplicator
from .gate_batcher import GateBatcher, GateBatch
from .metadata_store import MetadataStore
from .checkpoint_manager import CheckpointManager
from .recovery_manager import RecoveryManager


@dataclass
class SimulationResult:
    """
    Result of a quantum circuit simulation.
    
    Attributes:
        final_state_df: Final quantum state as DataFrame.
        final_version: Final state version number.
        n_qubits: Number of qubits in the circuit.
        n_gates: Total number of gates applied.
        n_batches: Number of batches processed.
        elapsed_time: Total simulation time in seconds.
        run_id: Unique identifier for this run.
    """
    final_state_df: DataFrame
    final_version: int
    n_qubits: int
    n_gates: int
    n_batches: int
    elapsed_time: float
    run_id: str


class SparkQuantumDriver:
    """
    Main driver for Spark-based quantum circuit simulation.
    
    Coordinates all components:
    - StateManager: Parquet-based state storage
    - GateApplicator: DataFrame gate transformations
    - GateBatcher: Gate batching for efficiency
    - MetadataStore: WAL and checkpoint metadata
    - CheckpointManager: State checkpointing
    - RecoveryManager: Crash recovery
    """
    
    def __init__(self, config: Optional[SimulatorConfig] = None):
        """
        Initialize the driver.
        
        Args:
            config: Simulation configuration. Uses defaults if not provided.
        """
        self.config = config or SimulatorConfig()
        self.config.ensure_paths()
        
        # Initialize Spark
        self.spark = get_or_create_spark_session(self.config)
        
        # Initialize components
        self.state_manager = StateManager(self.spark, self.config)
        self.gate_applicator = GateApplicator(
            self.spark, 
            num_partitions=self.config.spark_shuffle_partitions
        )
        self.gate_batcher = GateBatcher(self.config.batch_size)
        self.metadata_store = MetadataStore(self.config)
        self.checkpoint_manager = CheckpointManager(
            self.spark, 
            self.config,
            self.state_manager,
            self.metadata_store,
        )
        self.recovery_manager = RecoveryManager(
            self.spark,
            self.config,
            self.state_manager,
            self.metadata_store,
            self.checkpoint_manager,
        )
    
    def run_circuit(
        self,
        circuit_dict: Dict[str, Any],
        resume: bool = True,
    ) -> SimulationResult:
        """
        Run a quantum circuit simulation.
        
        Args:
            circuit_dict: Circuit definition with 'number_of_qubits' and 'gates'.
            resume: If True, attempt to resume from checkpoint.
            
        Returns:
            SimulationResult with final state and metadata.
        """
        start_time = time.time()
        
        # Parse circuit
        n_qubits, gates = circuit_dict_to_gates(circuit_dict)
        total_gates = len(gates)
        
        # Register gate matrices for broadcast
        self.gate_applicator.register_gates(gates)
        
        # Determine starting point
        if resume:
            recovery_state = self.recovery_manager.recover(n_qubits)
            current_state = recovery_state.state_df
            current_version = recovery_state.state_version
            start_gate_seq = recovery_state.last_gate_seq + 1
        else:
            current_state = self.state_manager.initialize_state(n_qubits)
            current_version = 0
            start_gate_seq = 0
        
        # Get remaining gates and batch them
        remaining_gates = gates[start_gate_seq:]
        batches = self.gate_batcher.create_batches(remaining_gates, start_gate_seq)
        
        n_batches_processed = 0
        
        # Process each batch
        for batch in batches:
            current_state, current_version = self._process_batch(
                current_state,
                current_version,
                batch,
            )
            n_batches_processed += 1
        
        elapsed_time = time.time() - start_time
        
        return SimulationResult(
            final_state_df=current_state,
            final_version=current_version,
            n_qubits=n_qubits,
            n_gates=total_gates,
            n_batches=n_batches_processed,
            elapsed_time=elapsed_time,
            run_id=self.config.run_id,
        )
    
    def _process_batch(
        self,
        state_df: DataFrame,
        version_in: int,
        batch: GateBatch,
    ) -> tuple[DataFrame, int]:
        """
        Process a single batch of gates.
        
        Flow:
        1. Write WAL PENDING
        2. Build lazy DataFrame plan
        3. Trigger ACTION (save to Parquet)
        4. Create checkpoint + mark WAL COMMITTED
        
        Args:
            state_df: Input state DataFrame.
            version_in: Input state version.
            batch: Batch of gates to apply.
            
        Returns:
            Tuple of (output_state_df, output_version).
        """
        version_out = version_in + 1
        
        # Step 1: Write WAL PENDING
        wal_id = self.metadata_store.wal_create_pending(
            run_id=self.config.run_id,
            gate_start=batch.start_seq,
            gate_end=batch.end_seq,
            state_version_in=version_in,
            state_version_out=version_out,
        )
        
        try:
            # Step 2: Build lazy DataFrame plan
            output_state = self.gate_applicator.apply_gates(state_df, batch.gates)
            
            # Step 3: Trigger ACTION - write to Parquet
            state_path = self.state_manager.save_state(output_state, version_out)
            
            # Step 4: Create checkpoint record (state already saved above)
            self.checkpoint_manager.create_checkpoint(
                state_version=version_out,
                last_gate_seq=batch.end_seq - 1,
                state_path=state_path,
            )
            
            self.metadata_store.wal_mark_committed(wal_id)
            
            # Load back the saved state for next batch (ensures we're reading from disk)
            output_state = self.state_manager.load_state(version_out)
            
            return output_state, version_out
            
        except Exception as e:
            # Mark WAL as failed and re-raise
            self.metadata_store.wal_mark_failed(wal_id)
            raise
    
    def run_ghz(self, n_qubits: int, **kwargs) -> SimulationResult:
        """Run GHZ circuit simulation."""
        circuit = generate_ghz_circuit(n_qubits)
        return self.run_circuit(circuit, **kwargs)
    
    def run_qft(self, n_qubits: int, **kwargs) -> SimulationResult:
        """Run QFT circuit simulation."""
        circuit = generate_qft_circuit(n_qubits)
        return self.run_circuit(circuit, **kwargs)
    
    def run_qpe(self, n_qubits: int, **kwargs) -> SimulationResult:
        """Run QPE circuit simulation."""
        circuit = generate_qpe_circuit(n_qubits)
        return self.run_circuit(circuit, **kwargs)
    
    def run_w(self, n_qubits: int, **kwargs) -> SimulationResult:
        """Run W-state circuit simulation."""
        circuit = generate_w_circuit(n_qubits)
        return self.run_circuit(circuit, **kwargs)
    
    def get_state_vector(self, result: SimulationResult):
        """
        Get the final state as a numpy array.
        
        Args:
            result: Simulation result.
            
        Returns:
            Complex numpy array of shape (2^n_qubits,).
        """
        return self.state_manager.get_state_as_array(
            result.final_state_df, 
            result.n_qubits
        )
    
    def get_state_dict(self, result: SimulationResult) -> Dict[int, complex]:
        """
        Get the final state as a sparse dictionary.
        
        Args:
            result: Simulation result.
            
        Returns:
            Dictionary mapping basis state index to complex amplitude.
        """
        return self.state_manager.get_state_as_dict(result.final_state_df)
    
    def cleanup(self):
        """Release resources."""
        self.gate_applicator.cleanup()
        self.metadata_store.close()
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        return False


# Convenience function for simple usage
def run_circuit(
    circuit_dict: Dict[str, Any],
    config: Optional[SimulatorConfig] = None,
) -> SimulationResult:
    """
    Run a quantum circuit simulation.
    
    Args:
        circuit_dict: Circuit definition.
        config: Optional configuration.
        
    Returns:
        SimulationResult with final state.
    """
    with SparkQuantumDriver(config) as driver:
        return driver.run_circuit(circuit_dict)
