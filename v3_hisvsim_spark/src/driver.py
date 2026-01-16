"""
Main driver for HiSVSIM + Spark integration.

Orchestrates:
1. Circuit partitioning (HiSVSIM)
2. Parallel execution (Spark)
3. State merging
4. WAL, checkpoints, and recovery (fault tolerance)

PROPER ARCHITECTURE:
- WAL PENDING → process → checkpoint → WAL COMMITTED
- Recovery from checkpoints
- Level-based parallelism with fault tolerance

IMPROVEMENTS:
- Proper logging (replaces print statements)
- State caching cleanup (unpersist old states)
- Adaptive checkpointing (based on state size and progress)
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass
import time
import logging

from utils.logging_config import get_logger, setup_logging

# Import from v2_common (copied from v2_spark)
from v2_common import (
    config, spark_session, state_manager, gate_applicator, frontend,
    metadata_store, checkpoint_manager, recovery_manager
)

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F
from pyspark import SparkContext

SimulatorConfig = config.SimulatorConfig
get_or_create_spark_session = spark_session.get_or_create_spark_session
StateManager = state_manager.StateManager
GateApplicator = gate_applicator.GateApplicator
MetadataStore = metadata_store.MetadataStore
CheckpointManager = checkpoint_manager.CheckpointManager
RecoveryManager = recovery_manager.RecoveryManager
circuit_dict_to_gates = frontend.circuit_dict_to_gates

from hisvsim.partition_adapter import HiSVSIMPartitionAdapter
from state_merger_module import StateMerger


@dataclass
class PartitionResult:
    """Result from simulating a partition."""
    partition_id: int
    state_df: DataFrame  # Spark DataFrame
    qubits_used: set[int]
    n_gates: int


@dataclass
class SimulationResult:
    """Final simulation result."""
    final_state_df: DataFrame
    n_qubits: int
    n_gates: int
    n_partitions: int
    elapsed_time: float
    run_id: str
    parallel_execution: bool = False


class SparkHiSVSIMDriver:
    """
    Driver for HiSVSIM + Spark quantum simulation.
    
    PROPER ARCHITECTURE with:
    - WAL (Write-Ahead Log) for durability
    - Checkpoints for recovery
    - Recovery Manager for crash recovery
    - Level-based parallelism
    - State caching with cleanup
    - Adaptive checkpointing
    - Proper logging
    - Optimized Spark operations
    """
    
    def __init__(self, config: SimulatorConfig, enable_parallel: bool = True):
        """
        Initialize the driver.
        
        Args:
            config: Simulator configuration.
            enable_parallel: If True, execute partitions in parallel (default: True).
        """
        self.config = config
        self.config.ensure_paths()  # Ensure all paths exist
        self.enable_parallel = enable_parallel
        self.spark = get_or_create_spark_session(config)
        self.state_manager = StateManager(self.spark, config)
        self.gate_applicator = GateApplicator(self.spark, config.spark_shuffle_partitions)
        # Use hybrid partitioning strategy (balance load and locality)
        self.partition_adapter = HiSVSIMPartitionAdapter(strategy="hybrid")
        self.state_merger = StateMerger(self.spark, config)
        
        # Fault tolerance components
        self.metadata_store = MetadataStore(config)
        self.checkpoint_manager = CheckpointManager(
            self.spark, config, self.state_manager, self.metadata_store
        )
        self.recovery_manager = RecoveryManager(
            self.spark, config, self.state_manager, 
            self.metadata_store, self.checkpoint_manager
        )
        
        # Initialize logging (if not already initialized)
        try:
            logging.getLogger('quantum_simulator').handlers
        except:
            setup_logging()
        
        # Logging
        self.logger = get_logger(__name__)
        
        # Track cached states for cleanup
        self._cached_states: List[DataFrame] = []
        
        # Track checkpoint timing for adaptive checkpointing
        self._last_checkpoint_time: Optional[float] = None
        self._gates_since_checkpoint: int = 0
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Clean up resources."""
        if hasattr(self, 'metadata_store'):
            self.metadata_store.close()
    
    def run_circuit(
        self, 
        circuit_dict: Dict, 
        n_partitions: int = None,
        enable_parallel: bool = None,
        resume: bool = True
    ) -> SimulationResult:
        """
        Run a quantum circuit simulation with proper architecture.
        
        Flow:
        1. Recover from checkpoint (if resume=True)
        2. Process gates in batches/levels with WAL/checkpoints
        3. Return final state
        
        Args:
            circuit_dict: Circuit dictionary with 'gates' and 'number_of_qubits'.
            n_partitions: Number of partitions (optional).
            enable_parallel: Override parallel execution setting.
            resume: If True, attempt to resume from checkpoint.
            
        Returns:
            SimulationResult with final state.
        """
        start_time = time.time()
        
        n_qubits, gates = circuit_dict_to_gates(circuit_dict)
        total_gates = len(gates)
        
        # Step 1: Recovery (if resume=True)
        gates_to_apply = gates  # Default: apply all gates
        start_gate_seq = 0
        current_state = self.state_manager.initialize_state(n_qubits)
        state_version = 0
        
        # Reset checkpoint tracking
        self._last_checkpoint_time = time.time()
        self._gates_since_checkpoint = 0
        
        if resume:
            recovery_state = self.recovery_manager.recover(n_qubits)
            if recovery_state.checkpoint_record is not None:
                self.logger.info(
                    "Recovered from checkpoint: state_version=%d, last_gate_seq=%d",
                    recovery_state.state_version,
                    recovery_state.last_gate_seq
                )
                start_gate_seq = recovery_state.last_gate_seq + 1
                current_state = recovery_state.state_df
                state_version = recovery_state.state_version
                
                # Check if simulation already complete
                if start_gate_seq >= total_gates:
                    self.logger.info("Simulation already complete!")
                    final_state = current_state
                    n_partitions_used = 1
                    elapsed_time = time.time() - start_time
                    return SimulationResult(
                        final_state_df=final_state,
                        n_qubits=n_qubits,
                        n_gates=total_gates,
                        n_partitions=n_partitions_used,
                        elapsed_time=elapsed_time,
                        run_id=self.config.run_id,
                        parallel_execution=False,
                    )
                else:
                    # Resume from checkpoint
                    gates_to_apply = gates[start_gate_seq:]
                    self.logger.info("Resuming from gate %d/%d", start_gate_seq, total_gates)
        
        # Use parallel execution if enabled
        use_parallel = enable_parallel if enable_parallel is not None else self.enable_parallel
        
        # Step 2: Process remaining gates
        if len(gates_to_apply) == 0:
            # Already complete
            final_state = current_state
            n_partitions_used = 1
        elif not use_parallel:
            # Sequential mode: apply gates with WAL/checkpoints
            self.logger.info("Sequential execution with WAL/checkpoints...")
            final_state, final_version = self._simulate_sequential_with_wal(
                current_state, gates_to_apply, state_version, start_gate_seq
            )
            n_partitions_used = 1
        else:
            # Parallel mode: use level-based parallelism with WAL/checkpoints
            if n_partitions is None:
                sc = self.spark.sparkContext
                n_partitions = max(1, sc.defaultParallelism // 2)
            
            self.logger.info("Parallel execution with WAL/checkpoints...")
            final_state, final_version = self._simulate_parallel_with_wal(
                current_state, gates_to_apply, n_qubits, state_version, start_gate_seq
            )
            n_partitions_used = n_partitions
        
        elapsed_time = time.time() - start_time
        
        return SimulationResult(
            final_state_df=final_state,
            n_qubits=n_qubits,
            n_gates=total_gates,
            n_partitions=n_partitions_used,
            elapsed_time=elapsed_time,
            run_id=self.config.run_id,
            parallel_execution=use_parallel,
        )
    
    def _simulate_sequential_with_wal(
        self,
        initial_state: DataFrame,
        gates: List,
        start_version: int,
        start_gate_seq: int
    ) -> Tuple[DataFrame, int]:
        """
        Sequential simulation with WAL and checkpoints.
        
        Flow per batch:
        1. Write WAL PENDING
        2. Apply gates
        3. Save state to Parquet
        4. Create checkpoint (adaptive)
        5. Mark WAL COMMITTED
        """
        self.logger.info("Simulating gates sequentially with WAL/checkpoints...")
        
        # Register all gates
        self.gate_applicator.register_gates(gates)
        
        current_state = initial_state
        current_version = start_version
        
        # Process gates in batches
        batch_size = self.config.batch_size
        for batch_start in range(0, len(gates), batch_size):
            batch_end = min(batch_start + batch_size, len(gates))
            batch_gates = gates[batch_start:batch_end]
            gate_start_seq = start_gate_seq + batch_start
            gate_end_seq = start_gate_seq + batch_end
            
            version_in = current_version
            version_out = current_version + 1
            
            # Step 1: Write WAL PENDING
            wal_id = self.metadata_store.wal_create_pending(
                run_id=self.config.run_id,
                gate_start=gate_start_seq,
                gate_end=gate_end_seq,
                state_version_in=version_in,
                state_version_out=version_out,
            )
            
            try:
                # Step 2: Build lazy DataFrame plan (apply all gates in batch)
                # This builds a lazy plan - no action triggered yet
                output_state = self.gate_applicator.apply_gates(current_state, batch_gates)
                
                # Step 3: Trigger ACTION - write state snapshot to Parquet
                # This forces execution of the lazy DataFrame plan
                state_path = self.state_manager.save_state(output_state, version_out)
                
                # Step 4: Create checkpoint (adaptive - only if needed)
                should_checkpoint = self._should_checkpoint(
                    output_state, 
                    len(batch_gates),
                    version_out
                )
                
                if should_checkpoint:
                    self.checkpoint_manager.create_checkpoint(
                        state_version=version_out,
                        last_gate_seq=gate_end_seq - 1,
                        state_path=state_path,
                    )
                    self._last_checkpoint_time = time.time()
                    self._gates_since_checkpoint = 0
                    self.logger.debug("Created checkpoint v%d", version_out)
                else:
                    self._gates_since_checkpoint += len(batch_gates)
                
                # Step 5: Mark WAL COMMITTED
                self.metadata_store.wal_mark_committed(wal_id)
                
                # IMPROVEMENT 1: Unpersist old cached state before loading new one
                if current_state.is_cached:
                    current_state.unpersist(blocking=False)
                
                # Load state back from disk for next batch (ensures we read from durable storage)
                current_state = self.state_manager.load_state(version_out)
                current_version = version_out
                
                if batch_end % 10 == 0 or batch_end == len(gates):
                    self.logger.info(
                        "Processed %d/%d gates (version %d)%s",
                        batch_end, len(gates), version_out,
                        " [checkpoint]" if should_checkpoint else ""
                    )
                    
            except Exception as e:
                # Mark WAL as failed and re-raise
                self.logger.error("Failed to process batch %d-%d: %s", gate_start_seq, gate_end_seq, e, exc_info=True)
                self.metadata_store.wal_mark_failed(wal_id)
                raise
        
        return current_state, current_version
    
    def _simulate_parallel_with_wal(
        self,
        initial_state: DataFrame,
        gates: List,
        n_qubits: int,
        start_version: int,
        start_gate_seq: int
    ) -> Tuple[DataFrame, int]:
        """
        Parallel simulation with WAL and checkpoints using level-based parallelism.
        
        Flow per level:
        1. Write WAL PENDING for level
        2. Apply gates in level
        3. Save state to Parquet
        4. Create checkpoint (adaptive)
        5. Mark WAL COMMITTED
        """
        self.logger.info("Simulating with level-based parallelism + WAL/checkpoints...")
        
        # Register all gates
        self.gate_applicator.register_gates(gates)
        
        # Build circuit graph to find topological levels
        G = self.partition_adapter._build_circuit_graph(gates)
        levels = self.partition_adapter._topological_levels(G, gates)
        
        self.logger.info("Found %d topological levels", len(levels))
        
        current_state = initial_state
        current_version = start_version
        gate_idx = 0
        
        # Process levels with WAL/checkpoints
        for level_idx, level in enumerate(levels):
            num_gates = len(level)
            level_start_seq = start_gate_seq + gate_idx
            level_end_seq = start_gate_seq + gate_idx + num_gates
            
            version_in = current_version
            version_out = current_version + 1
            
            # Step 1: Write WAL PENDING
            wal_id = self.metadata_store.wal_create_pending(
                run_id=self.config.run_id,
                gate_start=level_start_seq,
                gate_end=level_end_seq,
                state_version_in=version_in,
                state_version_out=version_out,
            )
            
            try:
                # Step 2: Build lazy DataFrame plan (apply all gates in level)
                # This builds a lazy plan - no action triggered yet
                self.logger.debug(
                    "Processing level %d/%d (%d gates): gates %d-%d",
                    level_idx + 1, len(levels), num_gates, level_start_seq, level_end_seq - 1
                )
                
                level_gates = [gates[gate_idx_in_level] for gate_idx_in_level in level]
                output_state = self.gate_applicator.apply_gates(current_state, level_gates)
                
                # Step 3: Trigger ACTION - write state snapshot to Parquet
                # This forces execution of the lazy DataFrame plan
                state_path = self.state_manager.save_state(output_state, version_out)
                
                # Step 4: Create checkpoint (adaptive - only if needed)
                should_checkpoint = self._should_checkpoint(
                    output_state,
                    num_gates,
                    version_out
                )
                
                if should_checkpoint:
                    self.checkpoint_manager.create_checkpoint(
                        state_version=version_out,
                        last_gate_seq=level_end_seq - 1,
                        state_path=state_path,
                    )
                    self._last_checkpoint_time = time.time()
                    self._gates_since_checkpoint = 0
                    self.logger.debug("Created checkpoint v%d", version_out)
                else:
                    self._gates_since_checkpoint += num_gates
                
                # Step 5: Mark WAL COMMITTED
                self.metadata_store.wal_mark_committed(wal_id)
                
                # IMPROVEMENT 1: Unpersist old cached state before loading new one
                if current_state.is_cached:
                    current_state.unpersist(blocking=False)
                
                # Load state back from disk for next level (ensures we read from durable storage)
                current_state = self.state_manager.load_state(version_out)
                current_version = version_out
                gate_idx += num_gates
                
                if (level_idx + 1) % 5 == 0 or level_idx == len(levels) - 1:
                    self.logger.info(
                        "Completed level %d/%d (%d gates)%s",
                        level_idx + 1, len(levels), num_gates,
                        " [checkpoint]" if should_checkpoint else ""
                    )
                
            except Exception as e:
                # Mark WAL as failed and re-raise
                self.logger.error(
                    "Failed to process level %d (gates %d-%d): %s",
                    level_idx + 1, level_start_seq, level_end_seq - 1, e,
                    exc_info=True
                )
                self.metadata_store.wal_mark_failed(wal_id)
                raise
        
        return current_state, current_version
    
    def _check_gates_independent(self, gates: List, gate_indices: List[int]) -> bool:
        """
        Check if gates in a level operate on independent qubits.
        
        Gates are independent if they don't share any qubits.
        """
        qubit_sets = [set(gates[idx].qubits) for idx in gate_indices]
        
        # Check for any overlap
        for i, qubits_i in enumerate(qubit_sets):
            for j, qubits_j in enumerate(qubit_sets):
                if i < j and qubits_i & qubits_j:
                    return False
        
        return True
    
    def _simulate_partitions_parallel_optimized(
        self,
        partition_circuits: List[Dict],
        gates: List,
        n_qubits: int
    ) -> DataFrame:
        """
        OPTIMIZED parallel simulation with:
        - Level-based parallelism
        - Parallel gate application within levels (when gates are independent)
        - State caching
        """
        self.logger.info("Simulating with OPTIMIZED level-based parallelism...")
        
        # Register all gates first
        self.gate_applicator.register_gates(gates)
        
        # Build circuit graph to find topological levels
        G = self.partition_adapter._build_circuit_graph(gates)
        levels = self.partition_adapter._topological_levels(G, gates)
        
        self.logger.info("Found %d topological levels", len(levels))
        
        # Start with initial state
        current_state = self.state_manager.initialize_state(n_qubits)
        
        # OPTIMIZATION: Cache state for reuse
        current_state.cache()
        
        # Apply gates level by level
        for level_idx, level in enumerate(levels):
            num_gates = len(level)
            self.logger.debug("Processing level %d/%d (%d gates)...", level_idx + 1, len(levels), num_gates)
            
            if num_gates == 1:
                # Single gate: apply directly
                gate = gates[level[0]]
                current_state = self.gate_applicator.apply_gate(current_state, gate)
            else:
                # Multiple gates in level
                # OPTIMIZATION: Check if gates are independent
                if self._check_gates_independent(gates, level):
                    # Gates are independent: can apply in parallel conceptually
                    # However, since they all operate on the same state DataFrame,
                    # we still apply sequentially but Spark parallelizes the operations
                    # Future: Could split state and apply gates in parallel, then merge
                    for gate_idx in level:
                        gate = gates[gate_idx]
                        current_state = self.gate_applicator.apply_gate(current_state, gate)
                else:
                    # Gates share qubits: must apply sequentially
                    for gate_idx in level:
                        gate = gates[gate_idx]
                        current_state = self.gate_applicator.apply_gate(current_state, gate)
            
            # IMPROVEMENT 1: Cache state after each level, unpersist previous
            if level_idx > 0 and len(self._cached_states) > 0:
                # Unpersist previous state to free memory
                prev_state = self._cached_states[-1]
                if prev_state.is_cached:
                    prev_state.unpersist(blocking=False)
                    self._cached_states.pop()
            
            # Cache new state
            current_state.cache()
            self._cached_states.append(current_state)
            
            # OPTIMIZATION: Repartition for better distribution
            # Only repartition if we have multiple shuffle partitions and more levels to go
            if self.config.spark_shuffle_partitions > 1 and level_idx < len(levels) - 1:
                current_state = current_state.repartition(
                    self.config.spark_shuffle_partitions, "idx"
                )
        
        return current_state
    
    def _simulate_partitions_sequential(
        self,
        partition_circuits: List[Dict],
        gates: List,
        n_qubits: int
    ) -> DataFrame:
        """
        Simulate partitions sequentially (legacy - use _simulate_sequential_optimized instead).
        
        This preserves correctness but doesn't use parallelism.
        """
        self.logger.info("Simulating partitions sequentially...")
        
        # Start with initial state
        current_state = self.state_manager.initialize_state(n_qubits)
        
        # Register all gates first
        self.gate_applicator.register_gates(gates)
        
        # Apply gates in their original order (not partition order)
        gate_list = []
        for partition_id, partition_circuit in enumerate(partition_circuits):
            _, partition_gates = circuit_dict_to_gates(partition_circuit)
            for gate in partition_gates:
                # Find original index by matching gate properties
                matched = False
                for orig_idx, orig_gate in enumerate(gates):
                    if (orig_gate.gate_name == gate.gate_name and 
                        orig_gate.qubits == gate.qubits and
                        orig_idx not in [g[0] for g in gate_list]):
                        gate_list.append((orig_idx, partition_id, gate))
                        matched = True
                        break
                
                if not matched:
                    raise ValueError(
                        f"Could not match gate {gate.gate_name} on qubits {gate.qubits} "
                        f"to original gates. This indicates a bug in partitioning or gate conversion."
                    )
        
        # Sort by original index and apply
        gate_list.sort(key=lambda x: x[0])
        current_partition = None
        for orig_idx, partition_id, gate in gate_list:
            if partition_id != current_partition:
                self.logger.debug("Simulating partition %d...", partition_id)
                current_partition = partition_id
            current_state = self.gate_applicator.apply_gate(current_state, gate)
        
        return current_state
    
    def get_state_vector(self, result: SimulationResult) -> 'np.ndarray':
        """Get final state as NumPy array."""
        return self.state_manager.get_state_as_array(
            result.final_state_df, 
            result.n_qubits
        )
    
    def _should_checkpoint(
        self,
        state_df: DataFrame,
        gates_in_batch: int,
        version_out: int
    ) -> bool:
        """
        Determine if checkpoint should be created (adaptive checkpointing).
        
        Checkpoints are created if:
        1. State size exceeds threshold (expensive to recompute)
        2. Many gates since last checkpoint
        3. Minimum time interval has passed
        4. First checkpoint (version 1)
        
        Args:
            state_df: Current state DataFrame.
            gates_in_batch: Number of gates in current batch/level.
            version_out: Output state version.
            
        Returns:
            True if checkpoint should be created.
        """
        # Always checkpoint first version
        if version_out == 1:
            return True
        
        # Check state size (expensive to recompute large states)
        try:
            state_size = state_df.count()
            if state_size > self.config.checkpoint_threshold_size:
                self.logger.debug(
                    "Checkpoint triggered: state size %d > threshold %d",
                    state_size, self.config.checkpoint_threshold_size
                )
                return True
        except Exception:
            # If count() fails, default to checkpointing
            self.logger.warning("Could not determine state size, creating checkpoint")
            return True
        
        # Check gates since last checkpoint
        if self._gates_since_checkpoint + gates_in_batch >= self.config.checkpoint_every_n_gates:
            self.logger.debug(
                "Checkpoint triggered: %d gates since last checkpoint >= %d",
                self._gates_since_checkpoint + gates_in_batch,
                self.config.checkpoint_every_n_gates
            )
            return True
        
        # Check time since last checkpoint
        if self._last_checkpoint_time is not None:
            time_since_checkpoint = time.time() - self._last_checkpoint_time
            if time_since_checkpoint >= self.config.checkpoint_min_interval_seconds:
                self.logger.debug(
                    "Checkpoint triggered: %.1fs since last checkpoint >= %.1fs",
                    time_since_checkpoint,
                    self.config.checkpoint_min_interval_seconds
                )
                return True
        
        # Default: checkpoint every N batches (backward compatibility)
        if version_out % self.config.checkpoint_every_n_batches == 0:
            return True
        
        return False
    
    def cleanup(self):
        """Clean up resources."""
        # IMPROVEMENT 1: Unpersist all cached states
        for cached_state in self._cached_states:
            if cached_state.is_cached:
                try:
                    cached_state.unpersist(blocking=False)
                except Exception:
                    pass  # Ignore errors during cleanup
        self._cached_states.clear()
        
        # Close metadata store
        if hasattr(self, 'metadata_store'):
            self.metadata_store.close()
