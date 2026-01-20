"""
Main Driver for Quantum Circuit Simulation with TRUE Parallel Gate Execution.

This is the MAIN driver for v3 - uses tensor product fusion for parallel gates.

Key Features:
1. TRUE parallel gate execution - independent gates fused into single transformations
2. Level-based circuit partitioning (HiSVSIM-style)
3. WAL + checkpoints for fault tolerance
4. Sparse state representation for scalability
5. Distributed state vector processing via Spark

Architecture:
- Circuit → Topological Levels → Parallel Gate Groups → Spark Execution
- Independent gates (non-overlapping qubits) are fused via tensor products
- Sequential gates applied one at a time
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
from dataclasses import dataclass
import time
import logging
import uuid

from pyspark.sql import SparkSession, DataFrame
from pyspark.sql import functions as F

# Import components
from v2_common import (
    config, spark_session, state_manager, frontend,
    metadata_store, checkpoint_manager, recovery_manager
)
from v2_common.config import SimulatorConfig
from v2_common.spark_session import get_or_create_spark_session
from v2_common.state_manager import StateManager
from v2_common.frontend import circuit_dict_to_gates
from v2_common.metadata_store import MetadataStore
from v2_common.checkpoint_manager import CheckpointManager
from v2_common.recovery_manager import RecoveryManager

from hisvsim.partition_adapter import HiSVSIMPartitionAdapter
from parallel_gate_applicator import ParallelGateApplicator

# Setup logging
try:
    from utils.logging_config import get_logger, setup_logging
    setup_logging()
except:
    logging.basicConfig(level=logging.INFO)
    def get_logger(name): return logging.getLogger(name)


@dataclass
class SimulationResult:
    """Result of a quantum circuit simulation."""
    final_state_df: DataFrame
    n_qubits: int
    n_gates: int
    n_levels: int
    parallel_groups: List[int]
    elapsed_time: float
    run_id: str


class SparkHiSVSIMDriver:
    """
    Main driver for quantum simulation with TRUE parallel gate execution.
    
    Key innovation: Independent gates are fused into single transformations
    using tensor products. For example:
    
    H(0), H(1), H(2), H(3) → 1 combined transformation (H⊗H⊗H⊗H)
    
    instead of 4 sequential transformations.
    
    Features:
    - TRUE parallel gate execution for independent gates
    - Level-based circuit partitioning
    - WAL and checkpointing for fault tolerance
    - Recovery from crashes
    - Sparse state representation
    - Distributed processing via Spark
    """
    
    def __init__(self, config: SimulatorConfig, enable_parallel: bool = True):
        """
        Initialize the driver.
        
        Args:
            config: Simulator configuration
            enable_parallel: If True, use parallel gate execution (default: True)
        """
        self.config = config
        self.config.ensure_paths()
        self.enable_parallel = enable_parallel
        
        # Initialize Spark
        self.spark = get_or_create_spark_session(config)
        
        # Initialize components
        self.state_manager = StateManager(self.spark, config)
        self.gate_applicator = ParallelGateApplicator(
            self.spark, 
            config.spark_shuffle_partitions
        )
        self.partition_adapter = HiSVSIMPartitionAdapter(strategy="hybrid")
        
        # Fault tolerance components
        self.metadata_store = MetadataStore(config)
        self.checkpoint_manager = CheckpointManager(
            self.spark, config, self.state_manager, self.metadata_store
        )
        self.recovery_manager = RecoveryManager(
            self.spark, config, self.state_manager,
            self.metadata_store, self.checkpoint_manager
        )
        
        # Logging
        self.logger = get_logger(__name__)
        
        # Checkpoint tracking
        self._last_checkpoint_time: Optional[float] = None
        self._gates_since_checkpoint: int = 0
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
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
        Run a quantum circuit simulation with TRUE parallel gate execution.
        
        Args:
            circuit_dict: Circuit with 'gates' and 'number_of_qubits'
            n_partitions: Number of Spark partitions (optional)
            enable_parallel: Override parallel execution setting
            resume: If True, attempt to resume from checkpoint
            
        Returns:
            SimulationResult with final state and statistics
        """
        start_time = time.time()
        
        n_qubits, gates = circuit_dict_to_gates(circuit_dict)
        total_gates = len(gates)
        
        # Register gates
        self.gate_applicator.register_gates(gates)
        
        # Recovery (if resume=True)
        gates_to_apply = gates
        start_gate_seq = 0
        current_state = self.state_manager.initialize_state(n_qubits)
        state_version = 0
        
        self._last_checkpoint_time = time.time()
        self._gates_since_checkpoint = 0
        
        if resume:
            recovery_state = self.recovery_manager.recover(n_qubits)
            if recovery_state.checkpoint_record is not None:
                self.logger.info(
                    "Recovered from checkpoint: state_version=%d, last_gate_seq=%d",
                    recovery_state.state_version, recovery_state.last_gate_seq
                )
                start_gate_seq = recovery_state.last_gate_seq + 1
                current_state = recovery_state.state_df
                state_version = recovery_state.state_version
                
                if start_gate_seq >= total_gates:
                    self.logger.info("Simulation already complete!")
                    return SimulationResult(
                        final_state_df=current_state,
                        n_qubits=n_qubits,
                        n_gates=total_gates,
                        n_levels=0,
                        parallel_groups=[],
                        elapsed_time=time.time() - start_time,
                        run_id=self.config.run_id,
                    )
                else:
                    gates_to_apply = gates[start_gate_seq:]
                    self.logger.info("Resuming from gate %d/%d", start_gate_seq, total_gates)
        
        # Use parallel execution
        use_parallel = enable_parallel if enable_parallel is not None else self.enable_parallel
        
        # Process gates
        if len(gates_to_apply) == 0:
            final_state = current_state
            n_levels = 0
            parallel_groups = []
        else:
            final_state, state_version, n_levels, parallel_groups = self._simulate_with_parallelism(
                current_state, gates_to_apply, n_qubits, state_version, start_gate_seq, use_parallel
            )
        
        elapsed_time = time.time() - start_time
        
        return SimulationResult(
            final_state_df=final_state,
            n_qubits=n_qubits,
            n_gates=total_gates,
            n_levels=n_levels,
            parallel_groups=parallel_groups,
            elapsed_time=elapsed_time,
            run_id=self.config.run_id,
        )
    
    def _simulate_with_parallelism(
        self,
        initial_state: DataFrame,
        gates: List,
        n_qubits: int,
        start_version: int,
        start_gate_seq: int,
        use_parallel: bool
    ) -> Tuple[DataFrame, int, int, List[int]]:
        """
        Simulate with TRUE parallel gate execution.
        
        Flow:
        1. Build topological levels
        2. For each level, group independent gates
        3. Apply parallel groups with tensor product fusion
        4. Checkpoint periodically
        """
        self.logger.info("Simulating with TRUE parallel gate execution...")
        
        # Build topological levels
        G = self.partition_adapter._build_circuit_graph(gates)
        levels = self.partition_adapter._topological_levels(G, gates)
        
        self.logger.info("Found %d topological levels", len(levels))
        
        current_state = initial_state
        current_version = start_version
        gate_idx = 0
        all_parallel_groups = []
        
        # Process each level
        for level_idx, level in enumerate(levels):
            level_gates = [gates[i] for i in level]
            num_gates = len(level_gates)
            level_start_seq = start_gate_seq + gate_idx
            level_end_seq = start_gate_seq + gate_idx + num_gates
            
            version_in = current_version
            version_out = current_version + 1
            
            # Write WAL PENDING
            wal_id = self.metadata_store.wal_create_pending(
                run_id=self.config.run_id,
                gate_start=level_start_seq,
                gate_end=level_end_seq,
                state_version_in=version_in,
                state_version_out=version_out,
            )
            
            try:
                # Group independent gates for parallel execution
                if use_parallel:
                    groups = self._group_independent_gates(level_gates)
                else:
                    groups = [[g] for g in level_gates]
                
                # Apply groups
                for group in groups:
                    all_parallel_groups.append(len(group))
                    
                    if len(group) == 1:
                        current_state = self.gate_applicator._apply_single_gate(
                            current_state, group[0]
                        )
                    else:
                        # TRUE PARALLEL: Apply independent gates together!
                        current_state = self.gate_applicator.apply_gates_parallel(
                            current_state, group
                        )
                
                # Save state
                state_path = self.state_manager.save_state(current_state, version_out)
                
                # Checkpoint if needed
                should_checkpoint = self._should_checkpoint(
                    current_state, num_gates, version_out
                )
                
                if should_checkpoint:
                    self.checkpoint_manager.create_checkpoint(
                        state_version=version_out,
                        last_gate_seq=level_end_seq - 1,
                        state_path=state_path,
                    )
                    self._last_checkpoint_time = time.time()
                    self._gates_since_checkpoint = 0
                else:
                    self._gates_since_checkpoint += num_gates
                
                # Mark WAL COMMITTED
                self.metadata_store.wal_mark_committed(wal_id)
                
                # Load state from disk for next level
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
                self.logger.error(
                    "Failed to process level %d: %s", level_idx + 1, e, exc_info=True
                )
                self.metadata_store.wal_mark_failed(wal_id)
                raise
        
        return current_state, current_version, len(levels), all_parallel_groups
    
    def _group_independent_gates(self, gates: List) -> List[List]:
        """
        Group gates by qubit independence.
        
        Gates operating on non-overlapping qubits are grouped together
        for parallel execution via tensor product fusion.
        """
        if not gates:
            return []
        
        groups = []
        current_group = []
        current_qubits: Set[int] = set()
        
        for gate in gates:
            gate_qubits = set(gate.qubits)
            
            if gate_qubits & current_qubits:
                # Overlap - save current group, start new
                if current_group:
                    groups.append(current_group)
                current_group = [gate]
                current_qubits = gate_qubits
            else:
                # Independent - add to current group
                current_group.append(gate)
                current_qubits.update(gate_qubits)
        
        if current_group:
            groups.append(current_group)
        
        return groups
    
    def _should_checkpoint(
        self,
        state_df: DataFrame,
        gates_in_batch: int,
        version_out: int
    ) -> bool:
        """Determine if checkpoint should be created."""
        # Always checkpoint first version
        if version_out == 1:
            return True
        
        # Check gates since last checkpoint
        if self._gates_since_checkpoint + gates_in_batch >= self.config.checkpoint_every_n_gates:
            return True
        
        # Check time since last checkpoint
        if self._last_checkpoint_time is not None:
            time_since = time.time() - self._last_checkpoint_time
            if time_since >= self.config.checkpoint_min_interval_seconds:
                return True
        
        # Check periodic checkpointing
        if version_out % self.config.checkpoint_every_n_batches == 0:
            return True
        
        return False
    
    def get_state_vector(self, result: SimulationResult):
        """Get final state as numpy array."""
        return self.state_manager.get_state_as_array(
            result.final_state_df,
            result.n_qubits
        )
    
    def get_state_dict(self, result: SimulationResult) -> Dict[int, complex]:
        """Get final state as sparse dictionary."""
        return self.state_manager.get_state_as_dict(result.final_state_df)
    
    def cleanup(self):
        """Clean up resources."""
        if hasattr(self, 'metadata_store'):
            self.metadata_store.close()


# Convenience function
def run_circuit(
    circuit_dict: Dict,
    config: Optional[SimulatorConfig] = None,
) -> SimulationResult:
    """Run a quantum circuit simulation."""
    if config is None:
        config = SimulatorConfig()
    
    with SparkHiSVSIMDriver(config) as driver:
        return driver.run_circuit(circuit_dict)
