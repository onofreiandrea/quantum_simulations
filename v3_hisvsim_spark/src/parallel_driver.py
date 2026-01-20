"""
Parallel Driver - TRUE parallel gate execution.

This driver applies independent gates simultaneously using tensor products.
Gates in the same topological level that don't share qubits are fused
into a single combined transformation.

Example:
  Level with H(0), H(1), H(2), H(3) - all independent
  → OLD: 4 sequential transformations
  → NEW: 1 combined transformation (tensor product H⊗H⊗H⊗H)
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple, Set
import time
import uuid

from pyspark.sql import SparkSession, DataFrame

# Import components
from v2_common.config import SimulatorConfig
from v2_common.spark_session import get_or_create_spark_session
from v2_common.state_manager import StateManager
from v2_common.frontend import circuit_dict_to_gates
from hisvsim.partition_adapter import HiSVSIMPartitionAdapter
from parallel_gate_applicator import ParallelGateApplicator


class ParallelQuantumDriver:
    """
    Driver with TRUE parallel gate execution.
    
    Key difference from original driver:
    - Independent gates in the same level are applied SIMULTANEOUSLY
    - Uses tensor product fusion for parallel gates
    - Only sequential for gates that share qubits
    """
    
    def __init__(self, config: SimulatorConfig):
        self.config = config
        self.config.ensure_paths()
        self.spark = get_or_create_spark_session(config)
        self.state_manager = StateManager(self.spark, config)
        self.gate_applicator = ParallelGateApplicator(
            self.spark, 
            config.spark_shuffle_partitions
        )
        self.partition_adapter = HiSVSIMPartitionAdapter(strategy="hybrid")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        pass
    
    def run_circuit(self, circuit_dict: Dict) -> Dict:
        """
        Run circuit with TRUE parallel gate execution.
        
        Returns dict with:
          - final_state: DataFrame
          - n_qubits: int
          - n_gates: int
          - n_levels: int
          - parallel_groups: list of group sizes
          - elapsed_time: float
        """
        start_time = time.time()
        
        n_qubits, gates = circuit_dict_to_gates(circuit_dict)
        
        # Register gates
        self.gate_applicator.register_gates(gates)
        
        # Build topological levels
        G = self.partition_adapter._build_circuit_graph(gates)
        levels = self.partition_adapter._topological_levels(G, gates)
        
        # Initialize state
        current_state = self.state_manager.initialize_state(n_qubits)
        
        parallel_groups = []
        
        # Process each level with TRUE parallelism
        for level_idx, level in enumerate(levels):
            level_gates = [gates[i] for i in level]
            
            # Group independent gates
            groups = self._group_independent_gates(level_gates)
            
            for group in groups:
                parallel_groups.append(len(group))
                
                if len(group) == 1:
                    # Single gate
                    current_state = self.gate_applicator._apply_single_gate(
                        current_state, group[0]
                    )
                else:
                    # PARALLEL execution of independent gates!
                    current_state = self.gate_applicator.apply_gates_parallel(
                        current_state, group
                    )
        
        elapsed_time = time.time() - start_time
        
        return {
            "final_state": current_state,
            "n_qubits": n_qubits,
            "n_gates": len(gates),
            "n_levels": len(levels),
            "parallel_groups": parallel_groups,
            "elapsed_time": elapsed_time,
        }
    
    def _group_independent_gates(self, gates: List) -> List[List]:
        """Group gates by qubit independence."""
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
    
    def get_state_vector(self, result: Dict):
        """Get final state as numpy array."""
        import numpy as np
        return self.state_manager.get_state_as_array(
            result["final_state"], 
            result["n_qubits"]
        )
    
    def get_state_dict(self, result: Dict) -> Dict[int, complex]:
        """Get state as sparse dictionary."""
        return self.state_manager.get_state_as_dict(result["final_state"])
