"""
Circuit graph representation for HiSVSIM-style partitioning.

Based on: "Efficient Hierarchical State Vector Simulation of Quantum Circuits 
via Acyclic Graph Partitioning", IEEE CLUSTER 2022
https://github.com/pnnl/hisvsim.git
"""
from __future__ import annotations

from typing import List, Set, Dict, Tuple
from collections import defaultdict, deque

from .gates import Gate


class CircuitGraph:
    """
    Represent quantum circuit as a Directed Acyclic Graph (DAG).
    
    Nodes: Gates
    Edges: Dependencies (gate A must execute before gate B if they share qubits)
    """
    
    def __init__(self, gates: List[Gate]):
        """
        Build circuit graph from gates.
        
        Args:
            gates: List of Gate objects in execution order.
        """
        self.gates = gates
        self.n_gates = len(gates)
        
        # Build dependency graph
        self.dependencies: Dict[int, Set[int]] = defaultdict(set)  # gate_id -> {dependent_gate_ids}
        self.reverse_dependencies: Dict[int, Set[int]] = defaultdict(set)  # gate_id -> {prerequisite_gate_ids}
        self.qubit_to_gates: Dict[int, List[int]] = defaultdict(list)  # qubit -> [gate_ids]
        
        self._build_dag()
    
    def _build_dag(self):
        """Build dependency graph based on qubit sharing."""
        # Map each qubit to gates that use it
        for gate_id, gate in enumerate(self.gates):
            for qubit in gate.qubits:
                self.qubit_to_gates[qubit].append(gate_id)
        
        # Build dependencies: gate B depends on gate A if:
        # 1. A comes before B in execution order
        # 2. A and B share at least one qubit
        for gate_id, gate in enumerate(self.gates):
            # Find all previous gates that share qubits
            for qubit in gate.qubits:
                for prev_gate_id in self.qubit_to_gates[qubit]:
                    if prev_gate_id < gate_id:
                        # prev_gate_id must execute before gate_id
                        self.dependencies[prev_gate_id].add(gate_id)
                        self.reverse_dependencies[gate_id].add(prev_gate_id)
    
    def get_independent_gates(self, executed: Set[int]) -> Set[int]:
        """
        Find gates that can execute now (all dependencies satisfied).
        
        Args:
            executed: Set of gate IDs that have already been executed.
            
        Returns:
            Set of gate IDs that can execute in parallel.
        """
        independent = set()
        
        for gate_id in range(self.n_gates):
            if gate_id in executed:
                continue
            
            # Check if all dependencies are satisfied
            deps = self.reverse_dependencies.get(gate_id, set())
            if deps.issubset(executed):
                independent.add(gate_id)
        
        return independent
    
    def get_gate_dependencies(self, gate_id: int) -> Set[int]:
        """Get all gates that must execute before this gate."""
        return self.reverse_dependencies.get(gate_id, set())
    
    def get_gate_dependents(self, gate_id: int) -> Set[int]:
        """Get all gates that depend on this gate."""
        return self.dependencies.get(gate_id, set())
    
    def topological_levels(self) -> List[List[int]]:
        """
        Partition gates into topological levels (layers).
        
        Gates in the same level can execute in parallel.
        
        Returns:
            List of levels, each level is a list of gate IDs.
        """
        levels = []
        executed = set()
        in_degree = {i: len(self.reverse_dependencies.get(i, set())) 
                     for i in range(self.n_gates)}
        
        while len(executed) < self.n_gates:
            # Find gates with no remaining dependencies
            current_level = [
                gate_id for gate_id in range(self.n_gates)
                if gate_id not in executed and in_degree[gate_id] == 0
            ]
            
            if not current_level:
                # Should not happen in acyclic graph
                raise ValueError("Circular dependency detected!")
            
            levels.append(current_level)
            executed.update(current_level)
            
            # Update in-degrees
            for gate_id in current_level:
                for dependent_id in self.dependencies.get(gate_id, set()):
                    in_degree[dependent_id] -= 1
        
        return levels
    
    def get_qubits_used(self, gate_ids: Set[int]) -> Set[int]:
        """Get all qubits used by a set of gates."""
        qubits = set()
        for gate_id in gate_ids:
            qubits.update(self.gates[gate_id].qubits)
        return qubits
    
    def is_acyclic_partition(self, partition: List[int]) -> bool:
        """
        Check if a partition of gates forms an acyclic subgraph.
        
        Args:
            partition: List of gate IDs in the partition.
            
        Returns:
            True if partition is acyclic.
        """
        partition_set = set(partition)
        
        # Check: all dependencies within partition must be satisfied
        # (i.e., if A depends on B, and both are in partition, A must come after B)
        for gate_id in partition:
            deps = self.reverse_dependencies.get(gate_id, set())
            partition_deps = deps & partition_set
            
            # For each dependency in partition, check ordering
            for dep_id in partition_deps:
                if partition.index(dep_id) >= partition.index(gate_id):
                    return False  # Dependency violation
        
        return True
    
    def get_partition_qubits(self, partition: List[int]) -> Set[int]:
        """Get all qubits used by gates in a partition."""
        qubits = set()
        for gate_id in partition:
            qubits.update(self.gates[gate_id].qubits)
        return qubits
    
    def __repr__(self) -> str:
        levels = self.topological_levels()
        return (
            f"CircuitGraph(n_gates={self.n_gates}, "
            f"n_levels={len(levels)}, "
            f"max_parallelism={max(len(level) for level in levels)})"
        )
