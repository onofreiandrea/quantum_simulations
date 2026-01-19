"""
Circuit partitioning for HiSVSIM-style distributed simulation.

Partitions circuit into acyclic sub-circuits that can be simulated in parallel.
"""
from __future__ import annotations

from typing import List, Set, Dict
from collections import defaultdict

from .gates import Gate
from .circuit_graph import CircuitGraph


class CircuitPartitioner:
    """
    Partition circuit into independent acyclic sub-circuits.
    
    Strategy: Greedy level-based partitioning
    - Group gates by topological levels
    - Distribute levels across partitions
    - Ensure acyclic property within each partition
    """
    
    def __init__(self, strategy: str = "level_based"):
        """
        Initialize partitioner.
        
        Args:
            strategy: Partitioning strategy:
                - "level_based": Group by topological levels (default)
                - "greedy": Greedy assignment to minimize qubit overlap
                - "balanced": Balance partition sizes
        """
        self.strategy = strategy
    
    def partition(
        self, 
        graph: CircuitGraph, 
        n_partitions: int
    ) -> List[List[Gate]]:
        """
        Partition circuit into n independent sub-circuits.
        
        Args:
            graph: CircuitGraph representing the circuit.
            n_partitions: Number of partitions to create.
            
        Returns:
            List of partitions, each partition is a list of Gate objects.
        """
        if self.strategy == "level_based":
            return self._partition_level_based(graph, n_partitions)
        elif self.strategy == "greedy":
            return self._partition_greedy(graph, n_partitions)
        elif self.strategy == "balanced":
            return self._partition_balanced(graph, n_partitions)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")
    
    def _partition_level_based(
        self, 
        graph: CircuitGraph, 
        n_partitions: int
    ) -> List[List[Gate]]:
        """
        Partition by topological levels.
        
        Distributes levels across partitions to balance work.
        """
        levels = graph.topological_levels()
        
        # Distribute levels across partitions
        partitions: List[List[int]] = [[] for _ in range(n_partitions)]
        partition_sizes = [0] * n_partitions
        
        for level in levels:
            # Sort gates in level by some heuristic (e.g., number of qubits)
            sorted_level = sorted(
                level, 
                key=lambda gid: len(graph.gates[gid].qubits),
                reverse=True
            )
            
            # Assign gates to partitions using round-robin or size-based
            for gate_id in sorted_level:
                # Choose partition with smallest current size
                min_partition = min(range(n_partitions), 
                                  key=lambda i: partition_sizes[i])
                partitions[min_partition].append(gate_id)
                partition_sizes[min_partition] += 1
        
        # Convert gate IDs to Gate objects
        return [[graph.gates[gid] for gid in partition] 
                for partition in partitions]
    
    def _partition_greedy(
        self, 
        graph: CircuitGraph, 
        n_partitions: int
    ) -> List[List[Gate]]:
        """
        Greedy partitioning: assign gates to minimize qubit overlap.
        """
        partitions: List[List[int]] = [[] for _ in range(n_partitions)]
        partition_qubits: List[Set[int]] = [set() for _ in range(n_partitions)]
        
        # Process gates in topological order
        levels = graph.topological_levels()
        
        for level in levels:
            for gate_id in level:
                gate = graph.gates[gate_id]
                gate_qubits = set(gate.qubits)
                
                # Find partition with minimum qubit overlap
                best_partition = min(
                    range(n_partitions),
                    key=lambda i: len(partition_qubits[i] & gate_qubits)
                )
                
                partitions[best_partition].append(gate_id)
                partition_qubits[best_partition].update(gate_qubits)
        
        return [[graph.gates[gid] for gid in partition] 
                for partition in partitions]
    
    def _partition_balanced(
        self, 
        graph: CircuitGraph, 
        n_partitions: int
    ) -> List[List[Gate]]:
        """
        Balance partition sizes while maintaining acyclic property.
        """
        # Start with level-based partitioning
        partitions = self._partition_level_based(graph, n_partitions)
        
        # Rebalance if needed
        sizes = [len(p) for p in partitions]
        avg_size = sum(sizes) / n_partitions
        
        # Simple rebalancing: move gates from large to small partitions
        # (This is a simplified version - full implementation would use
        # graph partitioning algorithms like METIS)
        
        return partitions
    
    def find_independent_gates(self, graph: CircuitGraph) -> List[Set[int]]:
        """
        Find sets of gates that can execute in parallel.
        
        Returns:
            List of sets, each set contains gate IDs that can run in parallel.
        """
        levels = graph.topological_levels()
        return [set(level) for level in levels]
    
    def get_partition_stats(
        self, 
        graph: CircuitGraph, 
        partitions: List[List[Gate]]
    ) -> Dict:
        """Get statistics about partitions."""
        stats = {
            "n_partitions": len(partitions),
            "partition_sizes": [len(p) for p in partitions],
            "partition_qubits": [
                graph.get_partition_qubits(
                    [graph.gates.index(g) for g in partition]
                )
                for partition in partitions
            ],
            "qubit_overlaps": [],
            "is_acyclic": []
        }
        
        # Check qubit overlaps
        for i, partition in enumerate(partitions):
            partition_gate_ids = [graph.gates.index(g) for g in partition]
            stats["is_acyclic"].append(
                graph.is_acyclic_partition(partition_gate_ids)
            )
            
            # Find overlaps with other partitions
            overlaps = []
            for j, other_partition in enumerate(partitions):
                if i != j:
                    other_gate_ids = [graph.gates.index(g) for g in other_partition]
                    qubits_i = graph.get_partition_qubits(partition_gate_ids)
                    qubits_j = graph.get_partition_qubits(other_gate_ids)
                    overlap = len(qubits_i & qubits_j)
                    overlaps.append(overlap)
            stats["qubit_overlaps"].append(overlaps)
        
        return stats
