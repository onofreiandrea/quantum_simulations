"""
Adapter to use HiSVSIM's partitioning logic with our circuit representation.

This module bridges between:
- Our circuit format (circuit_dict with gates)
- HiSVSIM's partitioning (DOT files, NetworkX graphs)

Optimized partitioning strategies:
1. Load-balanced: Distribute gates evenly considering complexity
2. Locality-aware: Minimize qubit overlaps between partitions
3. Hybrid: Balance load while minimizing overlaps
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Dict, Set, Tuple
import networkx as nx
import tempfile
import shutil

# Add HiSVSIM repo to path
HISVSIM_REPO = Path(__file__).parent.parent.parent / "hisvsim_repo"
if str(HISVSIM_REPO) not in sys.path:
    sys.path.insert(0, str(HISVSIM_REPO))

# Import from v2_common
from v2_common import gates, frontend

Gate = gates.Gate
circuit_dict_to_gates = frontend.circuit_dict_to_gates


class HiSVSIMPartitionAdapter:
    """
    Adapts HiSVSIM's partitioning approach to work with our circuit format.
    
    Optimized partitioning strategies:
    - Load-balanced: Even gate distribution
    - Locality-aware: Minimize qubit overlaps
    - Hybrid: Balance both load and locality
    """
    
    def __init__(self, strategy: str = "hybrid"):
        """
        Initialize the adapter.
        
        Args:
            strategy: Partitioning strategy - "load_balanced", "locality", or "hybrid"
        """
        self.strategy = strategy
        self.temp_dir = None
    
    def partition_circuit(
        self, 
        gates: List[Gate], 
        n_partitions: int = None
    ) -> List[List[int]]:
        """
        Partition circuit into acyclic sub-circuits using optimized strategies.
        
        Args:
            gates: List of Gate objects.
            n_partitions: Target number of partitions (optional).
            
        Returns:
            List of partitions, each partition is a list of gate indices.
        """
        if n_partitions is None:
            n_partitions = max(1, len(gates) // 10)  # Default: ~10 gates per partition
        
        # Build dependency graph
        G = self._build_circuit_graph(gates)
        
        # Find topological levels (independent gate sets)
        levels = self._topological_levels(G, gates)
        
        # Apply partitioning strategy
        if self.strategy == "load_balanced":
            partitions = self._partition_load_balanced(levels, gates, n_partitions)
        elif self.strategy == "locality":
            partitions = self._partition_locality_aware(levels, gates, n_partitions)
        else:  # hybrid
            partitions = self._partition_hybrid(levels, gates, n_partitions)
        
        return partitions
    
    def _build_circuit_graph(self, gates: List[Gate]) -> nx.MultiDiGraph:
        """
        Build NetworkX MultiDiGraph from gates (HiSVSIM style).
        
        Nodes: Gates
        Edges: Dependencies (gate A → gate B if they share qubits and A < B)
        
        FIXED: Ensures acyclic graph by only adding edges from earlier gates.
        """
        G = nx.MultiDiGraph()
        
        # Track qubit usage: qubit -> [gate_indices] (in order)
        qubit_to_gates: Dict[int, List[int]] = {}
        
        for gate_idx, gate in enumerate(gates):
            gate_name = f"gate_{gate_idx}_{gate.gate_name}"
            G.add_node(gate_name, gate_idx=gate_idx, gate=gate)
            
            # Track qubit dependencies - only add edges from ALL previous gates using same qubits
            for qubit in gate.qubits:
                if qubit not in qubit_to_gates:
                    qubit_to_gates[qubit] = []
                
                # Add edges from ALL previous gates using this qubit
                # This ensures sequential execution order
                for prev_gate_idx in qubit_to_gates[qubit]:
                    if prev_gate_idx < gate_idx:  # Only from earlier gates
                        prev_gate_name = f"gate_{prev_gate_idx}_{gates[prev_gate_idx].gate_name}"
                        # Use unique edge key to avoid MultiDiGraph issues
                        G.add_edge(prev_gate_name, gate_name, qubit=qubit, key=f"{prev_gate_idx}->{gate_idx}")
                
                qubit_to_gates[qubit].append(gate_idx)
        
        # Verify graph is acyclic
        if not nx.is_directed_acyclic_graph(G):
            # If cyclic, it's a bug - but try to fix by removing problematic edges
            # This shouldn't happen with proper sequential gate ordering
            raise ValueError(
                f"Circuit graph has cycles! This indicates gates are not properly ordered. "
                f"Gates: {[g.gate_name for g in gates]}"
            )
        
        return G
    
    def _topological_levels(
        self, 
        graph: nx.MultiDiGraph, 
        gates: List[Gate]
    ) -> List[List[int]]:
        """
        Partition gates into topological levels.
        
        Gates in the same level can execute in parallel.
        """
        levels = []
        executed = set()
        
        # Build in-degree map
        in_degree = {node: graph.in_degree(node) for node in graph.nodes()}
        
        while len(executed) < len(gates):
            # Find gates with no remaining dependencies
            current_level = []
            for node in graph.nodes():
                if node in executed:
                    continue
                
                gate_idx = graph.nodes[node]['gate_idx']
                if in_degree[node] == 0:
                    current_level.append(gate_idx)
            
            if not current_level:
                # This can happen if graph has cycles or if we've already processed all gates
                # Check if we've actually processed all gates
                if len(executed) < len(gates):
                    # There are unprocessed gates but none have zero in-degree
                    # This indicates a cycle or bug in graph construction
                    remaining = [gates[graph.nodes[node]['gate_idx']].gate_name 
                                for node in graph.nodes() if node not in executed]
                    raise ValueError(
                        f"Circular dependency detected! Remaining gates: {remaining}. "
                        f"This may be caused by duplicate gates on same qubits."
                    )
                break  # All gates processed
            
            levels.append(current_level)
            executed.update([f"gate_{idx}_{gates[idx].gate_name}" 
                            for idx in current_level])
            
            # Update in-degrees
            for gate_idx in current_level:
                node = f"gate_{gate_idx}_{gates[gate_idx].gate_name}"
                for successor in graph.successors(node):
                    in_degree[successor] -= 1
        
        return levels
    
    def _gate_complexity(self, gate: Gate) -> float:
        """
        Estimate computational complexity of a gate.
        
        Returns:
            Complexity score (higher = more expensive)
        """
        # Base complexity: 2-qubit gates are more expensive
        base = 1.0 if len(gate.qubits) == 1 else 2.0
        
        # Non-stabilizer gates are more expensive
        non_stabilizer_gates = {"T", "RY", "R", "G", "CU"}
        if any(gate.gate_name.startswith(name) for name in non_stabilizer_gates):
            base *= 1.5
        
        return base
    
    def _partition_load_balanced(
        self,
        levels: List[List[int]],
        gates: List[Gate],
        n_partitions: int
    ) -> List[List[int]]:
        """
        Partition using load-balanced strategy.
        
        Distributes gates evenly across partitions, considering gate complexity.
        """
        partitions: List[List[int]] = [[] for _ in range(n_partitions)]
        partition_loads = [0.0] * n_partitions
        
        # Flatten levels into a single list, preserving order
        all_gates = []
        for level in levels:
            all_gates.extend(level)
        
        # Sort by complexity (descending) for better load balance
        all_gates.sort(key=lambda idx: self._gate_complexity(gates[idx]), reverse=True)
        
        # Assign gates to partitions greedily
        for gate_idx in all_gates:
            complexity = self._gate_complexity(gates[gate_idx])
            
            # Choose partition with smallest current load
            min_partition = min(range(n_partitions), 
                              key=lambda i: partition_loads[i])
            partitions[min_partition].append(gate_idx)
            partition_loads[min_partition] += complexity
        
        return partitions
    
    def _partition_locality_aware(
        self,
        levels: List[List[int]],
        gates: List[Gate],
        n_partitions: int
    ) -> List[List[int]]:
        """
        Partition using locality-aware strategy.
        
        Minimizes qubit overlaps between partitions by grouping gates
        that share qubits together.
        """
        partitions: List[List[int]] = [[] for _ in range(n_partitions)]
        partition_qubits: List[Set[int]] = [set() for _ in range(n_partitions)]
        
        # Process levels sequentially to maintain dependencies
        for level in levels:
            # Sort gates in level by qubit overlap with existing partitions
            sorted_level = sorted(
                level,
                key=lambda idx: self._qubit_overlap_score(
                    gates[idx].qubits, partition_qubits
                )
            )
            
            # Assign gates to partitions
            for gate_idx in sorted_level:
                gate_qubits = set(gates[gate_idx].qubits)
                
                # Find partition with best locality (most qubit overlap)
                best_partition = max(
                    range(n_partitions),
                    key=lambda i: len(gate_qubits & partition_qubits[i])
                )
                
                # If no overlap, choose smallest partition
                if len(gate_qubits & partition_qubits[best_partition]) == 0:
                    best_partition = min(
                        range(n_partitions),
                        key=lambda i: len(partitions[i])
                    )
                
                partitions[best_partition].append(gate_idx)
                partition_qubits[best_partition].update(gate_qubits)
        
        return partitions
    
    def _partition_hybrid(
        self,
        levels: List[List[int]],
        gates: List[Gate],
        n_partitions: int
    ) -> List[List[int]]:
        """
        Partition using hybrid strategy.
        
        Balances load while minimizing qubit overlaps.
        """
        partitions: List[List[int]] = [[] for _ in range(n_partitions)]
        partition_loads = [0.0] * n_partitions
        partition_qubits: List[Set[int]] = [set() for _ in range(n_partitions)]
        
        # Process levels sequentially
        for level in levels:
            # Sort gates by complexity (descending)
            sorted_level = sorted(
                level,
                key=lambda idx: self._gate_complexity(gates[idx]),
                reverse=True
            )
            
            # Assign gates to partitions
            for gate_idx in sorted_level:
                gate = gates[gate_idx]
                gate_qubits = set(gate.qubits)
                complexity = self._gate_complexity(gate)
                
                # Score each partition: balance load and locality
                scores = []
                for i in range(n_partitions):
                    # Load component: prefer partitions with lower load
                    load_score = 1.0 / (1.0 + partition_loads[i])
                    
                    # Locality component: prefer partitions with qubit overlap
                    overlap = len(gate_qubits & partition_qubits[i])
                    locality_score = overlap + 0.1  # Small bias to avoid empty partitions
                    
                    # Combined score (weighted)
                    score = 0.6 * load_score + 0.4 * locality_score
                    scores.append((score, i))
                
                # Choose partition with best score
                best_partition = max(scores, key=lambda x: x[0])[1]
                
                partitions[best_partition].append(gate_idx)
                partition_loads[best_partition] += complexity
                partition_qubits[best_partition].update(gate_qubits)
        
        return partitions
    
    def _qubit_overlap_score(
        self,
        gate_qubits,
        partition_qubits: List[Set[int]]
    ) -> float:
        """
        Calculate qubit overlap score for a gate across partitions.
        
        Args:
            gate_qubits: Set or list of qubit indices
            partition_qubits: List of sets of qubits per partition
        
        Returns:
            Average overlap with existing partitions
        """
        if not partition_qubits:
            return 0.0
        
        # Convert to set if needed
        gate_qubits_set = set(gate_qubits) if not isinstance(gate_qubits, set) else gate_qubits
        
        overlaps = [len(gate_qubits_set & pq) for pq in partition_qubits]
        return sum(overlaps) / len(overlaps) if overlaps else 0.0
    
    def partition_circuit_dict(
        self, 
        circuit_dict: Dict, 
        n_partitions: int = None
    ) -> List[Dict]:
        """
        Partition a circuit dictionary into sub-circuits.
        
        Args:
            circuit_dict: Circuit dictionary with 'gates' and 'number_of_qubits'.
            n_partitions: Target number of partitions.
            
        Returns:
            List of circuit dictionaries, one per partition.
        """
        n_qubits, gates = circuit_dict_to_gates(circuit_dict)
        
        # Get partition indices
        partition_indices = self.partition_circuit(gates, n_partitions)
        
        # Build partition circuits
        partitions = []
        for part_indices in partition_indices:
            if not part_indices:
                continue
            
            # Sort indices to preserve original gate execution order
            partition_gates = [gates[i] for i in sorted(part_indices)]
            # Sort by original index to maintain execution order
            partition_gates.sort(key=lambda g: gates.index(g) if g in gates else 0)
            
            # Convert back to circuit dict format
            gate_dicts = []
            for gate in partition_gates:
                gate_dict = {
                    "qubits": gate.qubits,
                    "gate": gate.gate_name,
                }
                
                # Extract params based on gate type
                # CRGate stores k in gate_name as "CR{k}"
                if gate.gate_name.startswith("CR"):
                    # Extract k from gate_name (e.g., "CR3" -> k=3)
                    import re
                    match = re.match(r'CR(\d+)', gate.gate_name)
                    if match:
                        gate_dict["params"] = {"k": int(match.group(1))}
                    else:
                        gate_dict["params"] = {"k": 2}  # Default
                elif gate.gate_name.startswith("R") and gate.gate_name != "RY":
                    # RGate stores k in gate_name as "R{k}"
                    import re
                    match = re.match(r'R(\d+)', gate.gate_name)
                    if match:
                        gate_dict["params"] = {"k": int(match.group(1))}
                elif gate.gate_name == "RY":
                    if hasattr(gate, 'theta'):
                        gate_dict["params"] = {"theta": gate.theta}
                elif gate.gate_name == "G":
                    if hasattr(gate, 'p'):
                        gate_dict["params"] = {"p": gate.p}
                elif gate.gate_name == "CU":
                    if hasattr(gate, 'U') and hasattr(gate, 'exponent'):
                        gate_dict["params"] = {
                            "U": gate.U,
                            "exponent": gate.exponent,
                            "name": getattr(gate, 'name', None)
                        }
                
                gate_dicts.append(gate_dict)
            
            partition_dict = {
                "number_of_qubits": n_qubits,
                "gates": gate_dicts
            }
            partitions.append(partition_dict)
        
        return partitions
    
    def get_partition_stats(
        self, 
        gates: List[Gate], 
        partitions: List[List[int]]
    ) -> Dict:
        """Get statistics about partitions."""
        stats = {
            "n_partitions": len(partitions),
            "partition_sizes": [len(p) for p in partitions],
            "partition_loads": [],
            "partition_qubits": [],
            "qubit_overlaps": [],
        }
        
        # Calculate loads and qubits for each partition
        for partition in partitions:
            load = sum(self._gate_complexity(gates[idx]) for idx in partition)
            stats["partition_loads"].append(load)
            
            qubits = set()
            for gate_idx in partition:
                qubits.update(gates[gate_idx].qubits)
            stats["partition_qubits"].append(qubits)
        
        # Calculate overlaps
        for i, qubits_i in enumerate(stats["partition_qubits"]):
            overlaps = []
            for j, qubits_j in enumerate(stats["partition_qubits"]):
                if i != j:
                    overlaps.append(len(qubits_i & qubits_j))
            stats["qubit_overlaps"].append(overlaps)
        
        return stats
