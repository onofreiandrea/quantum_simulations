# HiSVSIM-Style Circuit Partitioning Integration Plan

## HiSVSIM Approach (from IEEE CLUSTER 2022)

**Key Innovation**: Partition the **circuit graph** into acyclic sub-circuits, not just distribute the state vector.

### Their Strategy:
1. **Graph-based partitioning**: Represent circuit as DAG (Directed Acyclic Graph)
2. **Acyclic sub-circuits**: Partition into independent blocks that can be simulated in parallel
3. **Hierarchical simulation**: Build state vectors hierarchically from smaller sub-circuits
4. **Three partitioning strategies**:
   - Acyclic graph partitioning (best time-to-solution)
   - Other strategies (faster partitioning, potentially slower simulation)

### Benefits:
- **Circuit-level parallelism**: Multiple sub-circuits simulated simultaneously
- **Better data locality**: Acyclic partitions reduce data movement
- **Reduced passes**: Fewer iterations through data
- **Scalability**: Can distribute across many workers

## Current Spark Implementation

### What We Do Now:
- ✅ **State vector distribution**: Partition amplitudes across workers
- ✅ **Gate-level parallelism**: Each worker processes its partition
- ❌ **Sequential gate application**: Gates applied one at a time
- ❌ **No circuit partitioning**: Entire circuit processed sequentially

### What We Need:
- ✅ **Circuit graph representation**: Build DAG from gates
- ✅ **Acyclic partitioning**: Find independent sub-circuits
- ✅ **Parallel sub-circuit simulation**: Run partitions on different workers
- ✅ **State merging**: Combine results from sub-circuits

## Integration Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Circuit Input (Gates)                                       │
└─────────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────────┐
│  Circuit Graph Builder                                       │
│  - Build DAG from gates                                      │
│  - Identify dependencies                                     │
│  - Find qubit connectivity                                  │
└─────────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────────┐
│  Acyclic Partitioning                                        │
│  - Partition into independent sub-circuits                  │
│  - Ensure acyclic property                                  │
│  - Balance partition sizes                                   │
└─────────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────────┐
│  Parallel Sub-Circuit Simulation                            │
│                                                              │
│  Worker 1: Sub-circuit A → State A                          │
│  Worker 2: Sub-circuit B → State B                         │
│  Worker 3: Sub-circuit C → State C                         │
│  ...                                                         │
└─────────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────────┐
│  State Merging                                               │
│  - Combine states from sub-circuits                          │
│  - Handle qubit overlaps                                    │
│  - Final state vector                                        │
└─────────────────────────────────────────────────────────────┘
```

## Implementation Plan

### Phase 1: Circuit Graph Representation

**File**: `src/circuit_graph.py`

```python
class CircuitGraph:
    """Represent quantum circuit as DAG."""
    
    def __init__(self, gates: List[Gate]):
        self.gates = gates
        self.dag = self._build_dag()
        self.qubit_dependencies = self._analyze_dependencies()
    
    def _build_dag(self) -> nx.DiGraph:
        """Build directed acyclic graph from gates."""
        # Nodes: gates
        # Edges: dependencies (gate A must execute before gate B)
        pass
    
    def _analyze_dependencies(self) -> Dict[int, Set[int]]:
        """Find which gates depend on which qubits."""
        pass
```

### Phase 2: Acyclic Partitioning

**File**: `src/circuit_partitioner.py`

```python
class CircuitPartitioner:
    """Partition circuit into acyclic sub-circuits."""
    
    def partition(self, graph: CircuitGraph, n_partitions: int) -> List[List[Gate]]:
        """
        Partition circuit into n independent acyclic sub-circuits.
        
        Strategy:
        1. Find independent gate sets (no dependencies)
        2. Ensure acyclic property
        3. Balance partition sizes
        4. Minimize qubit overlap
        """
        pass
    
    def _find_independent_gates(self, graph: CircuitGraph) -> List[Set[int]]:
        """Find gates that can execute in parallel."""
        pass
    
    def _ensure_acyclic(self, partition: List[Gate]) -> bool:
        """Verify partition is acyclic."""
        pass
```

### Phase 3: Parallel Sub-Circuit Execution

**File**: `src/driver.py` (modify)

```python
def run_circuit_with_partitioning(self, circuit_dict: dict) -> SimulationResult:
    """Run circuit with HiSVSIM-style partitioning."""
    
    # Build circuit graph
    n_qubits, gates = circuit_dict_to_gates(circuit_dict)
    graph = CircuitGraph(gates)
    
    # Partition circuit
    partitions = self.circuit_partitioner.partition(
        graph, 
        n_partitions=self.config.num_workers
    )
    
    # Simulate each partition in parallel (Spark mapPartitions)
    partition_states = (
        self.spark.parallelize(partitions, numSlices=len(partitions))
        .map(lambda partition: self._simulate_partition(partition, n_qubits))
        .collect()
    )
    
    # Merge states
    final_state = self._merge_partition_states(partition_states)
    
    return SimulationResult(...)
```

### Phase 4: State Merging

**File**: `src/state_merger.py`

```python
class StateMerger:
    """Merge states from parallel sub-circuit simulations."""
    
    def merge(self, states: List[DataFrame], qubit_overlaps: Dict) -> DataFrame:
        """
        Merge states from multiple sub-circuits.
        
        Challenge: Sub-circuits may share qubits.
        Solution: Tensor product + proper normalization.
        """
        pass
```

## Key Challenges

### 1. **Qubit Overlap**
- Sub-circuits may share qubits
- Need to properly combine states
- Solution: Use tensor product with proper qubit ordering

### 2. **Dependency Tracking**
- Gates may depend on previous gates
- Must respect execution order
- Solution: Topological sort of DAG

### 3. **State Merging Complexity**
- Merging states from different partitions
- Ensuring correctness
- Solution: Hierarchical merging (like HiSVSIM)

### 4. **Partition Balance**
- Uneven partitions → load imbalance
- Solution: Use graph partitioning algorithms (METIS, KaHIP)

## Spark Integration Strategy

### Option A: Spark MapPartitions
```python
# Each partition = one sub-circuit
partition_states = (
    spark.parallelize(partitions, numSlices=len(partitions))
    .mapPartitions(lambda partition_iter: 
        [self._simulate_partition(next(partition_iter), n_qubits)]
    )
    .collect()
)
```

### Option B: Spark Broadcast + Map
```python
# Broadcast circuit graph, map over partitions
graph_bc = spark.sparkContext.broadcast(circuit_graph)
partition_states = (
    spark.parallelize(range(len(partitions)))
    .map(lambda i: self._simulate_partition(partitions[i], graph_bc.value))
    .collect()
)
```

### Option C: Direct Spark DataFrame Operations
```python
# Represent partitions as DataFrame, use Spark SQL
partitions_df = spark.createDataFrame([
    (i, partition) for i, partition in enumerate(partitions)
], ["partition_id", "gates"])

# Use UDF to simulate each partition
result_df = partitions_df.select(
    simulate_partition_udf("gates").alias("state")
)
```

## Benefits for Our Implementation

1. **True Circuit-Level Parallelism**: Multiple sub-circuits run simultaneously
2. **Better Scalability**: Can use many workers effectively
3. **Reduced Communication**: Acyclic partitions minimize data movement
4. **Faster Simulation**: Parallel execution of independent gates

## Implementation Steps

1. ✅ **Add networkx dependency** (for graph algorithms)
2. ✅ **Implement CircuitGraph** (build DAG from gates)
3. ✅ **Implement CircuitPartitioner** (acyclic partitioning)
4. ✅ **Modify Driver** (parallel sub-circuit execution)
5. ✅ **Implement StateMerger** (merge partition results)
6. ✅ **Add tests** (verify correctness)
7. ✅ **Benchmark** (compare with sequential approach)

## References

- HiSVSIM Paper: "Efficient Hierarchical State Vector Simulation of Quantum Circuits via Acyclic Graph Partitioning", IEEE CLUSTER 2022
- Repository: https://github.com/pnnl/hisvsim.git
- arXiv: https://arxiv.org/pdf/2205.06973.pdf
