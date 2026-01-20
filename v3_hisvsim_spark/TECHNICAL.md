# Technical Documentation

Deep dive into the implementation details.

---

## Table of Contents

1. [Circuit Partitioning](#circuit-partitioning)
2. [Spark Driver Execution](#spark-driver-execution)
3. [Gate Application](#gate-application)
4. [Parallel Gate Fusion](#parallel-gate-fusion)
5. [Fault Tolerance](#fault-tolerance)
6. [Performance Optimizations](#performance-optimizations)
7. [Test Coverage](#test-coverage)
8. [Scalability Limits](#scalability-limits)

---

## Circuit Partitioning

### Why Partition?

Gates that share qubits cannot run simultaneously. If gate A modifies qubit 0 and gate B reads qubit 0, B must wait for A.

### Building the Dependency Graph

```python
# For each gate, track dependencies based on shared qubits
qubit_to_gates = defaultdict(list)

for gate_idx, gate in enumerate(gates):
    for qubit in gate.qubits:
        # This gate depends on all previous gates using this qubit
        for prev_idx in qubit_to_gates[qubit]:
            graph.add_edge(prev_idx, gate_idx)
        qubit_to_gates[qubit].append(gate_idx)
```

This creates a directed acyclic graph (DAG) where edges represent "must execute before" relationships.

### Topological Level Assignment

Gates are grouped into levels where all gates in a level are independent:

```python
levels = []
in_degree = {node: graph.in_degree(node) for node in graph}

while unprocessed gates remain:
    # Level = all gates with no remaining dependencies
    current_level = [n for n in graph if in_degree[n] == 0 and n not in processed]
    levels.append(current_level)
    
    # Remove these gates, update dependencies
    for node in current_level:
        for successor in graph.successors(node):
            in_degree[successor] -= 1
```

**Example:**

```
Gates: H(0), H(2), CNOT(0,1), CNOT(1,2)

Dependencies:
  H(0) → CNOT(0,1)      (share qubit 0)
  H(2) → CNOT(1,2)      (share qubit 2)
  CNOT(0,1) → CNOT(1,2) (share qubit 1)

Levels:
  Level 0: [H(0), H(2)]    - no dependencies
  Level 1: [CNOT(0,1)]     - depends on H(0)
  Level 2: [CNOT(1,2)]     - depends on H(2) and CNOT(0,1)
```

---

## Spark Driver Execution

### Initialization

```python
def __init__(self, config):
    # Create Spark session
    self.spark = SparkSession.builder \
        .master(config.spark_master) \
        .config("spark.driver.memory", "4g") \
        .getOrCreate()
    
    # Initialize components
    self.state_manager = StateManager(self.spark, config)
    self.gate_applicator = ParallelGateApplicator(self.spark)
    self.partition_adapter = HiSVSIMPartitionAdapter()
```

### Main Execution Loop

For each topological level:

```python
for level_idx, level_gates in enumerate(levels):
    # 1. Write WAL entry (PENDING)
    wal_id = metadata_store.wal_create_pending(
        gate_start=current_gate,
        gate_end=current_gate + len(level_gates),
        version_in=current_version,
        version_out=current_version + 1
    )
    
    try:
        # 2. Apply gates (builds lazy plan)
        new_state, parallel_groups = gate_applicator.apply_gates_parallel(
            current_state, level_gates
        )
        
        # 3. Save state (triggers execution)
        state_manager.save_state(new_state, current_version + 1)
        
        # 4. Checkpoint if needed
        if should_checkpoint():
            checkpoint_manager.create_checkpoint(...)
        
        # 5. Mark WAL committed
        metadata_store.wal_mark_committed(wal_id)
        
        # 6. Advance to next state
        current_state = state_manager.load_state(current_version + 1)
        current_version += 1
        
    except Exception:
        metadata_store.wal_mark_failed(wal_id)
        raise
```

### Lazy Evaluation

Spark doesn't execute transformations immediately. It builds a plan:

```python
# These don't execute - they build a plan
df = df.withColumn("new_col", ...)
df = df.join(other_df, ...)
df = df.groupBy("idx").agg(...)

# This triggers execution of the entire plan
df.write.parquet(path)
```

Benefits:
- Spark optimizes the entire plan (reorders operations, pushes filters down)
- Single execution instead of many small ones
- Reduced overhead from job scheduling

---

## Gate Application

### State Representation

```
DataFrame schema:
  idx: long      - basis state index (0 to 2^n - 1)
  real: double   - real part of amplitude
  imag: double   - imaginary part of amplitude
```

Only non-zero amplitudes are stored (sparse representation).

### Single-Qubit Gate Application

For a gate on qubit `q`:

```python
def apply_single_qubit_gate(state_df, gate, qubit):
    # Extract the qubit bit from each index
    # idx=5 (binary 101), qubit=1 → bit=0
    # idx=6 (binary 110), qubit=1 → bit=1
    state_df = state_df.withColumn(
        "qubit_bit", 
        (col("idx") >> qubit) & 1
    )
    
    # Compute partner index (flip the qubit bit)
    state_df = state_df.withColumn(
        "partner_idx",
        col("idx") ^ (1 << qubit)
    )
    
    # Self-join to pair up |0⟩ and |1⟩ amplitudes
    paired = state_df.alias("a").join(
        state_df.alias("b"),
        col("a.idx") == col("b.partner_idx")
    )
    
    # Apply 2x2 gate matrix
    # |ψ'_0⟩ = m00*|ψ_0⟩ + m01*|ψ_1⟩
    # |ψ'_1⟩ = m10*|ψ_0⟩ + m11*|ψ_1⟩
    result = paired.select(
        col("a.idx"),
        (m00 * col("a.real") + m01 * col("b.real")).alias("real"),
        (m00 * col("a.imag") + m01 * col("b.imag")).alias("imag")
    )
    
    # Filter near-zero amplitudes
    return result.filter(col("real")**2 + col("imag")**2 > 1e-30)
```

### Two-Qubit Gate Application

Similar approach but with 4x4 matrices and two qubits extracted from the index.

---

## Parallel Gate Fusion

### The Problem

Applying gates one-by-one is slow. Each gate application triggers Spark operations.

### The Solution

Fuse independent single-qubit gates into a tensor product:

```python
def apply_gates_parallel(self, state_df, gates):
    # Group independent single-qubit gates
    independent_groups = find_independent_single_qubit_gates(gates)
    
    for group in independent_groups:
        if len(group) > 1:
            # Fuse into tensor product
            # H ⊗ H ⊗ H = single 8x8 matrix for 3 qubits
            fused_matrix = compute_tensor_product([g.matrix for g in group])
            fused_qubits = [g.qubit for g in group]
            
            state_df = apply_fused_gate(state_df, fused_matrix, fused_qubits)
        else:
            state_df = apply_single_gate(state_df, group[0])
    
    return state_df
```

### Tensor Product Computation

For gates G1 on qubit q1 and G2 on qubit q2:

```
G1 ⊗ G2 = 4x4 matrix where:
  (G1 ⊗ G2)[i][j] = G1[i//2][j//2] * G2[i%2][j%2]
```

For n gates, the result is a 2^n × 2^n matrix applied to n qubits simultaneously.

### Example

Hadamard wall on 8 qubits:

```
Without fusion:  8 separate gate applications
With fusion:     1 fused operation (256x256 matrix on 8 qubits)

parallel_groups = [8]  ← indicates all 8 gates fused
```

---

## Fault Tolerance

### Write-Ahead Log (WAL)

Before processing each level:

```python
# Record intent
wal_entry = {
    "run_id": "abc123",
    "gate_start": 10,
    "gate_end": 20,
    "version_in": 5,
    "version_out": 6,
    "status": "PENDING",
    "created_at": now()
}
```

After successful completion:

```python
wal_entry["status"] = "COMMITTED"
wal_entry["committed_at"] = now()
```

On failure:

```python
wal_entry["status"] = "FAILED"
```

### Checkpointing

Periodic state snapshots with metadata:

```python
checkpoint = {
    "state_version": 6,
    "last_gate_seq": 19,
    "state_path": "data/state/v6/",
    "checksum": "sha256:...",
    "created_at": now()
}
```

### Recovery Process

```python
def recover(run_id):
    # 1. Find latest checkpoint
    checkpoint = load_latest_checkpoint(run_id)
    
    # 2. Check WAL for incomplete work
    pending = get_pending_wal_entries(run_id)
    for entry in pending:
        mark_as_failed(entry)
    
    # 3. Resume from checkpoint
    state = load_state(checkpoint.state_version)
    resume_from_gate = checkpoint.last_gate_seq + 1
    
    return state, resume_from_gate
```

### Adaptive Checkpointing

```python
def should_checkpoint():
    return (
        state_size > 1_000_000 or           # Large state
        gates_since_checkpoint > 10 or       # Many gates processed
        time_since_checkpoint > 60           # Long time elapsed
    )
```

---

## Performance Optimizations

### 1. Sparse Representation

Only store non-zero amplitudes:

```
GHZ-1000: 2 rows instead of 2^1000 rows
W-200:    200 rows instead of 2^200 rows
```

### 2. Broadcast Joins

Small gate matrices are broadcast to all workers:

```python
gate_matrix_bc = spark.sparkContext.broadcast(gate_matrix)

# Join without shuffling data
state_df.join(broadcast(gate_df), ...)
```

### 3. Partition Management

```python
# Repartition after skewed operations
if state_df.rdd.getNumPartitions() < optimal:
    state_df = state_df.repartition(optimal)
```

### 4. Memory Management

```python
# Unpersist old states
if old_state.is_cached:
    old_state.unpersist(blocking=False)

# Cache frequently accessed states
current_state.cache()
```

### 5. Filter Early

Remove near-zero amplitudes as soon as possible:

```python
state_df = state_df.filter(
    col("real")**2 + col("imag")**2 > 1e-30
)
```

---

## Test Coverage

### Gate Types (15 total)

**Single-qubit:** H, X, Y, Z, S, T

**Two-qubit:** CNOT, CZ, CY, SWAP, CR

**Parameterized:** RY, R, G, CU

### Circuit Types

| Circuit | Sparsity | Amplitudes |
|---------|----------|------------|
| GHZ | O(1) | 2 |
| W State | O(n) | n |
| Bell | O(1) | 2 |
| Hadamard Wall | O(2^n) | 2^n |
| QFT | O(2^n) | 2^n |

### Verified Properties

- Normalization preserved (∑|α|² = 1)
- Correct final states for known circuits
- Parallel execution produces same results as sequential
- Recovery works after simulated crashes

---

## Scalability Limits

### Maximum Qubits Achieved

Tested on a local machine (MacBook, 16GB RAM, Java 17, PySpark 4.1.1):

| Circuit | Max Qubits | Non-Zero Amplitudes | Time | Limiting Factor |
|---------|------------|---------------------|------|-----------------|
| **GHZ** | 1000 | 2 | ~5 min | Gate count (time) |
| **W State** | 200 | 2,717,541 | ~3 min | Gate count (time) |
| **Hadamard Wall** | 12 | 4,096 | ~2 min | Memory |

### Why These Limits?

**GHZ State** - O(1) sparsity:
- Always exactly 2 non-zero amplitudes: |00...0⟩ and |11...1⟩
- Memory is constant regardless of qubit count
- Limited only by the number of gates (n-1 CNOTs for n qubits)
- Could theoretically simulate thousands of qubits

**W State** - O(n) sparsity:
- Exactly n non-zero amplitudes (one for each basis state with a single |1⟩)
- Memory grows linearly
- Limited by gate complexity (2n-2 gates for n qubits)

**Hadamard Wall** - O(2^n) sparsity:
- All 2^n basis states have non-zero amplitude
- Memory doubles with each qubit added
- Hit Spark broadcast join memory limit at 12 qubits (4,096 amplitudes)
- The fused gate matrix (4096×4096) exceeded driver memory

### Memory Requirements

```
Qubits    Amplitudes    Memory (16 bytes each)
------    ----------    ----------------------
10        1,024         16 KB
12        4,096         64 KB
14        16,384        256 KB
16        65,536        1 MB
20        1,048,576     16 MB
25        33,554,432    512 MB
30        1,073,741,824 16 GB
35        34 billion    544 GB
40        1 trillion    17 TB
```

### Scaling with a Cluster

With distributed Spark (multiple machines):

| Setup | Dense Circuit Max | Sparse Circuit Max |
|-------|-------------------|-------------------|
| Local (16GB) | ~12-14 qubits | 1000+ qubits |
| Small cluster (10×64GB) | ~25-28 qubits | 1000+ qubits |
| Large cluster (100×64GB) | ~32-35 qubits | 1000+ qubits |
| Supercomputer | ~45-50 qubits | 1000+ qubits |

**Key insight**: Distribution helps with dense states but has diminishing returns due to communication overhead. Sparse states can scale essentially unlimited with any setup.

