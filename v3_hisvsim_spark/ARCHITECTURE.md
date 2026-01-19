# V3 Architecture - Complete Guide

**A comprehensive guide of the v3 quantum simulator architecture**

---

## Table of Contents

1. [Overview](#overview)
2. [What is Quantum Circuit Simulation?](#what-is-quantum-circuit-simulation)
3. [Why Spark? Why HiSVSIM?](#why-spark-why-hisvsim)
4. [High-Level Architecture](#high-level-architecture)
5. [Apache Spark Basics](#apache-spark-basics)
6. [Circuit Partitioning (HiSVSIM)](#circuit-partitioning-hisvsim)
7. [Level-Based Parallelism](#level-based-parallelism)
8. [State Vector Representation](#state-vector-representation)
9. [Gate Application](#gate-application)
10. [Fault Tolerance](#fault-tolerance)
11. [Data Flow](#data-flow)
12. [Key Components](#key-components)
13. [Performance Optimizations](#performance-optimizations)

---

## Overview

**V3** is a distributed quantum circuit simulator that combines:
- **HiSVSIM's partitioning algorithms** - Efficiently divides circuits into parallelizable parts
- **Apache Spark** - Distributed computing framework for parallel execution
- **Fault tolerance** - WAL, checkpoints, and recovery mechanisms

**Goal**: Simulate large quantum circuits using classical computers by leveraging distributed computing.

---

## What is Quantum Circuit Simulation?

### The Challenge

A quantum state with **N qubits** requires **2^N complex numbers** to represent:
- 10 qubits = 1,024 amplitudes
- 20 qubits = 1,048,576 amplitudes  
- 30 qubits = 1,073,741,824 amplitudes (1 billion!)
- 50 qubits = 1,125,899,906,842,624 amplitudes (1 quadrillion!)

**Memory requirement**: Each amplitude is a complex number (16 bytes)
- 30 qubits = ~16 GB of memory
- 50 qubits = ~18 Petabytes (impossible with current hardware!)

### The Solution: Distributed Simulation

Instead of storing everything on one machine:
1. **Partition** the circuit into smaller parts
2. **Distribute** computation across multiple machines
3. **Combine** results efficiently

This is what essentially the implementiation of the v3.

---

## Why Spark? Why HiSVSIM?

### Apache Spark

**What is Spark?**
- Distributed computing framework
- Handles data distribution, fault tolerance, and parallel execution
- Works with DataFrames (like SQL tables) for structured data

**Why Spark?**
- Built-in fault tolerance
- Automatic data distribution
- Efficient in-memory processing
- Handles large datasets (billions of rows)
- Lazy evaluation (optimizes execution plan)

### HiSVSIM Partitioning


**Why HiSVSIM?**
- Proven partitioning strategies
- Creates acyclic sub-circuits (no circular dependencies)
- Maximizes parallelism
- Handles complex gate dependencies


---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    USER INPUT                               │
│  Circuit Dict: {number_of_qubits: N, gates: [...]}        │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│              SparkHiSVSIMDriver                             │
│  - Orchestrates entire simulation                          │
│  - Manages Spark session                                   │
│  - Coordinates components                                   │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│        1. CIRCUIT PARTITIONING (HiSVSIM)                   │
│                                                             │
│  HiSVSIMPartitionAdapter:                                   │
│  ├─ Build circuit graph (DAG)                             │
│  ├─ Find topological levels                                │
│  └─ Partition into independent levels                     │
│                                                             │
│  Output: Levels of gates that can run in parallel          │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│        2. PARALLEL EXECUTION (Spark)                        │
│                                                             │
│  For each level:                                           │
│  ├─ Write WAL PENDING                                      │
│  ├─ Apply gates in parallel (Spark DataFrame ops)         │
│  ├─ Save state to Parquet (durable storage)               │
│  ├─ Create checkpoint (if needed)                         │
│  └─ Mark WAL COMMITTED                                     │
│                                                             │
│  GateApplicator:                                           │
│  ├─ Applies gates using Spark DataFrame transformations    │
│  ├─ Uses bitwise operations for qubit manipulation       │
│  └─ Groups and sums amplitudes                             │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│        3. FAULT TOLERANCE                                   │
│                                                             │
│  MetadataStore (DuckDB):                                   │
│  ├─ WAL entries (PENDING → COMMITTED)                      │
│  └─ Checkpoint records                                     │
│                                                             │
│  CheckpointManager:                                         │
│  ├─ Creates periodic snapshots                             │
│  └─ Stores state versions                                  │
│                                                             │
│  RecoveryManager:                                           │
│  ├─ Loads latest checkpoint                                │
│  ├─ Reconciles WAL entries                                │
│  └─ Resumes from last committed gate                       │
└─────────────────────────────────────────────────────────────┘
                         ↓
┌─────────────────────────────────────────────────────────────┐
│                    FINAL STATE                              │
│  Spark DataFrame: (idx, real, imag)                        │
│  - Sparse representation (only non-zero amplitudes)        │
│  - Stored in Parquet files                                  │
└─────────────────────────────────────────────────────────────┘
```

---

## Apache Spark Basics

### Key Concepts

#### 1. SparkSession
```python
spark = SparkSession.builder \
    .master("local[*]") \      # Use all CPU cores locally
    .appName("quantum_sim") \
    .getOrCreate()
```
- **Entry point** for Spark operations
- Manages Spark context and resources
- Similar to a database connection

#### 2. DataFrame
```python
# A DataFrame is like a SQL table
state_df = spark.createDataFrame([
    (0, 1.0, 0.0),  # (idx, real, imag)
    (1, 0.0, 0.0),
    ...
], schema=["idx", "real", "imag"])
```


#### 3. Transformations vs Actions

**Transformations** (Lazy - don't execute immediately):
```python
# This builds a plan but doesn't execute
new_state = state_df \
    .filter(F.col("real") > 0.1) \
    .withColumn("amplitude", F.sqrt(F.col("real")**2 + F.col("imag")**2))
```

**Actions** (Eager - trigger execution):
```python
# This actually executes the plan
count = state_df.count()  # Triggers computation
state_df.write.parquet("path")  # Triggers write
```

**Why lazy?** Spark can optimize the entire plan before executing!

#### 4. Partitions

**Partitions** = How data is split across machines
```python
# Repartition data for better distribution
state_df = state_df.repartition(200, "idx")
```

- **200 partitions** = Data split into 200 chunks
- Each chunk can be processed on a different machine
- More partitions = More parallelism (but more overhead)

#### 5. Broadcast Variables

**Broadcast** = Send small data to all workers
```python
gate_matrix = F.broadcast(small_matrix_df)
# All workers get a copy of gate_matrix
```

- **Gate matrices** are small (2x2 or 4x4)
- Broadcast them to all workers for fast joins
- Avoids shuffling large data

---

## Circuit Partitioning (HiSVSIM)

### The Problem

**Not all gates can run in parallel!**

Example circuit:
```
Gate 0: H on qubit 0
Gate 1: CNOT(0, 1)  ← Depends on Gate 0!
Gate 2: H on qubit 2  ← Independent!
Gate 3: CNOT(1, 2)  ← Depends on Gate 1 and Gate 2!
```

**Dependencies**: Gates that share qubits must run sequentially.

### Solution: Build a Dependency Graph (DAG)

**DAG** = Directed Acyclic Graph
- **Nodes** = Gates
- **Edges** = Dependencies (Gate A → Gate B means A must run before B)

```
    Gate 0 (H on q0)
         ↓
    Gate 1 (CNOT q0→q1)
         ↓
    Gate 3 (CNOT q1→q2)
    
    Gate 2 (H on q2) ──┘
```

### Topological Levels

**Level 0**: Gates with no dependencies
- Gate 0 (H on q0)
- Gate 2 (H on q2)

**Level 1**: Gates that depend only on Level 0
- Gate 1 (CNOT q0→q1)

**Level 2**: Gates that depend on Level 1
- Gate 3 (CNOT q1→q2)

**Key Insight**: All gates in the same level can run in parallel!

### How HiSVSIM Does It

```python
# 1. Build graph
G = build_circuit_graph(gates)
# Creates edges: gate_i → gate_j if they share qubits and i < j

# 2. Find topological levels
levels = topological_levels(G, gates)
# Returns: [[0, 2], [1], [3]]
#         Level0 Level1 Level2

# 3. Process level by level
for level in levels:
    # All gates in this level run in parallel!
    apply_gates_parallel(level)
```

### Partitioning Strategies

**1. Load-Balanced**
- Distribute gates evenly across partitions
- Goal: Equal work per partition

**2. Locality-Aware**
- Minimize qubit overlaps between partitions
- Goal: Reduce communication overhead

**3. Hybrid** (Default)
- Balance both load and locality
- Best overall performance

---

## Level-Based Parallelism

### The Execution Model

Instead of applying gates one-by-one:

**Old Way** (Sequential):
```
Gate 0 → Gate 1 → Gate 2 → Gate 3 → Gate 4 → ...
```

**New Way** (Level-Based):
```
Level 0: [Gate 0, Gate 2] ──┐
                            ├─→ Parallel execution
Level 1: [Gate 1] ──────────┤
                            │
Level 2: [Gate 3, Gate 4] ──┘
```

### How It Works

```python
# Build levels
levels = partition_adapter._topological_levels(graph, gates)
# levels = [[0, 2], [1], [3, 4]]

current_state = initial_state  # |00...0⟩

for level_idx, level in enumerate(levels):
    # Get all gates in this level
    level_gates = [gates[i] for i in level]
    
    # Apply ALL gates in level in parallel
    # Spark builds a lazy plan combining all gates
    current_state = gate_applicator.apply_gates(
        current_state, 
        level_gates
    )
    
    # Save state (triggers execution)
    save_state(current_state)
```

### Why This Is Better

1. **More Parallelism**: Multiple gates per level
2. **Fewer Actions**: One save per level (not per gate)
3. **Better Optimization**: Spark optimizes entire level at once
4. **Less Overhead**: Fewer state saves/loads

---

## State Vector Representation

### Dense vs Sparse

**Dense State** (Traditional):
```python
# All 2^N amplitudes stored
state = [a₀, a₁, a₂, ..., a₂ᴺ₋₁]
# Memory: 2^N × 16 bytes
```

**Sparse State** (Our Approach):
```python
# Only non-zero amplitudes stored
state_df = [
    (idx=0, real=0.707, imag=0.0),   # |000⟩
    (idx=1, real=0.707, imag=0.0),   # |001⟩
    # ... only non-zero entries
]
# Memory: Only non-zero amplitudes × 24 bytes (idx + real + imag)
```

**Why Sparse?**
- Many quantum states have mostly zero amplitudes
- Saves memory for large qubit counts
- Still works for dense states (just stores all amplitudes)

### Spark DataFrame Schema

```python
STATE_SCHEMA = StructType([
    StructField("idx", LongType()),      # Basis state index (0 to 2^N-1)
    StructField("real", DoubleType()),  # Real part of amplitude
    StructField("imag", DoubleType()),  # Imaginary part of amplitude
])
```

**Example**:
```
idx  | real      | imag
-----|-----------|-------
0    | 0.707     | 0.0      ← |000⟩ amplitude
1    | 0.707     | 0.0      ← |001⟩ amplitude
2    | 0.0       | 0.0      ← |010⟩ (zero, not stored)
...
```

### Initial State

```python
# Initial state: |00...0⟩ with amplitude 1
initial_state = spark.createDataFrame(
    [(0, 1.0, 0.0)],  # Only |000...0⟩ has amplitude 1
    schema=STATE_SCHEMA
)
```

---

## Gate Application

### How Gates Transform States

**Quantum Gate** = Unitary matrix that transforms amplitudes

**Example: Hadamard Gate (H)**
```
H = 1/√2 [1  1 ]
         [1 -1 ]

H|0⟩ = 1/√2 (|0⟩ + |1⟩)
H|1⟩ = 1/√2 (|0⟩ - |1⟩)
```

### Applying a Gate in Spark

**Step-by-Step** (for 1-qubit gate):

```python
def apply_one_qubit_gate(state_df, gate_name, qubit):
    # 1. Extract qubit bit from each basis state
    #    idx = 5 (binary: 101) → qubit_bit = (5 >> 1) & 1 = 0
    state_df = state_df.withColumn(
        "qubit_bit", 
        (F.col("idx") >> qubit) & 1
    )
    
    # 2. Join with gate matrix
    #    Gate matrix tells us: qubit_bit=0 → new_bit=0 or 1
    result = state_df.join(
        broadcast(gate_matrix),
        F.col("qubit_bit") == F.col("gate_col")
    )
    
    # 3. Compute new index
    #    Clear old qubit bit, set new bit
    result = result.withColumn(
        "new_idx",
        (F.col("idx") & ~(1 << qubit)) | (F.col("gate_row") << qubit)
    )
    
    # 4. Multiply amplitudes
    #    (a + bi) × (c + di) = (ac - bd) + (ad + bc)i
    result = result.withColumn(
        "new_real",
        F.col("gate_real") * F.col("real") - F.col("gate_imag") * F.col("imag")
    )
    result = result.withColumn(
        "new_imag",
        F.col("gate_real") * F.col("imag") + F.col("gate_imag") * F.col("real")
    )
    
    # 5. Group by new index and sum
    #    Multiple paths can lead to same state (superposition)
    result = result.groupBy("new_idx").agg(
        F.sum("new_real").alias("real"),
        F.sum("new_imag").alias("imag")
    )
    
    # 6. Filter zero amplitudes (sparse representation)
    result = result.filter(
        (F.abs(F.col("real")) > 1e-15) | 
        (F.abs(F.col("imag")) > 1e-15)
    )
    
    return result
```

### Bitwise Operations

**Why bitwise?** Fast and efficient!

- **Extract qubit bit**: `(idx >> qubit) & 1`
- **Clear qubit bit**: `idx & ~(1 << qubit)`
- **Set qubit bit**: `idx | (bit << qubit)`

**Example**: For qubit 2 in state |101⟩ (idx=5):
```
idx = 5 (binary: 101)
qubit = 2

Extract: (5 >> 2) & 1 = 1  ← qubit 2 is 1
Clear:   5 & ~(1 << 2) = 5 & ~4 = 1  ← |001⟩
Set:     1 | (0 << 2) = 1  ← |001⟩ (if new bit is 0)
```

### Gate Matrix Storage

**Gate matrices** are stored as Spark DataFrames:

```python
GATE_MATRIX_SCHEMA = StructType([
    StructField("gate_name", StringType()),  # "H", "CNOT", etc.
    StructField("arity", IntegerType()),      # 1 or 2
    StructField("row", IntegerType()),       # Matrix row (0 or 1)
    StructField("col", IntegerType()),       # Matrix column (0 or 1)
    StructField("real", DoubleType()),       # Real part of matrix element
    StructField("imag", DoubleType()),       # Imaginary part
])
```

**Example** (Hadamard gate):
```
gate_name | arity | row | col | real      | imag
----------|-------|-----|-----|-----------|-----
H         | 1     | 0   | 0   | 0.707     | 0.0
H         | 1     | 0   | 1   | 0.707     | 0.0
H         | 1     | 1   | 0   | 0.707     | 0.0
H         | 1     | 1   | 1   | -0.707    | 0.0
```

**Broadcast**: Gate matrices are small, so we broadcast them to all workers for fast joins.

---

## Fault Tolerance

### Why We Need It

**Problem**: Long-running simulations can crash!
- 30 qubits = ~6 minutes of computation
- Network failures, memory issues, hardware problems

**Solution**: WAL + Checkpoints + Recovery

### Write-Ahead Log (WAL)

**WAL** = Record of operations before they're applied

**Flow**:
```
1. Write WAL PENDING: "About to apply gates 0-10"
2. Apply gates
3. Save state
4. Write WAL COMMITTED: "Gates 0-10 applied successfully"
```

**If crash happens**:
- PENDING entries = Incomplete work (mark as FAILED)
- COMMITTED entries = Completed work (safe to resume from here)

### Checkpoints

**Checkpoint** = Snapshot of state at a specific point

**When to checkpoint?**
- Adaptive: Based on state size, gates processed, time elapsed
- Large states = More frequent checkpoints
- Small states = Less frequent checkpoints

**What's stored?**
- State DataFrame (in Parquet)
- Metadata (version, gate sequence, checksum)

### Recovery Process

**After crash**:
```
1. Load latest checkpoint
   → Get state at gate sequence N

2. Check WAL
   → Find last COMMITTED entry (gate sequence M)
   → Find PENDING entries (mark as FAILED)

3. Resume from max(M, N)
   → Continue simulation from last safe point
```

### Components

**MetadataStore** (DuckDB):
- Stores WAL entries and checkpoint records
- Fast queries for recovery

**CheckpointManager**:
- Creates checkpoints
- Loads checkpoints
- Verifies integrity

**RecoveryManager**:
- Orchestrates recovery
- Determines resume point
- Handles WAL reconciliation

---

## Data Flow

### Complete Flow Example

**Input**: 3-qubit circuit with gates [H(0), CNOT(0,1), H(2)]

```
┌─────────────────────────────────────────┐
│ 1. INITIAL STATE                        │
│    idx | real | imag                    │
│    0   | 1.0  | 0.0  ← |000⟩           │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│ 2. PARTITIONING                         │
│    Level 0: [H(0), H(2)]                │
│    Level 1: [CNOT(0,1)]                 │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│ 3. LEVEL 0: Apply H(0) and H(2)        │
│    WAL PENDING: gates 0-1              │
│                                          │
│    Apply H(0):                          │
│    |000⟩ → 0.707|000⟩ + 0.707|001⟩     │
│                                          │
│    Apply H(2):                          │
│    |000⟩ → 0.707|000⟩ + 0.707|100⟩     │
│    |001⟩ → 0.707|001⟩ + 0.707|101⟩     │
│                                          │
│    Result:                              │
│    idx | real      | imag               │
│    0   | 0.5       | 0.0  ← |000⟩      │
│    1   | 0.5       | 0.0  ← |001⟩      │
│    4   | 0.5       | 0.0  ← |100⟩      │
│    5   | 0.5       | 0.0  ← |101⟩      │
│                                          │
│    Save to Parquet                      │
│    WAL COMMITTED                        │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│ 4. LEVEL 1: Apply CNOT(0,1)            │
│    WAL PENDING: gate 2                  │
│                                          │
│    CNOT flips qubit 1 if qubit 0 is 1  │
│    |000⟩ → |000⟩ (no change)            │
│    |001⟩ → |011⟩ (flip qubit 1)         │
│    |100⟩ → |110⟩ (flip qubit 1)         │
│    |101⟩ → |101⟩ (no change)            │
│                                          │
│    Result:                              │
│    idx | real      | imag               │
│    0   | 0.5       | 0.0  ← |000⟩      │
│    3   | 0.5       | 0.0  ← |011⟩      │
│    4   | 0.5       | 0.0  ← |100⟩      │
│    5   | 0.5       | 0.0  ← |101⟩      │
│                                          │
│    Save to Parquet                      │
│    Create checkpoint (if needed)        │
│    WAL COMMITTED                        │
└─────────────────────────────────────────┘
              ↓
┌─────────────────────────────────────────┐
│ 5. FINAL STATE                          │
│    idx | real      | imag               │
│    0   | 0.5       | 0.0                │
│    3   | 0.5       | 0.0                │
│    4   | 0.5       | 0.0                │
│    5   | 0.5       | 0.0                │
│                                          │
│    Stored in Parquet files              │
└─────────────────────────────────────────┘
```

---

## Key Components

### 1. SparkHiSVSIMDriver (`src/driver.py`)

**Main orchestrator** - Coordinates everything

**Responsibilities**:
- Initialize Spark session
- Parse circuit input
- Coordinate partitioning, execution, and recovery
- Manage WAL and checkpoints

**Key Methods**:
```python
run_circuit(circuit_dict, enable_parallel=True, resume=False)
# Main entry point - runs entire simulation

_simulate_parallel_with_wal(...)
# Parallel execution with fault tolerance

_should_checkpoint(...)
# Adaptive checkpointing logic
```

### 2. HiSVSIMPartitionAdapter (`src/hisvsim/partition_adapter.py`)

**Circuit partitioning** - Finds what can run in parallel

**Key Methods**:
```python
partition_circuit(gates, n_partitions)
# Main partitioning method

_build_circuit_graph(gates)
# Builds dependency graph (DAG)

_topological_levels(graph, gates)
# Finds levels of independent gates
```

### 3. GateApplicator (`src/v2_common/gate_applicator.py`)

**Gate application** - Applies gates using Spark DataFrame operations

**Key Methods**:
```python
apply_gate(state_df, gate)
# Apply single gate

apply_gates(state_df, gates)
# Apply multiple gates (builds lazy plan)

apply_one_qubit_gate(state_df, gate_name, qubit)
# Apply 1-qubit gate with bitwise operations

apply_two_qubit_gate(state_df, gate_name, qubit0, qubit1)
# Apply 2-qubit gate
```

### 4. StateManager (`src/v2_common/state_manager.py`)

**State storage** - Manages state persistence

**Key Methods**:
```python
initialize_state(n_qubits)
# Create initial |00...0⟩ state

save_state(state_df, version)
# Save state to Parquet

load_state(version)
# Load state from Parquet
```

### 5. MetadataStore (`src/v2_common/metadata_store.py`)

**Fault tolerance metadata** - Stores WAL and checkpoint records

**Storage**: DuckDB database

**Key Methods**:
```python
wal_create_pending(...)
# Create WAL entry (PENDING)

wal_mark_committed(wal_id)
# Mark WAL entry as COMMITTED

checkpoint_create(...)
# Create checkpoint record

checkpoint_get_latest(run_id)
# Get latest checkpoint
```

### 6. RecoveryManager (`src/v2_common/recovery_manager.py`)

**Crash recovery** - Handles resumption after failures

**Key Methods**:
```python
recover(n_qubits)
# Perform recovery and return state

get_resume_point(total_gates)
# Determine where to resume

is_simulation_complete(total_gates)
# Check if simulation already finished
```

---

## Performance Optimizations

### 1. Lazy Evaluation

**Problem**: Applying gates one-by-one triggers many actions

**Solution**: Build lazy plan for entire level, then trigger once

```python
# OLD (inefficient):
for gate in gates:
    state = apply_gate(state, gate)  # Action triggered!
    save_state(state)                 # Another action!

# NEW (efficient):
level_gates = [gate1, gate2, gate3]
state = apply_gates(state, level_gates)  # Builds lazy plan
save_state(state)  # Single action triggers all gates!
```

### 2. State Caching

**Problem**: Loading state from disk repeatedly is slow

**Solution**: Cache frequently-used states in memory

```python
state_df = state_df.cache()  # Cache in memory
# Use state_df multiple times
state_df.unpersist()  # Free memory when done
```

### 3. Broadcast Variables

**Problem**: Gate matrices are small but joined repeatedly

**Solution**: Broadcast to all workers

```python
gate_matrix = F.broadcast(small_matrix_df)
# All workers get copy - fast joins!
```

### 4. Adaptive Checkpointing

**Problem**: Checkpointing too often = slow, too rarely = long recovery

**Solution**: Checkpoint based on:
- State size (larger = more frequent)
- Gates processed (more gates = checkpoint)
- Time elapsed (longer = checkpoint)

```python
def _should_checkpoint(state_df, num_gates, version):
    state_size = state_df.count()
    
    # Checkpoint if:
    # - State is large (> threshold)
    # - Many gates processed (> threshold)
    # - Long time since last checkpoint (> threshold)
    
    return (
        state_size > checkpoint_threshold_size or
        self._gates_since_checkpoint > checkpoint_every_n_gates or
        (time.time() - self._last_checkpoint_time) > checkpoint_min_interval
    )
```

### 5. Repartitioning

**Problem**: Data becomes skewed after operations

**Solution**: Repartition periodically for better distribution

```python
state_df = state_df.repartition(200, "idx")
# Redistribute data across 200 partitions
```

### 6. Filter Zero Amplitudes

**Problem**: Storing zero amplitudes wastes memory

**Solution**: Filter out near-zero amplitudes (sparse representation)

```python
state_df = state_df.filter(
    (F.abs(F.col("real")) > 1e-15) | 
    (F.abs(F.col("imag")) > 1e-15)
)
```

---

## Summary

**V3 Architecture** combines:
1. **HiSVSIM partitioning** - Finds parallelizable gates
2. **Spark execution** - Distributes computation efficiently
3. **Level-based parallelism** - Processes independent gates together
4. **Fault tolerance** - WAL, checkpoints, recovery
5. **Optimizations** - Lazy evaluation, caching, adaptive checkpointing

**Tested Non-Stabilizer Gates**:
- **RY gates**: 30 qubits (1.07B amplitudes) - RY(π/4) on each qubit
- **H+T gates**: 25 qubits (33.5M amplitudes) - H then T on each qubit  
- **H+T+CR gates**: Testing 25-30 qubits - H, T, then CR between adjacent qubits
- **G gates**: Testing 25-30 qubits - H then G(p=3) on each qubit
- **R gates**: Testing 25-30 qubits - H then R(k=3) on each qubit
- **CU gates**: Testing 25-30 qubits - H then CU between adjacent qubits

**Maximum Achieved**: **30 qubits** (RY gates, dense state)


