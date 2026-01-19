# Detailed Explanations: Circuit Partitioning & Spark Driver

## Circuit Partitioning - Complete Technical Explanation

### The Problem

**Why gates can't all run in parallel:**

Quantum gates operate on qubits. When two gates share qubits, they must execute sequentially because:
- Gate A modifies qubit 0
- Gate B reads qubit 0
- Gate B must see Gate A's result, not the original state

**Example:**
```
Gate 0: H(0)        → Creates superposition on qubit 0
Gate 1: CNOT(0,1)   → Reads qubit 0, flips qubit 1 if qubit 0 is |1⟩
Gate 2: H(2)        → Creates superposition on qubit 2
```

Gate 1 depends on Gate 0 (they share qubit 0). Gate 2 is independent (uses qubit 2).

### Building the Dependency Graph (DAG)

**Algorithm:**

```python
# Track which gates use which qubits
qubit_to_gates = {}  # qubit → [list of gate indices]

for gate_idx, gate in enumerate(gates):
    # For each qubit this gate uses
    for qubit in gate.qubits:
        # Add dependency edges from ALL previous gates using this qubit
        for prev_gate_idx in qubit_to_gates[qubit]:
            if prev_gate_idx < gate_idx:
                # Create edge: prev_gate → current_gate
                graph.add_edge(prev_gate, current_gate)
        
        # Record that this gate uses this qubit
        qubit_to_gates[qubit].append(gate_idx)
```

**Why this works:**
- Only creates edges from earlier gates (prev_gate_idx < gate_idx)
- Ensures graph is acyclic (no cycles possible)
- Captures all dependencies (if gates share ANY qubit, there's a dependency)

**Visual Example:**
```
Gates: [H(0), CNOT(0,1), H(2), CNOT(1,2)]

Graph:
H(0) ──→ CNOT(0,1) ──→ CNOT(1,2)
H(2) ──────────────────┘

Dependencies:
- CNOT(0,1) depends on H(0) (share qubit 0)
- CNOT(1,2) depends on H(2) (share qubit 2)
- CNOT(1,2) depends on CNOT(0,1) (share qubit 1)
```

### Topological Sorting - Finding Levels

**Algorithm:**

```python
levels = []
executed = set()
in_degree = {node: graph.in_degree(node) for node in graph.nodes()}

while len(executed) < len(gates):
    # Find all gates with zero dependencies (in_degree == 0)
    current_level = []
    for node in graph.nodes():
        if node not in executed and in_degree[node] == 0:
            current_level.append(gate_index)
    
    # Add this level
    levels.append(current_level)
    executed.update(current_level)
    
    # Update in-degrees: remove edges from executed gates
    for gate_idx in current_level:
        for successor in graph.successors(gate_node):
            in_degree[successor] -= 1  # One less dependency
```

**Step-by-step for example:**

**Initial state:**
- in_degree[H(0)] = 0
- in_degree[H(2)] = 0
- in_degree[CNOT(0,1)] = 1 (depends on H(0))
- in_degree[CNOT(1,2)] = 2 (depends on H(2) and CNOT(0,1))

**Level 0:**
- Gates with in_degree == 0: H(0), H(2)
- Execute H(0) and H(2) in parallel
- Update: in_degree[CNOT(0,1)] = 0, in_degree[CNOT(1,2)] = 1

**Level 1:**
- Gates with in_degree == 0: CNOT(0,1)
- Execute CNOT(0,1)
- Update: in_degree[CNOT(1,2)] = 0

**Level 2:**
- Gates with in_degree == 0: CNOT(1,2)
- Execute CNOT(1,2)
- Done!

**Result:** [[H(0), H(2)], [CNOT(0,1)], [CNOT(1,2)]]

### Partitioning Strategies

**1. Load-Balanced:**

Distributes gates evenly across partitions.

```python
gates_per_partition = len(level_gates) / n_partitions
for i, gate_idx in enumerate(level_gates):
    partition = int(i / gates_per_partition)
    partitions[partition].append(gate_idx)
```

**Example:** 100 gates, 10 partitions → 10 gates per partition

**2. Locality-Aware:**

Groups gates by qubit usage to minimize data movement.

```python
# Group gates by qubit sets
qubit_groups = {}
for gate_idx in level_gates:
    qubits = tuple(sorted(gate.qubits))
    if qubits not in qubit_groups:
        qubit_groups[qubits] = []
    qubit_groups[qubits].append(gate_idx)

# Distribute groups across partitions
```

**Example:** Gates using qubits {0,1} go together, gates using {2,3} go together

**3. Hybrid:**

Balances load and locality.

```python
# Sort gates by qubit usage
# Distribute while keeping qubit groups together when possible
# But ensure partitions are roughly equal size
```

---

## Spark Driver - Complete Technical Explanation

### Driver Initialization

**What happens when driver starts:**

```python
# 1. Create SparkSession
spark = SparkSession.builder
    .master("local[*]")  # Use all CPU cores
    .appName("QuantumSimulator")
    .config("spark.driver.memory", "4g")
    .config("spark.executor.memory", "4g")
    .getOrCreate()

# 2. Initialize components
state_manager = StateManager(spark, config)
gate_applicator = GateApplicator(spark, partitions=200)
partition_adapter = HiSVSIMPartitionAdapter(strategy="hybrid")
metadata_store = MetadataStore(config)  # DuckDB for WAL/checkpoints
checkpoint_manager = CheckpointManager(...)
recovery_manager = RecoveryManager(...)
```

**Components explained:**
- **StateManager**: Handles reading/writing state DataFrames to Parquet
- **GateApplicator**: Applies gates using Spark DataFrame operations
- **PartitionAdapter**: Builds DAG and finds levels
- **MetadataStore**: Stores WAL entries and checkpoint records in DuckDB
- **CheckpointManager**: Creates/loads checkpoints
- **RecoveryManager**: Handles crash recovery

### Execution Flow - Per Level

**Complete workflow with code explanation:**

```python
for level_idx, level in enumerate(levels):
    # LEVEL PREPARATION
    level_gates = [gates[i] for i in level]
    num_gates = len(level_gates)
    level_start_seq = start_gate_seq + gate_idx
    level_end_seq = start_gate_seq + gate_idx + num_gates
    
    version_in = current_version      # e.g., version 5
    version_out = current_version + 1  # e.g., version 6
    
    # STEP 1: Write WAL PENDING
    wal_id = metadata_store.wal_create_pending(
        run_id="abc123",
        gate_start=level_start_seq,      # e.g., gate 10
        gate_end=level_end_seq,          # e.g., gate 20
        state_version_in=version_in,     # version 5
        state_version_out=version_out    # version 6
    )
    # WAL entry: "About to process gates 10-19, v5 → v6, status=PENDING"
    
    try:
        # STEP 2: Build Lazy Execution Plan
        # This is KEY - no execution happens yet!
        output_state = gate_applicator.apply_gates(
            current_state,  # DataFrame with state version 5
            level_gates     # List of Gate objects
        )
        # What happens:
        # - GateApplicator builds DataFrame transformations
        # - For each gate: join, compute, group, filter
        # - Chains them together: gate1 → gate2 → gate3
        # - Returns a DataFrame that represents the PLAN
        # - NO DATA PROCESSED YET!
        
        # STEP 3: Trigger ACTION - Execute Plan
        state_path = state_manager.save_state(output_state, version_out)
        # What happens:
        # - save_state() calls: output_state.write.parquet(path)
        # - This is an ACTION - triggers execution!
        # - Spark looks at the entire plan
        # - Optimizes: reorders operations, combines steps
        # - Executes across all executors in parallel
        # - Each executor processes its partition
        # - Results written to Parquet files
        
        # STEP 4: Adaptive Checkpointing Decision
        should_checkpoint = _should_checkpoint(
            output_state,    # New state DataFrame
            num_gates,       # Gates in this level
            version_out      # New version number
        )
        # Checks:
        # - state_df.count() > checkpoint_threshold_size (1M)?
        # - gates_since_checkpoint > checkpoint_every_n_gates (10)?
        # - time_since_checkpoint > checkpoint_min_interval (60s)?
        
        if should_checkpoint:
            checkpoint_manager.create_checkpoint(
                state_version=version_out,
                last_gate_seq=level_end_seq - 1,
                state_path=state_path
            )
            # Saves to DuckDB:
            # - State version: 6
            # - Last gate: 19
            # - Path: data/state/run_id=abc/state_version=6/
            # - Checksum: abc123def456
            # - Timestamp: 2025-01-16 13:00:00
        
        # STEP 5: Mark WAL COMMITTED
        metadata_store.wal_mark_committed(wal_id)
        # Updates WAL entry:
        # - Status: PENDING → COMMITTED
        # - committed_at: current timestamp
        # - Now we know gates 10-19 completed successfully
        
        # STEP 6: Cleanup and Load Next State
        if current_state.is_cached:
            current_state.unpersist(blocking=False)
            # Free memory from old state
        
        current_state = state_manager.load_state(version_out)
        # Load state version 6 from Parquet for next level
        current_version = version_out
        
    except Exception as e:
        # ERROR HANDLING
        metadata_store.wal_mark_failed(wal_id)
        # Mark WAL as FAILED - this level didn't complete
        raise  # Re-raise to stop execution
```

### Lazy Evaluation - Deep Dive

**Why lazy evaluation matters:**

**Without lazy evaluation (inefficient):**
```python
state = initial_state
for gate in gates:
    state = apply_gate(state, gate)  # ACTION - triggers execution!
    save_state(state)                 # Another ACTION!
# Result: 100 gates = 200 actions = 200 Spark job submissions
```

**With lazy evaluation (efficient):**
```python
state = initial_state
level_gates = [gate1, gate2, ..., gate10]
state = apply_gates(state, level_gates)  # Builds PLAN - no execution!
save_state(state)  # Single ACTION - executes entire plan!
# Result: 10 gates = 1 action = 1 Spark job submission
```

**What Spark optimizes:**

1. **Predicate pushdown**: Filter operations moved earlier
2. **Projection pushdown**: Only select needed columns
3. **Join reordering**: Optimize join order
4. **Combining operations**: Merge multiple transformations
5. **Partition pruning**: Skip partitions that don't contain needed data

**Example optimization:**
```
Original plan:
  Filter (amplitude > 0) → Join → Group → Filter (idx < 1000)

Optimized plan:
  Filter (amplitude > 0 AND idx < 1000) → Join → Group
  (Filters combined and pushed down)
```

### Adaptive Checkpointing Logic

**Decision function:**

```python
def _should_checkpoint(state_df, num_gates, version):
    state_size = state_df.count()  # Number of amplitudes
    
    # Checkpoint if ANY condition met:
    return (
        state_size > checkpoint_threshold_size or           # Large state
        self._gates_since_checkpoint > checkpoint_every_n_gates or  # Many gates
        (time.time() - self._last_checkpoint_time) > checkpoint_min_interval  # Long time
    )
```

**Why adaptive:**

- **Small states** (sparse): Checkpoint less frequently (saves time)
- **Large states** (dense): Checkpoint more frequently (faster recovery)
- **Many gates**: Checkpoint to avoid losing too much work
- **Long time**: Checkpoint to ensure progress saved

**Example:**
- State with 100K amplitudes: Checkpoint every 20 gates
- State with 10M amplitudes: Checkpoint every 5 gates
- State with 1B amplitudes: Checkpoint every level

### Error Handling and Recovery

**What happens on crash:**

```python
# Simulation crashes mid-execution
# WAL entries:
# - Gates 0-9: COMMITTED
# - Gates 10-19: PENDING (crashed here)
# - Gates 20-29: Not started

# Recovery process:
recovery_state = recovery_manager.recover(n_qubits)

# 1. Load latest checkpoint
checkpoint = checkpoint_manager.load_latest_checkpoint()
# Returns: state_version=5, last_gate=9

# 2. Check WAL
pending = metadata_store.wal_get_pending(run_id)
# Finds: Gates 10-19 PENDING

# 3. Mark PENDING as FAILED
for entry in pending:
    metadata_store.wal_mark_failed(entry.wal_id)

# 4. Resume from checkpoint
resume_from_gate = checkpoint.last_gate_seq + 1  # Gate 10
resume_from_version = checkpoint.state_version    # Version 5

# 5. Continue execution from gate 10
```

**Result:** Lost only gates 10-19, resume from gate 10 with state version 5.

---

## Key Takeaways

**Circuit Partitioning:**
- DAG captures all dependencies
- Topological sorting finds parallelizable groups
- Levels execute sequentially, gates within level execute in parallel
- Strategies balance load and locality

**Spark Driver:**
- Single coordinator managing entire workflow
- Lazy evaluation builds optimized plans
- WAL tracks progress for fault tolerance
- Adaptive checkpointing balances performance and recovery
- Error handling ensures no data loss
