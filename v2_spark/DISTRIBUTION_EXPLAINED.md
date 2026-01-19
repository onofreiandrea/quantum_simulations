# How Distribution Works in the Spark Quantum Simulator

## Current Distribution Strategy

### What Gets Distributed: **State Vector, Not Circuit**

The circuit is **NOT** distributed across workers. Instead, we distribute the **state vector** (the quantum amplitudes).

### How It Works

```
┌─────────────────────────────────────────────────────────────┐
│  Gate Application (Sequential - NOT Distributed)            │
├─────────────────────────────────────────────────────────────┤
│  1. Apply Gate H[0]                                         │
│     ↓                                                        │
│  2. Apply Gate CNOT[0,1]                                    │
│     ↓                                                        │
│  3. Apply Gate H[2]                                         │
│     ↓                                                        │
│  ... (one gate at a time)                                   │
└─────────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────────┐
│  State Vector Distribution (After Each Gate)                │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  State DataFrame partitioned by idx (hash partitioning):    │
│                                                              │
│  Worker 1:  idx % num_partitions ∈ {0, 4, 8, 12, ...}      │
│  Worker 2:  idx % num_partitions ∈ {1, 5, 9, 13, ...}      │
│  Worker 3:  idx % num_partitions ∈ {2, 6, 10, 14, ...}      │
│  Worker 4:  idx % num_partitions ∈ {3, 7, 11, 15, ...}      │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

## Code Flow

### 1. Gate Application (Sequential)

```python
# In driver.py - gates are applied ONE AT A TIME
for batch in batches:
    for gate in batch.gates:
        state_df = gate_applicator.apply_gate(state_df, gate)  # Sequential!
```

### 2. State Repartitioning (After Each Gate)

```python
# In gate_applicator.py - after applying a gate:
result = (
    state_df
    .join(gate_matrix)  # Broadcast join (small, replicated to all workers)
    .withColumn("new_idx", ...)  # Compute new indices
    .groupBy("new_idx")  # Shuffle happens here
    .agg(F.sum("new_real"), F.sum("new_imag"))
)

# KEY: Repartition by idx to distribute across workers
if self.num_partitions > 1:
    result = result.repartition(self.num_partitions, "idx")
```

### 3. Hash Partitioning Strategy

Spark uses **hash partitioning** on `idx`:

```python
partition_id = hash(idx) % num_partitions
```

This means:
- `idx=0` → partition `hash(0) % 16` (e.g., partition 0)
- `idx=1` → partition `hash(1) % 16` (e.g., partition 5)
- `idx=2` → partition `hash(2) % 16` (e.g., partition 10)
- etc.

## What This Means

### ✅ What IS Distributed:
- **State vector rows**: Each amplitude `(idx, real, imag)` is on one partition/worker
- **Gate application computation**: When applying a gate, each worker processes its partition
- **GroupBy aggregation**: Shuffle happens across workers

### ❌ What is NOT Distributed:
- **Circuit gates**: Applied sequentially, one at a time
- **Gate batching**: Batches are processed sequentially
- **Gate matrices**: Broadcast to all workers (small, not distributed)

## Example: Applying H Gate to 3-Qubit State

### Initial State (|000⟩):
```
Worker 1: [(idx=0, real=1.0, imag=0.0)]  ← Single row
```

### After H[0]:
```
Worker 1: [(idx=0, real=0.707, imag=0.0)]   ← |000⟩
Worker 2: [(idx=1, real=0.707, imag=0.0)]   ← |001⟩
```

### After H[1]:
```
Worker 1: [(idx=0, real=0.5, imag=0.0)]     ← |000⟩
Worker 2: [(idx=1, real=0.5, imag=0.0)]     ← |001⟩
Worker 3: [(idx=2, real=0.5, imag=0.0)]     ← |010⟩
Worker 4: [(idx=3, real=0.5, imag=0.0)]     ← |011⟩
```

Each worker processes its partition independently during gate application.

## Why This Works

1. **State grows exponentially**: 2^n amplitudes → perfect for distribution
2. **Gate operations are parallelizable**: Each amplitude transformation is independent
3. **GroupBy requires shuffle**: Natural distribution point
4. **Sparse states benefit**: Only non-zero amplitudes stored

## Limitations

### Current Approach:
- ✅ Good for **dense states** (QFT, random circuits)
- ✅ Good for **large state vectors** (many amplitudes)
- ❌ **NOT good for sparse states** (GHZ only has 2 rows - can't distribute)
- ❌ **Sequential gate application** (no circuit-level parallelism)

### What We're NOT Doing:
- ❌ **Gate-level parallelism**: Can't apply multiple gates simultaneously
- ❌ **Circuit-level distribution**: Can't split circuit across workers
- ❌ **Pipeline parallelism**: Can't overlap gate applications

## Potential Improvements

### 1. Gate-Level Parallelism (Future)
```python
# Apply independent gates in parallel
independent_gates = find_independent_gates(gates)
for gate_group in independent_gates:
    # Apply all gates in group simultaneously
    state_df = apply_gates_parallel(state_df, gate_group)
```

### 2. Circuit Splitting (Future)
```python
# Split circuit into sub-circuits
subcircuits = split_circuit(circuit, n_workers)
# Each worker simulates a sub-circuit
# Combine results at the end
```

### 3. Better Partitioning Strategy
```python
# Instead of hash partitioning, use range partitioning
# for better locality
result = result.repartitionByRange(self.num_partitions, "idx")
```

## Current Distribution Summary

| Aspect | Status | Details |
|--------|--------|---------|
| **State Distribution** | ✅ Yes | Hash partitioned by `idx` |
| **Gate Distribution** | ❌ No | Applied sequentially |
| **Circuit Distribution** | ❌ No | Entire circuit on one driver |
| **Gate Matrix Distribution** | ❌ No | Broadcast (small, replicated) |
| **Computation Distribution** | ✅ Yes | Each worker processes its partition |

**Bottom Line**: We distribute the **state vector computation**, not the circuit structure.
