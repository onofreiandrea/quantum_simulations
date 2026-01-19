# How v3 Works

A straightforward explanation of the quantum simulator architecture.

## The Problem

Quantum states grow exponentially. For N qubits, you need 2^N complex numbers:
- 10 qubits = 1,024 amplitudes
- 20 qubits = 1 million amplitudes
- 30 qubits = 1 billion amplitudes (~16 GB)
- 50 qubits = 1 quadrillion amplitudes (~18 petabytes - impossible)

So we split the work across multiple machines using Spark.

## The Solution

1. **Partition the circuit** - Figure out which gates can run in parallel
2. **Run gates in parallel** - Use Spark to distribute work
3. **Combine results** - Merge states back together

## Why Spark?

Spark handles:
- Distributing data across machines
- Fault tolerance (if a machine crashes)
- Parallel execution
- Large datasets (billions of rows)

We use Spark DataFrames to store quantum states - each row is one amplitude with its index and value.

## How Circuit Partitioning Works

Not all gates can run at the same time. If gate A uses qubit 0 and gate B also uses qubit 0, B must wait for A to finish.

**Step 1: Build a dependency graph**
- Each gate is a node
- Draw an arrow from gate A to gate B if B depends on A (they share qubits)

**Step 2: Find levels**
- Level 0: Gates with no dependencies (can all run in parallel)
- Level 1: Gates that only depend on Level 0 gates (can run in parallel after Level 0 finishes)
- Level 2: Gates that depend on Level 0 and Level 1, and so on

**Example:**
```
Gates: H(0), H(2), CNOT(0,1), CNOT(1,2)

Level 0: H(0), H(2)  ← independent, run together
Level 1: CNOT(0,1)   ← needs H(0) done first
Level 2: CNOT(1,2)   ← needs H(2) and CNOT(0,1) done first
```

Within each level, gates run in parallel. Between levels, we wait for the previous level to finish.

## How Gates Are Applied

We store the quantum state as a Spark DataFrame:
```
idx  | real      | imag
-----|-----------|----------
0    | 0.707107  | 0.0
1    | 0.0       | 0.0
2    | 0.0       | 0.0
3    | 0.707107  | 0.0
```

To apply a gate:
1. Extract the qubit bit from the index (using bitwise operations)
2. Join with the gate matrix (small matrices are broadcast to all workers)
3. Multiply amplitudes by matrix values
4. Group by new index and sum (handles superposition)

**Why efficient:**
- Bitwise operations are fast
- Broadcast joins avoid shuffling data
- Parallel partitions mean each worker processes millions of amplitudes

## Fault Tolerance

Long simulations can crash. We use three mechanisms:

**1. Write-Ahead Log (WAL)**
- Before applying gates, write "about to apply gates X-Y" with status PENDING
- After success, mark as COMMITTED
- If crash happens, we know what was in progress

**2. Checkpoints**
- Periodically save the full state to disk
- Store metadata: which gates were applied, state version, checksum

**3. Recovery**
- After crash, load latest checkpoint
- Find last COMMITTED WAL entry
- Mark any PENDING entries as FAILED
- Resume from the checkpoint

**Adaptive checkpointing:**
- Checkpoint if state is large (> threshold)
- Checkpoint if many gates processed (> threshold)
- Checkpoint if long time elapsed (> threshold)
- Balances speed vs recovery time

## The Driver

The `SparkHiSVSIMDriver` orchestrates everything:

1. **Initialize**: Create Spark session, load gate matrices, setup WAL/checkpoints
2. **Partition**: Build dependency graph, find levels
3. **For each level**:
   - Write WAL PENDING
   - Build lazy execution plan (Spark doesn't execute yet)
   - Trigger action (save state) - this forces Spark to execute
   - Checkpoint if needed
   - Mark WAL COMMITTED
   - Load next state
4. **Cleanup**: Stop Spark, close connections

**Key optimization**: We build a plan for all gates in a level, then execute once. This is much faster than executing gate-by-gate.

## State Representation

We use sparse representation - only store non-zero amplitudes. This saves huge amounts of memory.

For 30 qubits:
- Dense: 2^30 × 16 bytes = 16 GB
- Sparse (GHZ): 2 × 24 bytes = 48 bytes (just 2 amplitudes!)

We filter out near-zero amplitudes (below 1e-15) to keep it sparse.

## Performance Optimizations

**1. Lazy evaluation**
- Spark builds an execution plan without running it
- Optimizes the plan (reorder operations, combine steps)
- Only executes when we need results (save state)

**2. Caching**
- Cache frequently-used states in memory
- Unpersist old states to free memory

**3. Repartitioning**
- After operations, data can become skewed
- Repartition periodically for better distribution

**4. Adaptive checkpointing**
- Not too often (slow), not too rare (long recovery)
- Based on state size, gates processed, time elapsed

## Tested Gates and Results

**Non-stabilizer gates tested:**
- **RY gates**: 30 qubits (1.07B amplitudes) - RY(π/4) on each qubit
- **H+T gates**: 25 qubits (33.5M amplitudes) - H then T on each qubit
- **H+T+CR gates**: Testing 25-30 qubits - H, T, then CR between adjacent qubits
- **G gates**: Testing 25-30 qubits - H then G(p=3) on each qubit
- **R gates**: Testing 25-30 qubits - H then R(k=3) on each qubit
- **CU gates**: Testing 25-30 qubits - H then CU between adjacent qubits

**Maximum achieved**: 30 qubits (RY gates, dense state)

## Summary

v3 combines:
1. HiSVSIM partitioning - finds parallelizable gates
2. Spark execution - distributes computation
3. Level-based parallelism - processes independent gates together
4. Fault tolerance - WAL, checkpoints, recovery
5. Optimizations - lazy evaluation, caching, adaptive checkpointing

Result: Can simulate up to 30 qubits (1.07 billion amplitudes) using distributed computing.
