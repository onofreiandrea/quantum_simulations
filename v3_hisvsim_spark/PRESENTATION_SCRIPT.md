# V3 Quantum Simulator - What to Say (5 Minutes)

## 1. Introduction (30 seconds)

**Say this:**

"Good [morning/afternoon]. I'll present our distributed quantum circuit simulator combining HiSVSIM partitioning with Apache Spark.

The challenge: Classical quantum simulation requires exponential memory - 30 qubits needs 1 billion amplitudes, 50 qubits needs 18 petabytes, which is impossible.

Our solution: A distributed simulator handling 30 qubits - 1.07 billion amplitudes - using distributed computing and intelligent partitioning."

---

## 2. High-Level Architecture (30 seconds)

**Say this:**

"Our system has three layers:

First: Convert circuit to dependency graph showing gate dependencies.

Second: Partition into independent levels - gates in same level execute in parallel.

Third: Spark executes levels across workers, managing distributed state.

Result: Scalable simulation beyond single-machine limits."

---

## 3. Circuit Partitioning - DETAILED (1.5 minutes)

**Say this:**

"Circuit partitioning is crucial because not all gates can run in parallel. Let me explain exactly how this works.

**The Problem in Detail:**

Imagine we have a circuit: First, a Hadamard gate on qubit 0, then a CNOT gate from qubit 0 to qubit 1, then another Hadamard on qubit 2.

The CNOT gate depends on the first Hadamard because they both use qubit 0. The CNOT reads the state of qubit 0 that the Hadamard created, so they must run sequentially. However, the Hadamard on qubit 2 is independent - it uses a different qubit - so it can run in parallel with the first Hadamard.

**Building the Dependency Graph:**

We build what's called a Directed Acyclic Graph, or DAG. Think of it like a flowchart:
- Each **node** is a gate
- Each **edge** represents a dependency - if gate A affects qubits that gate B uses, we draw an arrow from A to B

The algorithm works like this: For each gate, we check all previous gates. If a previous gate uses any of the same qubits, we create a dependency edge. This ensures gates that share qubits execute in order.

**Topological Sorting - Finding Levels:**

Once we have the graph, we perform topological sorting. This finds groups of gates that have no dependencies on each other - these are our levels.

Here's how it works step by step:
1. **Level 0**: Find all gates with zero incoming edges - these have no dependencies, so they can all run in parallel
2. Remove Level 0 gates from consideration
3. **Level 1**: Find gates that now have zero dependencies - they only depended on Level 0 gates, so they can run in parallel
4. Continue until all gates are assigned to levels

**Example:**

If we have gates: H(0), H(2), CNOT(0,1), CNOT(1,2)
- Level 0: H(0) and H(2) - independent, parallel
- Level 1: CNOT(0,1) - depends on H(0), parallelizable with other Level 1 gates
- Level 2: CNOT(1,2) - depends on H(2) and CNOT(0,1)

**The Pipeline:**

This creates a pipeline architecture: Within each level, gates execute in parallel across Spark executors. Between levels, we execute sequentially - Level 0 completes, then Level 1, then Level 2, and so on.

**Partitioning Strategies:**

We have three strategies for distributing gates within levels:

**Load-balanced**: Distributes gates evenly across partitions. If we have 100 gates in a level and 10 partitions, each partition gets 10 gates. Goal: Equal work per partition.

**Locality-aware**: Groups gates that use similar qubits together. If gates A, B, C all use qubits 0-3, they go in the same partition. Goal: Minimize data movement between partitions.

**Hybrid**: Our default strategy. It balances both - tries to distribute gates evenly while keeping gates with qubit overlaps together. This gives us the best overall performance.

**Why This Matters:**

Without partitioning, we'd apply gates one-by-one sequentially - very slow. With partitioning, we can apply multiple gates simultaneously, dramatically speeding up execution. For a circuit with 100 gates, if we find 10 levels with 10 gates each, we get roughly 10x speedup from parallelism."

---

## 4. Spark Driver - The Orchestrator - DETAILED (1.5 minutes)

**Say this:**

"The Spark Driver is the brain of our system. It's a single process running on one machine that coordinates the entire distributed simulation. Let me explain exactly what it does.

**Initialization:**

When the driver starts, it:
1. Creates a SparkSession - this is the connection to the Spark cluster
2. Initializes all components: StateManager for storing states, GateApplicator for applying gates, MetadataStore for WAL and checkpoints, CheckpointManager and RecoveryManager for fault tolerance
3. Registers all gate matrices - these are small 2x2 or 4x4 matrices that define how each gate transforms the quantum state

**Circuit Processing:**

The driver takes the input circuit - a list of gates - and passes it to the partition adapter. The adapter builds the dependency graph and finds topological levels, as I explained earlier.

**Execution Loop - Per Level:**

For each level, the driver orchestrates a complete workflow:

**Step 1: Write WAL PENDING**
Before doing anything, the driver writes a Write-Ahead Log entry to DuckDB. This entry says: 'I'm about to process gates 10 through 19, starting from state version 5, will produce state version 6'. Status: PENDING. This is like writing in a logbook before starting work.

**Step 2: Build Lazy Execution Plan**
Here's the key optimization: The driver doesn't immediately execute. Instead, it builds a lazy execution plan. It calls the gate applicator with all gates in the level, and the gate applicator builds a DataFrame transformation plan. This plan says: 'Join state with gate matrices, compute new indices, multiply amplitudes, group and sum'. But nothing executes yet - Spark just builds the plan.

**Step 3: Trigger Action - Execute**
Now the driver triggers an action - it saves the state to Parquet. This forces Spark to actually execute the entire lazy plan. Spark looks at the plan, optimizes it - maybe reorders operations, combines steps, decides which executors handle which partitions - then executes everything in parallel across all executors.

**Step 4: Save State**
The state DataFrame is written to Parquet files in a directory structure: data/state/run_id=X/state_version=Y/. Parquet is columnar storage - very efficient for our data. The state is now durably stored on disk.

**Step 5: Adaptive Checkpointing**
The driver checks: Should I create a checkpoint? It considers three factors:
- State size: If state has more than 1 million amplitudes, checkpoint
- Gates processed: If we've processed more than 10 gates since last checkpoint, checkpoint
- Time elapsed: If more than 60 seconds since last checkpoint, checkpoint

If checkpointing is needed, the driver saves checkpoint metadata to DuckDB: state version, which gates were applied, path to state files, checksum for verification.

**Step 6: Mark WAL COMMITTED**
Finally, the driver updates the WAL entry status from PENDING to COMMITTED. This means: 'Gates 10-19 completed successfully, state version 6 is valid'.

**Step 7: Load Next State**
The driver loads the new state from Parquet for the next level. It unpersists the old cached state to free memory, then loads the new one.

**Why Lazy Evaluation Matters:**

This is crucial. If we applied gates one-by-one, we'd trigger Spark execution 100 times for 100 gates. Each execution has overhead. Instead, we build a plan for 10 gates, execute once. Spark can optimize: combine multiple joins, reorder operations, push filters down. This gives us massive performance gains.

**Error Handling:**

If anything fails - say an executor crashes or we run out of memory - the WAL entry stays PENDING. On recovery, we know this level didn't complete, so we mark it FAILED and resume from the previous checkpoint.

**The Driver's Role:**

The driver is like a conductor - it doesn't play the music, but it coordinates all the musicians. It manages the workflow, tracks progress, handles errors, and ensures everything happens in the right order. All executors follow the driver's instructions."

---

## 5. Spark Executors - The Workers (45 seconds)

**Say this:**

"Spark Executors are workers that apply quantum gates to state vectors.

Our state is a Spark DataFrame with three columns: idx - the basis state index, real - real part of amplitude, imag - imaginary part. We use sparse representation - only non-zero amplitudes.

Gate application uses bitwise operations:
1. Extract qubit bit from basis state index
2. Join with broadcast gate matrix - small matrices broadcasted to all executors
3. Compute new index by clearing old bit and setting new bit
4. Multiply amplitudes using complex arithmetic
5. Group and sum - multiple paths can lead to same state
6. Filter zeros to maintain sparsity

Why efficient: Broadcast joins avoid data shuffling, sparse representation saves memory, and parallel partitions mean each executor processes millions of amplitudes simultaneously. For 30 qubits, we might have 200 executors working in parallel."

---

## 6. Fault Tolerance (45 seconds)

**Say this:**

"Long simulations can crash. Our fault tolerance has three components:

First, Write-Ahead Logging: We log operations as PENDING before execution, COMMITTED after success. This tracks exactly what completed.

Second, Checkpoints: Periodic state snapshots to Parquet with metadata - which gates were applied, state version, checksum.

Third, Recovery Manager: After crash, it loads the latest checkpoint, finds the last COMMITTED WAL entry, marks PENDING as FAILED, and resumes from the safe point.

Adaptive checkpointing balances frequency - not too often, which slows execution, or too rare, which means long recovery. It's based on state size, gates processed, and time elapsed.

Result: If simulation crashes after 5 minutes, we resume from checkpoint 30 seconds earlier - minimal progress loss."

---

## 7. Results (30 seconds)

**Say this:**

"We achieved 30 qubits - 1.07 billion amplitudes - exceeding the previous 26-qubit limit.

Tested non-stabilizer gates: RY gates (30 qubits), H+T gates (25 qubits), H+T+CR, G, R, CU gates (testing 25-30 qubits).

Key metrics: 30 qubits gives us 1.07 billion amplitudes in ~6 minutes. Memory usage is about 16 to 24 gigabytes for 30 qubits with sparse representation plus Spark overhead.

Impact: While 50 qubits needs 18 petabytes - impossible - we've pushed classical limits significantly. This enables research on larger circuits and validates algorithms before running on quantum hardware."

---

## 8. Conclusion (15 seconds)

**Say this:**

"In summary, we built a distributed quantum simulator with intelligent partitioning to maximize parallelism, Spark distributed computing for scalability, fault tolerance for reliability, and we achieved 30 qubits - 1.07 billion amplitudes.

Proper architecture plus distributed computing pushes classical limits further than previously possible.

Thank you. Questions?"

---

## Quick Reference - Key Points to Remember

**Problem**: Exponential memory (30 qubits = 1B amplitudes, 50 qubits = 18PB impossible)

**Solution**: Distributed computing + partitioning

**Partitioning**: DAG → topological levels → parallel within levels

**Driver**: Orchestrates, WAL, lazy evaluation, adaptive checkpointing

**Executors**: Bitwise operations, broadcast joins, sparse representation, parallel partitions

**Fault Tolerance**: WAL (PENDING→COMMITTED), checkpoints, recovery manager

**Results**: 30 qubits (RY gates), 1.07B amplitudes. Tested: RY, H+T, H+T+CR, G, R, CU gates
