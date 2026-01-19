# What to Say - 5 Minute Presentation

## 1. Introduction (30 seconds)

"Hi. I'll show you our distributed quantum circuit simulator. It combines HiSVSIM partitioning with Apache Spark.

The problem: Quantum simulation needs exponential memory. 30 qubits needs 1 billion amplitudes. 50 qubits needs 18 petabytes - that's impossible.

Our solution: A distributed simulator that handles 30 qubits - 1.07 billion amplitudes - using distributed computing and smart partitioning."

## 2. High-Level Architecture (30 seconds)

"Our system has three parts:

First, we convert the circuit into a dependency graph showing which gates depend on which.

Second, we partition into levels - gates in the same level can run in parallel.

Third, Spark executes these levels across workers, managing the distributed state.

Result: We can simulate beyond what a single machine can handle."

## 3. Circuit Partitioning (1.5 minutes)

"Not all gates can run in parallel. Here's why.

Say we have: Hadamard on qubit 0, then CNOT from qubit 0 to qubit 1, then Hadamard on qubit 2.

The CNOT depends on the first Hadamard because they both use qubit 0. The CNOT reads the state that the Hadamard created, so they must run in order. But the Hadamard on qubit 2 is independent - different qubit - so it can run in parallel with the first Hadamard.

**Building the graph:**

We build a dependency graph. Each gate is a node. If gate A affects qubits that gate B uses, we draw an arrow from A to B.

The algorithm: For each gate, check all previous gates. If they share qubits, create a dependency. This ensures gates execute in the right order.

**Finding levels:**

Once we have the graph, we find levels - groups of gates with no dependencies on each other.

How it works:
1. Level 0: Gates with no dependencies - these can all run in parallel
2. Remove Level 0 gates
3. Level 1: Gates that now have no dependencies - they only depended on Level 0, so they can run in parallel
4. Keep going until all gates are assigned

**Example:**

Gates: H(0), H(2), CNOT(0,1), CNOT(1,2)

- Level 0: H(0) and H(2) - independent, run together
- Level 1: CNOT(0,1) - needs H(0) done first
- Level 2: CNOT(1,2) - needs H(2) and CNOT(0,1) done first

This creates a pipeline: Within each level, gates run in parallel. Between levels, we wait for the previous level to finish.

**Partitioning strategies:**

We have three ways to distribute gates within levels:

Load-balanced: Spread gates evenly. 100 gates, 10 partitions = 10 gates each. Goal: Equal work.

Locality-aware: Group gates that use similar qubits together. Goal: Less data movement.

Hybrid: Our default. Balances both - even distribution plus keeping related gates together. Best performance.

Without partitioning, we'd apply gates one-by-one - very slow. With partitioning, we can apply multiple gates at once, getting roughly 10x speedup."

## 4. Spark Driver (1.5 minutes)

"The Spark Driver is the brain. It's one process that coordinates everything.

**When it starts:**

1. Creates SparkSession - connection to the cluster
2. Sets up components: StateManager, GateApplicator, MetadataStore, CheckpointManager, RecoveryManager
3. Registers gate matrices - small 2x2 or 4x4 matrices that define how gates work

**For each level:**

Step 1: Write WAL PENDING. Before doing anything, write to the log: 'About to process gates 10-19, starting from state version 5, will produce state version 6'. Status: PENDING.

Step 2: Build lazy plan. The driver doesn't execute yet. It calls the gate applicator with all gates in the level, and builds a DataFrame transformation plan. The plan says: 'Join state with matrices, compute new indices, multiply amplitudes, group and sum'. But nothing runs yet - Spark just builds the plan.

Step 3: Trigger execution. Now the driver saves the state to Parquet. This forces Spark to execute the entire plan. Spark optimizes it - maybe reorders operations, combines steps, decides which executors handle which partitions - then runs everything in parallel.

Step 4: Save state. State gets written to Parquet files. Parquet is efficient columnar storage. State is now on disk.

Step 5: Checkpoint if needed. Driver checks: Should I checkpoint? Considers state size, gates processed, time elapsed. If yes, saves checkpoint metadata: state version, which gates were applied, file path, checksum.

Step 6: Mark WAL COMMITTED. Update the log entry to COMMITTED. Means: 'Gates 10-19 completed successfully, state version 6 is valid'.

Step 7: Load next state. Load the new state from Parquet for the next level. Free memory from old cached state.

**Why lazy evaluation matters:**

If we applied gates one-by-one, we'd trigger Spark execution 100 times for 100 gates. Each execution has overhead. Instead, we build a plan for 10 gates, execute once. Spark can optimize: combine joins, reorder operations, push filters down. This gives huge performance gains.

**If something fails:**

WAL entry stays PENDING. On recovery, we know this level didn't complete, mark it FAILED, resume from the previous checkpoint.

The driver is like a conductor - it coordinates everything but doesn't do the actual work. All executors follow its instructions."

## 5. Spark Executors (45 seconds)

"Spark Executors are the workers that apply gates.

Our state is a Spark DataFrame: idx (basis state index), real (real part), imag (imaginary part). We use sparse representation - only non-zero amplitudes.

To apply a gate:
1. Extract qubit bit from index using bitwise operations
2. Join with broadcast gate matrix - small matrices sent to all executors
3. Compute new index
4. Multiply amplitudes
5. Group and sum - multiple paths can lead to same state
6. Filter zeros to keep it sparse

Why efficient: Broadcast joins avoid shuffling data, sparse representation saves memory, parallel partitions mean each executor processes millions of amplitudes at once. For 30 qubits, we might have 200 executors working in parallel."

## 6. Fault Tolerance (45 seconds)

"Long simulations can crash. We have three parts:

Write-Ahead Logging: Log operations as PENDING before execution, COMMITTED after success. Tracks what completed.

Checkpoints: Periodic state snapshots to Parquet with metadata - which gates applied, state version, checksum.

Recovery Manager: After crash, loads latest checkpoint, finds last COMMITTED WAL entry, marks PENDING as FAILED, resumes from safe point.

Adaptive checkpointing: Not too often (slow), not too rare (long recovery). Based on state size, gates processed, time elapsed.

Result: If simulation crashes after 5 minutes, we resume from checkpoint 30 seconds earlier - minimal loss."

## 7. Results (30 seconds)

"We achieved 30 qubits - 1.07 billion amplitudes - beating the previous 26-qubit limit.

Tested non-stabilizer gates: RY gates (30 qubits), H+T gates (25 qubits), H+T+CR, G, R, CU gates (testing 25-30 qubits).

30 qubits gives us 1.07 billion amplitudes in about 6 minutes. Memory usage is 16-24 gigabytes for 30 qubits with sparse representation plus Spark overhead.

While 50 qubits needs 18 petabytes - impossible - we've pushed classical limits. This lets us research larger circuits and validate algorithms before running on quantum hardware."

## 8. Conclusion (15 seconds)

"In summary: Distributed quantum simulator with smart partitioning for parallelism, Spark for scalability, fault tolerance for reliability. We achieved 30 qubits - 1.07 billion amplitudes.

Proper architecture plus distributed computing pushes classical limits further.

Thanks. Questions?"

## Quick Reference

**Problem**: Exponential memory (30 qubits = 1B amplitudes, 50 qubits = 18PB impossible)

**Solution**: Distributed computing + partitioning

**Partitioning**: DAG → topological levels → parallel within levels

**Driver**: Orchestrates, WAL, lazy evaluation, adaptive checkpointing

**Executors**: Bitwise operations, broadcast joins, sparse representation, parallel partitions

**Fault Tolerance**: WAL (PENDING→COMMITTED), checkpoints, recovery manager

**Results**: 30 qubits (RY gates), 1.07B amplitudes. Tested: RY, H+T, H+T+CR, G, R, CU gates
