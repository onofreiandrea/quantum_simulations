# System Architecture

## Overview

`wenbo_engine` is an **out-of-core quantum state vector simulator**.
It stores the full state vector (2^n complex amplitudes) on disk as
dense chunk files and streams them through CPU kernels one chunk at a time.

Design principles:
- **Spark is orchestration only** — no DataFrame joins/groupBy for amplitudes.
- **Dense storage** — chunk files are raw `complex64` arrays on disk.
- **Atomic writes** — every chunk and manifest write uses `tmp + fsync + os.replace`.
- **Double-buffer recovery** — source is never modified; safe to retry on crash.
- **CPU first, GPU later** — kernels use NumPy/BLAS; GPU path is a future extension.

---

## Package structure

```
wenbo_engine/
├── circuit/              # Circuit input layer
│   ├── io.py             #   validate, normalize, levelize
│   ├── import_qiskit.py  #   Qiskit QuantumCircuit → circuit dict
│   ├── fusion.py         #   1Q gate fusion + level batching
│   └── staging.py        #   Atlas-style circuit staging + qubit remapping
│
├── kernel/               # Numerical kernels (pure math, no I/O)
│   ├── gates.py          #   Canonical gate matrices (2×2, 4×4)
│   ├── ref_dense.py      #   In-memory dense simulator (oracle, n ≤ 20)
│   ├── cpu_scalar.py     #   Scalar 1Q/2Q kernels (loop-based)
│   ├── cpu_batched.py    #   Batched GEMM kernels (gather → matmul → scatter)
│   └── cpu_nonlocal.py   #   Non-local pair/quad array operations
│
├── storage/              # On-disk state management
│   ├── manifest.py       #   Manifest read/write/validate
│   ├── block_store.py    #   Chunk read/write, init |0⟩ state
│   └── versioning.py     #   Version directory layout (legacy, unused)
│
├── runner/               # Execution engines
│   ├── single_node.py    #   Sequential out-of-core runner
│   ├── pipeline.py       #   Threaded reader→worker→writer pipeline
│   └── spark_runner.py   #   Spark-distributed runner (orchestration only)
│
├── wal/                  # Write-ahead log + recovery
│   ├── wal.py            #   Atomic JSON WAL (step-level commit)
│   ├── recovery.py       #   Crash recovery entry point
│   └── fencing.py        #   Fencing lock (prevents concurrent runs)
│
├── bench/                # Benchmarks
│   ├── io.py             #   Sequential I/O bandwidth
│   ├── kernel.py         #   Scalar vs batched kernel throughput
│   ├── matmul_vs_io.py   #   Compute vs I/O bottleneck analysis
│   ├── hyperparam_sweep.py # Chunk size / buffer depth / fusion sweep
│   ├── end_to_end.py     #   Full simulation throughput
│   └── mqt_bench_runner.py # MQT Bench circuits: correctness + perf
│
├── tests/                # Test suite (pytest)
│   ├── fixtures/         #   Shared circuit fixtures
│   ├── test_contract.py
│   ├── test_endianness_lock.py
│   ├── test_ref_known_states.py
│   ├── test_qiskit_oracle.py
│   ├── test_kernel_vs_ref.py
│   ├── test_nonlocal.py
│   ├── test_out_of_core_e2e.py
│   ├── test_recovery_crash.py
│   ├── test_spark_runner_small.py
│   ├── test_fusion.py
│   └── test_staging.py
│
└── docs/                 # Documentation
    ├── circuit_contract.md
    ├── storage_spec.md
    ├── non_local_gates.md
    ├── recovery_strategies.md
    ├── architecture.md        ← this file
    └── v3_comparison.md
```

---

## Data flow

```
              ┌─────────────────────────────────────┐
              │         Circuit dict (JSON)         │
              │    { n_qubits, gates: [...] }       │
              └─────────────────┬───────────────────┘
                                │
                           validate
                                │
                                ▼
              ┌─────────────────┴───────────────────┐
              │                                     │
      (default / fusion)                    (staging mode)
      levelize → batch_levels            atlas_stages(method=…)
          + fuse_1q                     operates on gate order,
              │                         picks local qubits per
              │                         stage (ILP / heuristic /
              │                         greedy), inserts SWAPs
              │                                     │
              └─────────────────┬───────────────────┘
                                │
                                ▼
              ┌─────────────────────────────────────┐
              │  Steps: [{ local_ops,               │
              │            nonlocal_ops }, ...]     │
              └─────────────────┬───────────────────┘
                                │
                                ▼
              ┌─────────────────────────────────────┐
              │              Runner                 │
              │                                     │
              │  for each step:                     │
              │    src = committed buffer (a or b)  │
              │    dst = other buffer               │
              │                                     │
              │    for each work item:              │
              │      read chunk(s) from src         │
              │      apply kernel (scalar / batched)│
              │      write chunk(s) to dst (atomic) │
              │                                     │
              │    write manifest to dst            │
              │    WAL commit (swap committed buf)  │
              └─────────────────────────────────────┘
```

---

## State storage layout

Each buffer is a self-contained directory:

```
work_dir/
├── state_a/
│   ├── manifest.json
│   └── chunks/
│       ├── chunk_000000.bin    (complex64, raw bytes)
│       ├── chunk_000001.bin
│       └── ...
├── state_b/
│   ├── manifest.json
│   └── chunks/
│       └── ...
├── wal.json
└── qubit_mapping.json   (only if staging produced a non-identity permutation)
```

- **chunk_size** (default 2^20 = 1M amplitudes = 8 MB) is tunable.
- **n_chunks** = 2^n / chunk_size.
- Chunks are numbered contiguously; chunk `i` holds amplitudes
  `[i * chunk_size .. (i+1) * chunk_size)`.

---

## Gate execution model

### Levelization

Gates are grouped into **levels** — each level contains gates that
share no qubits, so they can be applied in any order within the level.

```
Circuit:  H(0), CNOT(0,1), H(2)
Level 0:  [H(0), H(2)]       ← independent
Level 1:  [CNOT(0,1)]        ← depends on H(0)
```

### Local vs non-local classification

Given `k = log2(chunk_size)`:

- **Local gate:** all target qubits < k.
  Both amplitudes to be paired live within the same chunk.
  Process one chunk at a time — no inter-chunk communication.

- **Non-local gate:** at least one target qubit ≥ k.
  Paired amplitudes live in different chunks.
  Requires loading a **group** of partner chunks simultaneously
  (butterfly exchange pattern).

### Fusion optimizations

1. **Level batching:** Consecutive all-local levels are merged into a
   single step (one I/O pass instead of one per level).

2. **1Q gate fusion:** Consecutive single-qubit gates on the same qubit
   (with no intervening 2Q gate on that qubit) are pre-multiplied into
   a single 2×2 matrix.

Both reduce the number of I/O passes over the full state vector.

### Circuit staging (`use_staging=True`)

Circuit staging physically rearranges qubits so that more gates become
chunk-local, reducing the number of expensive I/O passes.  Our
implementation is based on the **Atlas** system (Xu et al., "Atlas:
Hierarchical Partitioning for Quantum Circuit Simulation on GPUs",
SC'24, arXiv:2408.09055; https://github.com/quantum-compiler/atlas).

**Core idea:** Given `k = log2(chunk_size)`, only qubits at physical
positions 0..k-1 are "local".  Staging partitions the circuit into
**stages**, each with a chosen set of k local qubits.  Between stages,
SWAP operations rearrange the state vector so that the next stage's
local qubits occupy positions 0..k-1.

**Three staging methods** (selected via `staging_method` in the runner
or `method` in `atlas_stages()`):

1. **`"ilp"`** — Integer Linear Program (requires PuLP).  Faithful
   reimplementation of Atlas's `compute_local_qubits_with_ilp`.
   Binary-searches on the number of stages S, solving an ILP that:
   - Assigns each gate to exactly one stage.
   - Respects gate dependencies (topological ordering).
   - Requires all non-insular qubits of a gate to be local in its stage.
   - Enforces exactly k local qubits per stage.
   - Minimises total qubit transitions (SWAP cost) between stages.
   Produces the optimal partitioning for the given S.

2. **`"heuristic"`** *(default)* — Dependency-aware greedy, ported from
   Atlas's `num_iterations_by_heuristics` (circuit.cc).  Iteratively:
   - Executes all gates whose non-insular qubits are in the current
     local set, respecting dependency order.
   - When blocked, selects the next local set by priority: qubits in
     the first unexecuted gate > global gate count > local gate count
     > qubit index.

3. **`"greedy"`** — Simple frequency-based lookahead (original, pre-Atlas).
   Scans gates in order, accumulates local gates, and on encountering
   a non-local gate, uses a 200-gate lookahead to pick the most frequent
   qubits as the next local set.

**Insular qubit optimisation** (Atlas §3.1): Sparse/diagonal gates
are handled by Atlas's `is_sparse()` check — when a gate is sparse,
the locality constraint is **entirely skipped** for ALL its qubits.
Detection is implemented in `non_insular_qubits()`:
- Z/S/T/CZ/CR: no qubits need to be local (sparse gates).
- All other gates (H, X, Y, CNOT, SWAP, RY, ...): all qubits must
  be local.

**What we skip from Atlas** (GPU-specific, not applicable to our CPU
out-of-core model):
- DP kernel scheduling (GPU kernel batching / fusion kernels).
- NCCL/MPI communication primitives.
- cuQuantum / cuStateVec integration.

**Execution flow:**

1. The ILP or heuristic produces a list of local-qubit sets (one per stage).
2. `_local_sets_to_steps()` converts these into runner steps:
   - Emits SWAP ops to transition between consecutive local sets.
   - Collects all executable gates per stage (dependency-aware).
   - Produces fused local-only steps and non-local butterfly steps.
3. The runner stores the final `QubitMap` as `qubit_mapping.json`.
4. `collect_state(apply_permutation=True)` uses `permute_state()` to
   reorder the result back to standard logical qubit order.

**Trade-off:** staging adds SWAP steps (each is a non-local I/O pass),
so it only helps when the circuit has clear qubit locality phases.
For circuits where all qubits are uniformly active, staging may not
improve or could slightly increase step count.

---

## Kernels

All kernels operate on in-memory NumPy arrays. No kernel does I/O.

### cpu_scalar (default)

Loop-based: for each pair of indices within a chunk, apply the 2×2
(or 4×4) matrix update.  Simple, correct, easy to verify.

### cpu_batched

Gather/scatter + GEMM: collect amplitude pairs into a (2 × M) or
(4 × M) matrix, multiply by the gate matrix, scatter results back.
Leverages NumPy's BLAS-backed `matmul`.  Faster for large chunks
when the working set fits in CPU cache.

### cpu_nonlocal

Pure array operations for non-local gates.  Four cases:
- 1Q non-local: pair of chunks → element-wise 2×2.
- 2Q, qa local: pair of chunks → local qa pairs within each chunk.
- 2Q, qb local: pair of chunks → local qb pairs within each chunk.
- 2Q, both non-local: quad of 4 chunks → element-wise 4×4.

---

## Runners

### single_node (sequential)

Processes work items one at a time.  Minimal memory footprint:
holds at most one chunk group in RAM at a time.

Supports both `scalar` and `batched` kernels.

### pipeline (threaded)

Three threads: reader → compute worker → writer.
Bounded queues (configurable `buffer_depth`) overlap I/O with compute.
Non-local groups are processed sequentially outside the pipeline
(they require multiple chunks loaded simultaneously).

### spark_runner (distributed)

Spark parallelizes local-gate chunk processing across executors.
Non-local groups are processed on the driver (sequential).
Spark is used **only for task distribution** — no amplitude shuffles.

---

## Write-ahead log (WAL)

A single atomic JSON file (`wal.json`) in the work directory.

```json
{
  "circuit_hash": "a3f8c012...",
  "committed_buf": "a",
  "done_steps": 5
}
```

### Fields

| Field | Purpose |
|-------|---------|
| `circuit_hash` | SHA-256 prefix of the circuit — detects accidental reuse of a work dir with a different circuit |
| `committed_buf` | Which buffer ("a" or "b") holds the latest committed state |
| `done_steps` | How many steps have been fully committed |

### Atomicity

Every WAL write follows: write to `.tmp` → `fsync` → `os.replace`.
This is atomic on POSIX and works on BeeGFS / NFS / Lustre.

### Recovery flow

1. Read `wal.json`.
2. `done_steps` → resume from that step.
3. `committed_buf` → use it as src.
4. Wipe dst buffer entirely (may contain stale partial data from previous crash).
5. Redo the full step from src (source is never modified, so input is always valid).
6. Write manifest to dst, commit step in WAL.

This is **Strategy 1** (double-buffer + step-level WAL) from `recovery_strategies.md`.
Upgrading to intra-step checkpointing (Strategy 2) for 45+ qubit runs is straightforward —
add a `checkpoint` field to the WAL and skip already-written items on recovery.

---

## Fencing lock

Prevents two processes from running on the same work directory
simultaneously (important on shared cluster filesystems).

Implemented as an atomic lock file (`run.lock`) containing PID,
hostname, and timestamp.  On acquisition: check if the holder is
still alive (same host: `kill(pid, 0)`; different host: staleness
timeout of 24 hours).

---

## Correctness guarantees

### Endianness

**Little-endian:** qubit 0 is bit 0 (LSB) of the state vector index.
Locked by `test_endianness_lock.py`: X on qubit 0 from |000⟩ must
produce amplitude 1.0 at index 1.

### Gate matrices

Canonical matrices are defined in `kernel/gates.py`.
2-qubit gates use big-endian subspace order (standard textbook convention):
`|qa=0,qb=0⟩, |qa=0,qb=1⟩, |qa=1,qb=0⟩, |qa=1,qb=1⟩`.

### Oracles

1. **ref_dense** — in-memory state-vector simulation (practical up to n ≈ 20).
   Applies gates one-by-one to the state vector; used as ground truth in kernel tests.

2. **Qiskit Statevector** — independent simulator.
   Circuits (including MQT Bench) are converted to our format via
   `import_qiskit.py` and compared against Qiskit's output.

### Test coverage

137 tests covering:
- Circuit validation and schema enforcement
- Endianness lock
- Known quantum states (Bell, GHZ, QFT)
- Kernel correctness vs ref_dense (scalar, batched, non-local)
- Out-of-core end-to-end vs oracle
- Crash recovery (simulated crash via subprocess + re-run)
- WAL circuit hash mismatch detection
- Double-buffer alternation
- Fencing lock semantics
- Spark runner correctness
- Gate fusion correctness
- Atlas-style circuit staging + qubit permutation

---

## Scaling characteristics

| Qubits | Amplitudes | State size | Chunks (8 MB) | Disk (2×) |
|--------|-----------|-----------|---------------|-----------|
| 20 | 1M | 8 MB | 1 | 16 MB |
| 30 | 1B | 8 GB | 1,024 | 16 GB |
| 40 | 1T | 8 TB | 1,048,576 | 16 TB |
| 45 | 32T | 256 TB | 33,554,432 | 512 TB |
| 50 | 1P | 8 PB | 1,073,741,824 | 16 PB |

### Bottlenecks by scale

| Scale | Bottleneck | Mitigation |
|-------|-----------|------------|
| ≤ 30q | None (fits in RAM) | Use ref_dense or in-memory mode |
| 30–40q | Disk I/O bandwidth | Pipeline runner, SSD, level batching |
| 40–45q | Disk capacity + I/O | Cluster storage (BeeGFS), Spark distribution |
| 45–50q | Everything | In-memory MPI/GPU cluster, or heroic out-of-core on large parallel FS |

### Key tuning parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| `chunk_size` | 2^20 (8 MB) | Larger → more qubits local (fewer non-local exchanges), more memory per chunk |
| `buffer_depth` | 4 | Pipeline queue depth — overlap I/O with compute |
| `use_fusion` | False | Enable level batching + 1Q gate fusion |
| `use_staging` | False | Enable circuit staging with qubit remapping |
| `staging_method` | "heuristic" | `"ilp"` (optimal, needs PuLP), `"heuristic"` (Atlas greedy), `"greedy"` (legacy) |
| `kernel` | "scalar" | "batched" uses GEMM — faster for large chunks that fit in cache |
