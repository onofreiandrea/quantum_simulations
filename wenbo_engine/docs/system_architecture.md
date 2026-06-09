# wenbo_engine — System Architecture

An out-of-core distributed quantum state-vector simulator.

## Motivation

Exact state-vector simulation of quantum circuits is exponential: each additional qubit doubles the memory required. At 40 qubits the state vector is 8.8 TB (complex64) or 17.6 TB (complex128). Mainstream simulators — QuEST (Jones et al., 2019), Qiskit Aer, Intel-QS, Cirq/qsim — all require the full state to fit in aggregate RAM across nodes. For example, the AWS QuEST benchmark (Baruffa et al., 2022) uses 256x c5.18xlarge instances (36.8 TB total RAM) to simulate 40 qubits in-memory.

wenbo_engine takes a different approach: **the state vector lives on disk instead of RAM.** Chunks are stored on NVMe and streamed through a small RAM window via a pipelined reader-compute-writer architecture. This trades wall-clock time for hardware requirements — our 40-qubit/50-gate benchmark takes ~7 hours where an in-memory simulator finishes in minutes, but runs on just 8 commodity instances (32 GB RAM each) instead of 256 high-memory nodes.

A few recent research projects explore similar ideas — QDAO (Zhao et al., 2023) spills state to SSD on a single node, BMQSim (2024) combines lossy compression with SSD storage for up to 47 qubits, and QVecOpt (2025) uses hierarchical storage with block optimization. wenbo_engine differs from these by combining out-of-core storage with MPI distribution across multiple nodes, qubit reordering to minimize I/O, and WAL-based crash recovery for long-running simulations.

**Proven at scale:** 40 qubits / 50 gates / 8.8 TB state vector on 8x i3en.xlarge in 6h 52min for ~$18.

---

## Architecture Overview

```
Circuit (gates)
    │
    ▼
┌──────────────────────────────────────────────────┐
│  circuit/                                        │
│  parse → levelize → reorder qubits → fuse        │
│  → compile into steps (3-way gate classification) │
└──────────────────┬───────────────────────────────┘
                   │  steps: [local, rank-NL, MPI-NL]
                   ▼
┌──────────────────────────────────────────────────┐
│  runner/ or mpi/                                 │
│  For each step:                                  │
│    Reader thread → Worker thread → Writer thread  │
│    (NVMe read)     (apply gates)   (NVMe write)   │
│                                                  │
│  Nonlocal: load chunk groups, butterfly exchange  │
│  MPI-nonlocal: MPI.Sendrecv with partner rank     │
└──────────┬────────────────┬──────────────────────┘
           │                │
           ▼                ▼
┌─────────────────┐  ┌─────────────────────────────┐
│  kernel/        │  │  storage/                    │
│  Numba JIT:     │  │  Atomic chunk I/O            │
│  apply_1q       │  │  (tmp → fsync → rename)      │
│  apply_2q       │  │  Double buffer:              │
│  butterfly pair │  │    state_a/ ↔ state_b/       │
│  butterfly quad │  │  WAL for crash recovery       │
│  (zero allocs)  │  │                              │
└─────────────────┘  └─────────────────────────────┘
```

Kernels are pure math (no I/O). Runners handle data movement. Storage handles durability. This separation is deliberate — each layer can be tested and benchmarked independently.

### Package Structure

| Package | Purpose |
|---------|---------|
| `circuit/` | Parse, validate, levelize, qubit reorder, 1Q fusion, Atlas staging |
| `kernel/` | Gate matrices, Numba JIT local/nonlocal kernels, numpy fallback, in-memory reference simulator (n ≤ 20) |
| `storage/` | Chunk I/O (atomic writes), manifest, zero-state init |
| `runner/` | Single-node sequential, threaded pipeline, Spark runners (3 variants) |
| `mpi/` | MPI orchestrator, benchmark driver, smoke tests |
| `wal/` | Write-ahead log, fencing lock, recovery |
| `bench/` | Kernel, I/O, and end-to-end microbenchmarks |
| `tests/` | 153+ pytest tests, shared circuit fixtures |

---

## Core Concepts

### State Vector as Chunks on Disk

A quantum state of n qubits has 2^n complex amplitudes. At 40 qubits that's 8.8 TB — far beyond any single machine's RAM. We split the state into fixed-size **chunks** stored as raw `complex64` binary files on disk.

```
chunk_size = 2^k amplitudes (default k=26 -> 512 MB per chunk)
n_chunks  = 2^n / chunk_size = 2^(n-k)

Chunk i holds amplitudes [i * chunk_size ... (i+1) * chunk_size)
```

For 40 qubits with k=26: 16,384 chunks of 512 MB each = 8,192 GB total.

### Double Buffering

A simulation that takes hours on a cluster will eventually experience a crash — a process killed by OOM, a network hiccup, or an EC2 spot termination. If the crash corrupts the state vector, the entire run must restart from scratch. Double buffering prevents this by ensuring the source state is never modified:

Two complete copies of the state exist on disk at all times:

```
work_dir/
├── state_a/
│   ├── manifest.json
│   └── chunks/
│       ├── chunk_000000.bin
│       ├── chunk_000001.bin
│       └── ...
├── state_b/
│   └── (same structure)
└── wal.json
```

Each simulation step reads from the **source** buffer and writes to the **destination** buffer. The source is never modified. On crash, the source is always a valid, complete state — just re-run the step.

### Sparse File Initialization

The initial state |0...0> has amplitude 1.0 at index 0 and zeros everywhere else. Instead of writing terabytes of zeros, we use OS sparse files:

```python
f.truncate(chunk_bytes)  # OS returns zeros on read, no disk write
```

Only chunk 0 of rank 0 gets a real write (512 MB). Initialization is near-instant regardless of state size.

---

## Gate Execution Model

### Levelization

Gates are grouped into **levels** — sets of gates that share no qubits and can be applied in any order:

```
Circuit:  H(0), CZ(0,1), H(2), RY(3)
Level 0:  [H(0), H(2), RY(3)]     <- independent
Level 1:  [CZ(0,1)]               <- depends on H(0)
```

### Local vs Nonlocal Classification

Given k = log2(chunk_size), each gate is classified:

- **Local** (all qubits < k): Both amplitudes live in the same chunk. Process one chunk at a time, no inter-chunk I/O.

- **Nonlocal** (some qubit >= k): Paired amplitudes live in different chunks. Requires loading a **group** of partner chunks simultaneously (butterfly exchange).

Partner relationship: chunk `c` pairs with `c XOR (1 << (q - k))` for qubit q.

### Three-Way Classification (MPI)

The MPI runner further splits nonlocal gates:

```
Qubit ranges for 40q, k=26, 8 ranks (rank_bits=3):
  Local:          qubits 0..25     (within each chunk)
  Rank-nonlocal:  qubits 26..36    (partner chunk on same rank's NVMe)
  MPI-nonlocal:   qubits 37..39    (partner chunk on different rank)
```

- **Rank-nonlocal**: Partner chunks are on the same node. Handled by reading both chunks from local disk, applying the butterfly exchange, and writing back. No network I/O.

- **MPI-nonlocal**: Partner chunks are on a different node. Requires MPI Sendrecv to exchange chunk data with the partner rank.

### Qubit Reordering

In an out-of-core simulator, nonlocal gates are expensive — each one requires loading entire groups of partner chunks from disk (rank-nonlocal) or exchanging data across the network (MPI-nonlocal). MPI-nonlocal gates are the worst case: a 512 MB chunk must be sent over a 25 Gbps link (~170 ms per chunk), compared to ~1.7s for a local NVMe read. Eliminating MPI-nonlocal gates entirely means zero network traffic during compute, which is critical when a run takes hours.

`circuit/reorder.py` permutes qubits so the most frequently targeted ones occupy the lowest (local) bit positions. The circuit is mathematically identical — only the physical data layout changes. Qubits that appear in many gates are placed in the chunk-internal range (qubits 0..k-1), so those gates become local. Rarely-targeted qubits are pushed to the highest positions (the MPI-nonlocal range), where they do no harm because no gate touches them.

For the 40q/50g benchmark, reordering eliminated **all MPI-nonlocal gates** (0 out of 50), leaving only 42 local + 8 rank-nonlocal gates. Zero cross-node communication during compute.

### Fusion Optimizations

1. **Level batching**: Consecutive all-local levels merge into one step (one I/O pass instead of many).

2. **1Q gate fusion**: Consecutive 1Q gates on the same qubit pre-multiply into a single 2x2 matrix.

Both reduce the number of full I/O passes over the state vector.

---

## Kernels

All kernels operate on in-memory numpy arrays. No kernel does I/O. This separation is deliberate: runners handle data movement, kernels handle pure math.

### Kernel Evolution: Three Iterations from Failed Runs

The kernel implementation went through three versions, each driven by a failed AWS run that revealed a performance bottleneck.

**Version 1: Fancy indexing (622 ms/gate).** The original implementation used `np.arange(N)` + fancy indexing (`chunk[idx0]`, `chunk[idx1]`) to gather paired amplitudes. On a 128 MB chunk (2^24 complex64), one H gate took 622 ms. At that rate, the 38q benchmark would have taken ~36 hours — we discovered this only after deploying to EC2, when step 0 was progressing far slower than expected.

**Why it was slow:** Fancy indexing on large arrays is cache-hostile. For qubit q, the index pattern has stride 2^q. At q=20, paired elements are 8 MB apart — a guaranteed L3 cache miss on every access. The `np.arange(N)` also allocates a full 128 MB index array just to select half the elements.

**Version 2: Reshape + numpy (182 ms/gate).** Replaced fancy indexing with `chunk.reshape(-1, block)` + contiguous slicing. Reshape creates views (no copy), so this was a ~3.4x improvement. But a dry run on EC2 showed the pipeline was still compute-bound rather than I/O-bound — the kernel couldn't keep up with NVMe read speed.

**Why it was still slow:** NumPy can't fuse element-wise operations. `U[0,0] * lo + U[0,1] * hi` creates 3 temporary arrays (one per `*`, one for `+`), each 256 MB for a 512 MB chunk. A single 1-qubit gate creates ~1.5 GB of temporary arrays, causing:
1. **Memory pressure**: Temporaries compete with pipeline buffers for RAM.
2. **Cache thrashing**: Each temporary is written then read once, evicting useful data.
3. **Bandwidth saturation**: 6+ full array passes for what should be 1 pass of trivial arithmetic.

**Version 3: Numba JIT (7.8 ms/gate).** Replaced all gate kernels with `@numba.njit(parallel=True)` loops. Each amplitude pair is loaded into CPU registers, the 2x2 (or 4x4) unitary is applied in registers, and the result is written back. Zero temporary arrays, single pass, multi-core via `numba.prange`. This made the pipeline I/O-bound as intended — compute finishes in ~8 ms while NVMe read takes ~1.7s per chunk.

### Numba JIT Kernels (primary, cpu_batched.py + cpu_nonlocal.py)

All hot-path kernels use `@numba.njit(parallel=True, cache=True)` with `numba.prange` for multi-core parallelism. The key insight: each amplitude pair (for 1Q) or quad (for 2Q) is independent — perfect for data-parallel execution with zero synchronization.

#### 1-Qubit Local Kernel

```python
@numba.njit(cache=True, parallel=True)
def _apply_1q_numba(chunk, qubit, u00, u01, u10, u11):
    step = 1 << qubit          # distance between paired amplitudes
    block = step << 1          # size of one (lo, hi) block
    N = len(chunk)
    n_blocks = N >> (qubit + 1)
    for blk in numba.prange(n_blocks):   # parallel across blocks
        base = blk * block
        for off in range(step):          # sequential within block
            i = base + off
            j = i + step
            a = chunk[i]; b = chunk[j]   # load pair into registers
            chunk[i] = u00 * a + u01 * b # write back immediately
            chunk[j] = u10 * a + u11 * b # no temporary arrays
```

Why this is fast:
- **Zero allocations**: `a` and `b` live in CPU registers, not RAM. The result is written back to the same memory locations. Total extra memory: 0 bytes.
- **Single pass**: Each element is read once and written once. For a 512 MB chunk, that's 1 GB of memory traffic total (read + write), versus ~3 GB for numpy.
- **Parallel**: `numba.prange` splits the outer loop across all CPU cores. On a 4-core i3en.xlarge, 4 threads process non-overlapping blocks simultaneously.
- **Cache-friendly**: Sequential access within each block means the hardware prefetcher keeps the pipeline full. Paired amplitudes `(i, j)` are at most `step` elements apart — for low qubits, they're in the same cache line.
- **Compiled to native code**: Numba compiles to LLVM IR → machine code on first call. Subsequent calls use the cached `.nbi` file with zero overhead. The gate matrix elements (`u00`, `u01`, `u10`, `u11`) are passed as scalar arguments, avoiding any matrix indexing overhead.

#### 2-Qubit Local Kernel

The 2-qubit kernel operates on quads of amplitudes `|qa=0,qb=0>`, `|qa=0,qb=1>`, `|qa=1,qb=0>`, `|qa=1,qb=1>`:

```python
@numba.njit(cache=True, parallel=True)
def _apply_2q_numba(chunk, qa, qb, U):
    # qa < qb guaranteed by caller (swaps + permutes U if needed)
    N = len(chunk)
    A = N >> (qb + 1)
    C = 1 << qa
    B = 1 << (qb - qa - 1)
    total = A * B * C
    for idx in numba.prange(total):
        # Decode 3D index into flat array positions
        c = idx % C;  ab = idx // C;  b = ab % B;  a = ab // B
        i00 = (a << (qb+1)) | (b << (qa+1)) | c
        i01 = i00 | (1 << qb)
        i10 = i00 | (1 << qa)
        i11 = i00 | (1 << qa) | (1 << qb)
        # Load 4 amplitudes into registers
        s0, s1, s2, s3 = chunk[i00], chunk[i01], chunk[i10], chunk[i11]
        # Apply 4x4 unitary in-place
        chunk[i00] = U[0,0]*s0 + U[0,1]*s1 + U[0,2]*s2 + U[0,3]*s3
        chunk[i01] = U[1,0]*s0 + U[1,1]*s1 + U[1,2]*s2 + U[1,3]*s3
        chunk[i10] = U[2,0]*s0 + U[2,1]*s1 + U[2,2]*s2 + U[2,3]*s3
        chunk[i11] = U[3,0]*s0 + U[3,1]*s1 + U[3,2]*s2 + U[3,3]*s3
```

The bit-manipulation index decoding looks complex but compiles to fast integer shifts. Each iteration: 4 loads, 16 multiplies, 12 adds, 4 stores — all in registers.

#### Performance Results

| Kernel | numpy (ms) | numba (ms) | Speedup | Temp memory |
|--------|-----------|-----------|---------|-------------|
| apply_1q (512 MB chunk) | 112 | 7.8 | 14x | 0 vs ~1.5 GB |
| apply_2q (512 MB chunk) | 288 | 17.6 | 16x | 0 vs ~2 GB |
| apply_1q_pair (nonlocal) | 80 | 9.6 | 8x | 0 vs ~1 GB |
| apply_2q_quad (nonlocal) | 320 | 22.4 | 14x | 0 vs ~2 GB |

The speedup is even more critical in the out-of-core context: compute time per chunk directly determines whether the pipeline is I/O-bound (good) or compute-bound (wasted I/O bandwidth). At 300 MB/s NVMe read speed, reading a 512 MB chunk takes ~1.7s. A numba kernel finishes in ~8-18ms per gate — negligible. With numpy, a single gate takes ~112-288ms, and 9 local gates per step would take ~1-2.6s — dangerously close to the I/O time, eliminating the pipeline overlap benefit.

#### Why Not GPU?

GPUs excel at SIMD computation but our bottleneck is disk I/O, not compute. At 40 qubits, each step reads and writes 1 TB per rank from NVMe. Even with numba's 8ms per gate, the total compute for 9 local gates is ~72ms per chunk — versus ~3.4s of I/O per chunk (read + write at 300 MB/s). The pipeline hides the compute behind I/O. A GPU would finish the compute in microseconds but still wait 3.4s for the disk. The speedup would be <1%.

### Nonlocal Kernels (cpu_nonlocal.py)

When a gate targets qubit q >= k (the chunk bits), the paired amplitudes live in different chunks. The caller loads the partner chunks into memory; the kernel applies the gate across them.

Four butterfly-exchange variants handle all cases:

**1Q nonlocal** — Two partner chunks `(c0, c1)` where `c1 = c0 XOR (1 << (q-k))`:
```python
@numba.njit(cache=True, parallel=True)
def _apply_1q_pair_numba(c0, c1, u00, u01, u10, u11):
    for i in numba.prange(len(c0)):
        a = c0[i]; b = c1[i]       # same position in both chunks
        c0[i] = u00 * a + u01 * b  # element-wise 2x2 transform
        c1[i] = u10 * a + u11 * b
```

This is a pure element-wise operation — `c0[i]` pairs with `c1[i]` because the nonlocal bit distinguishes the two chunks while all local bits are shared.

**2Q, one local + one nonlocal** — Two chunks, but the local qubit creates pairs *within* each chunk. Two sub-variants depending on which qubit is local (`apply_2q_pair_qa_local`, `apply_2q_pair_qb_local`). Each reads 4 values (2 from each chunk), applies the 4x4 unitary, writes back.

**2Q, both nonlocal** — Four chunks `(c00, c01, c10, c11)` representing the four combinations of two nonlocal bits. Element-wise 4x4 across the quad:

```python
@numba.njit(cache=True, parallel=True)
def _apply_2q_quad_numba(c00, c01, c10, c11, U):
    for i in numba.prange(len(c00)):
        s0, s1, s2, s3 = c00[i], c01[i], c10[i], c11[i]
        c00[i] = U[0,0]*s0 + U[0,1]*s1 + U[0,2]*s2 + U[0,3]*s3
        c01[i] = U[1,0]*s0 + U[1,1]*s1 + U[1,2]*s2 + U[1,3]*s3
        c10[i] = U[2,0]*s0 + U[2,1]*s1 + U[2,2]*s2 + U[2,3]*s3
        c11[i] = U[3,0]*s0 + U[3,1]*s1 + U[3,2]*s2 + U[3,3]*s3
```

### Reference Simulator (ref_dense.py)

In-memory dense simulator for correctness testing (n <= ~20). Applies gates directly to the full state vector using numpy. Used as ground truth in all kernel and end-to-end tests — every test compares the out-of-core result against `ref_dense` to verify bit-exact (within floating-point tolerance) correctness.

---

## Runners

### Single-Node Runner (runner/single_node.py)

Sequential out-of-core simulation with double buffering. Processes one chunk (or one chunk group for nonlocal gates) at a time. Minimal memory footprint.

### Pipeline Runner (runner/pipeline.py)

Three threads with bounded queues:

```
Reader thread  --[Queue(depth)]--> Worker thread --[Queue(depth)]--> Writer thread
  (read chunk)                     (apply gates)                    (write chunk)
```

Overlaps disk I/O with compute. While the worker applies gates to chunk N, the reader pre-fetches chunk N+1 and the writer flushes chunk N-1.

Nonlocal groups are processed outside the pipeline (they need multiple chunks simultaneously).

### Spark Runner (runner/spark_runner.py)

Spark distributes local-gate chunk processing across executors via `rdd.map()`. Nonlocal gates are processed on the driver node sequentially. All nodes access chunks via NFS.

**Limitation**: The driver must hold an entire nonlocal group in RAM. For large groups this causes OOM.

### RDD Runner (runner/rdd_runner.py)

State vector persisted as an RDD of (chunk_idx, bytes) pairs. Uses `MEMORY_AND_DISK` or `DISK_ONLY` persistence. Nonlocal gates via `groupByKey` on partner chunk indices.

### Distributed Runner (runner/distributed_runner.py)

All nodes access chunks via a shared filesystem (NFS, BeeGFS). Spark distributes both local and nonlocal chunk processing. Each chunk has an owner partition for write scheduling.

---

## MPI Runner (mpi/mpi_runner.py) — Primary

The MPI runner is the primary execution engine for large-scale simulations. Each MPI rank owns a partition of chunks on local NVMe. No shared filesystem needed — all inter-node communication is explicit via MPI.

### Architecture

```
Rank 0 (node 0)              Rank 1 (node 1)           ... Rank 7 (node 7)
┌─────────────────────┐      ┌─────────────────────┐      ┌─────────────────────┐
│ /mnt/nvme/           │      │ /mnt/nvme/           │      │ /mnt/nvme/           │
│   rank_0/            │      │   rank_1/            │      │   rank_7/            │
│     state_a/chunks/  │      │     state_a/chunks/  │      │     state_a/chunks/  │
│     state_b/chunks/  │      │     state_b/chunks/  │      │     state_b/chunks/  │
│     wal.json         │      │     wal.json         │      │     wal.json         │
│                      │      │                      │      │                      │
│  [2048 chunks each]  │      │  [2048 chunks each]  │      │  [2048 chunks each]  │
└──────────┬───────────┘      └──────────┬───────────┘      └──────────┬───────────┘
           │                  MPI Sendrecv                             │
           └──────────────────────────────────────────────────────────┘
                         (only for MPI-nonlocal gates)
```

Each rank maps to one node. The state vector is partitioned by the highest `p = log2(n_ranks)` bits of the chunk index: rank `r` owns all chunks where the top `p` bits equal `r`. For 40 qubits with k=26 and 8 ranks: each rank owns 2,048 chunks (1 TB of state + 1 TB double buffer = 2 TB NVMe used).

### Chunk Ownership and Partner Mapping

The chunk index `c` encodes the amplitude range that chunk holds. For a gate on qubit `q >= k`, the partner chunk is at index `c XOR (1 << (q - k))`. This XOR flips one bit in the chunk index:

```
Example: 40q, k=26, 8 ranks (p=3)

Chunk index bits:  [rank bits (3)] [local bits (11)]
                    ─────────────   ───────────────
                    qubits 37-39    qubits 26-36

Gate on qubit 30 (rank-nonlocal):
  partner_bit = 30 - 26 = 4  (within local bits)
  → Partner chunk is on the SAME rank ✓

Gate on qubit 38 (MPI-nonlocal):
  partner_bit = 38 - 26 = 12  (within rank bits)
  → Partner chunk is on a DIFFERENT rank → needs MPI
```

### Circuit Compilation

Before simulation, the circuit is compiled into a sequence of **steps**. Each step corresponds to one level of the levelized circuit (gates that share no qubits and can be applied in any order).

`_compile_steps()` performs three-way classification on each level:

```python
def _compile_steps(levels, k, n_local_bits):
    for level in levels:
        local_ops = []       # all qubits < k
        rank_nl_ops = []     # some qubit >= k, partner bit < n_local_bits
        mpi_nl_ops = []      # some qubit has partner bit >= n_local_bits
        for gate in level:
            if all(q < k for q in gate.qubits):
                local_ops.append(gate)
            elif any((q - k) >= n_local_bits for q in gate.qubits if q >= k):
                mpi_nl_ops.append(gate)
            else:
                rank_nl_ops.append(gate)
```

For the 40q/50g benchmark: `n_local_bits = 40 - 26 - 3 = 11`. After reordering, all 50 gates map to qubits 0-36, so no gate has `partner_bit >= 11`, yielding 0 MPI-nonlocal gates.

### Step Execution

Each step processes its gates in three sequential phases. Within each phase, I/O and compute are overlapped via pipelining.

#### Phase 1: Rank-Nonlocal Gates (+ Local Gates)

This is the most complex phase. Rank-nonlocal gates require loading **groups** of partner chunks simultaneously — all chunks that differ only in the nonlocal bit positions must be in memory together for the butterfly exchange.

**Group construction**: For `b` distinct nonlocal bits, each group contains `2^b` chunks. The chunks in a group share the same values for all non-nonlocal bits and differ in the nonlocal bits. For example, with nonlocal bits {2, 5}, each group has 4 chunks: `base`, `base|4`, `base|32`, `base|36`.

**Pipeline**: The group reader, group worker, and group writer run in separate threads:

```
Group Reader          Group Worker              Group Writer
   │                     │                         │
   ├─ read group N+1     ├─ apply local ops        ├─ write group N-1
   │   (all 2^b chunks)  │   to each chunk         │   (all 2^b chunks)
   │                     ├─ apply nonlocal ops      │
   │                     │   across chunk pairs     │
   │                     │   (butterfly exchange)   │
   ▼                     ▼                         ▼
        [Queue(depth)]        [Queue(depth)]
```

The group worker first applies all local gates to each chunk in the group independently, then applies the nonlocal butterfly exchanges across chunk pairs. This means chunks processed in groups also get their local gates applied — avoiding a redundant I/O pass.

After all groups are processed, the pipeline processes remaining **local-only chunks** (those not part of any nonlocal group) with the simpler chunk-level pipeline:

```
Chunk Reader → Chunk Worker → Chunk Writer
  read 512 MB    apply local    write 512 MB
                 gates
```

The `skip_set` parameter ensures chunks already processed in groups are not re-processed.

#### Phase 2: MPI-Nonlocal Gates

MPI-nonlocal gates require exchanging chunk data with a partner rank on a different node. Three sub-cases:

**1Q MPI-nonlocal**: The partner rank is `rank XOR (1 << rank_bit)`. Uses a 3-stage pipeline:

```
Reader thread: pre-fetches next chunk from NVMe
Main thread:   MPI.Sendrecv (exchange chunk with partner) + numba compute
Writer thread: writes result to NVMe
```

`MPI.Sendrecv` is used instead of separate `Send`/`Recv` to avoid deadlocks — both ranks exchange simultaneously. The i3en.xlarge has 25 Gbps network (~3 GB/s), so exchanging a 512 MB chunk takes ~170ms. While the main thread does Sendrecv for chunk N, the reader pre-fetches chunk N+1 from NVMe and the writer flushes chunk N-1.

**2Q, one MPI-nonlocal qubit**: If the other qubit is local, each chunk exchanges with its partner, then the 2Q kernel runs on the pair. If the other qubit is rank-nonlocal, we need to load *two* local chunks (the pair for the rank-nonlocal bit), exchange *both* with the partner rank, then apply the 4x4 quad kernel.

**2Q, both MPI-nonlocal**: Four partner ranks are involved. Each rank exchanges its chunk with the other three via multiple `MPI.Sendrecv` calls, then applies the 4x4 quad kernel. This is the most network-intensive case but is rare — the qubit reordering specifically targets these gates.

#### Phase Ordering

The three phases execute sequentially within each step:
1. Rank-nonlocal + local gates: `src_dir → dst_dir`
2. Local-only chunks: `src_dir → dst_dir` (skipping chunks already done in phase 1)
3. MPI-nonlocal gates: `dst_dir → dst_dir` (in-place update)

After all three phases complete, `comm.Barrier()` synchronizes all ranks before committing the WAL.

### Batch Splitting

When rank-nonlocal gates in a step touch many distinct nonlocal bits, the group size can exceed RAM. For example, 6 nonlocal bits = 64 chunks x 512 MB = 32 GB per group — the entire RAM of an i3en.xlarge.

This was discovered during our first AWS run: step 0 of the 38q benchmark had 8 nonlocal bits, producing groups of 32 chunks × 128 MB = 4 GB each. The pipeline held multiple groups simultaneously, exceeding the 32 GB available and causing OOM. Batch splitting was implemented to partition gates into memory-safe subsets.

`_split_nonlocal_batches()` solves this by partitioning gates into memory-safe batches:

1. **Extract nonlocal bits** per gate: each gate contributes `{q - k for q in qubits if q >= k}`
2. **Union-find**: Gates sharing any nonlocal bit are connected. Connected components are groups of gates that *must* be processed together (their nonlocal bits overlap).
3. **Greedy packing**: Sort components by size (smallest first). Pack into batches where `2^(total_bits_in_batch) * chunk_bytes <= max_group_mem`.

```
Example from step 1 of 40q benchmark:
  Gate A targets qubit 28 → nonlocal bit 2
  Gate B targets qubits 26,28 → nonlocal bits {0, 2}
  Gate C targets qubit 31 → nonlocal bit 5
  Gate D targets qubit 33 → nonlocal bit 7

  Union-find: A,B connected (share bit 2). C independent. D independent.
  Components: {A,B} uses bits {0,2}, {C} uses bit {5}, {D} uses bit {7}

  max_group_mem = 4.8 GB (24 GB usable / 5)
  max_bits = log2(4.8 GB / 512 MB) = 3

  Batch 1: {C}+{D} → bits {5,7} → 4 chunks × 512 MB = 2 GB ✓
  Batch 2: {A,B} → bits {0,2} → 4 chunks × 512 MB = 2 GB ✓
```

The first batch processes `src → dst` and includes all local gates. Subsequent batches process `dst → dst` (in-place), with no local gates (already applied).

### Auto-Detected Memory Limits

The runner auto-detects available RAM and adapts:

```python
total_ram = psutil.virtual_memory().total  # or os.sysconf fallback
ram_avail = max(total_ram - 8 GB, 16 GB)   # reserve 8 GB for OS + Python
max_group = ram_avail // 5                  # max group mem for batch splitting
```

On i3en.xlarge (32 GB RAM): `ram_avail = 24 GB`, `max_group = 4.8 GB`, `max_bits = 3` (8 chunks of 512 MB).

### Safe Pipeline Depth

The same first AWS run also revealed that the pipeline buffer depth couldn't be hardcoded. The original `buffer_depth=4` meant the 3-stage pipeline (reader/worker/writer) could hold up to 11 groups simultaneously — with 4 GB groups that's 44 GB peak, far exceeding the 32 GB available. The depth must scale with group size.

The pipeline has `2 * depth + 3` groups simultaneously in memory at worst case:

```
reader has read 1 group + depth groups in reader queue
+ 1 group being processed by worker
+ depth groups in writer queue + 1 being written

Total: 2 * depth + 3 groups
```

`_safe_depth()` solves for the maximum safe depth:

```python
depth = (max_avail / group_mem - 3) / 2
```

For a group with 2 nonlocal bits (4 × 512 MB = 2 GB): `depth = (24 GB / 2 GB - 3) / 2 = 4`. For 4 nonlocal bits (16 × 512 MB = 8 GB): `depth = (24 GB / 8 GB - 3) / 2 = 0`, which gets clamped to 1.

### Initialization

The initial state `|0...0>` is set up using sparse files:

```python
for local_ci in range(n_chunks_per_rank):
    if rank == 0 and local_ci == 0:
        data = np.zeros(chunk_size, dtype=complex64)
        data[0] = 1.0 + 0j
        write_chunk_atomic(path, data)  # 512 MB real write
    else:
        f.truncate(sparse_bytes)  # OS returns zeros on read, no disk write
```

Only rank 0, chunk 0 gets a real write (512 MB). All other chunks use sparse files — the OS maps them to zeros without writing any data. For the 40q benchmark, initialization writes 512 MB out of 16 TB total state — near-instant.

### WAL Recovery and Rank Synchronization

Each rank maintains its own WAL. On recovery, ranks may disagree on `done_steps` if one rank committed its WAL but another crashed before committing. The runner synchronizes via `comm.allgather()`:

```python
all_done = comm.allgather(done_steps)
min_done = min(all_done)
# All ranks restart from the minimum safe step
```

This ensures all ranks replay from the same consistent point. The destination buffer from the crashed step gets fully overwritten — the source buffer is always intact.

### Norm Computation

After all gates are applied, the norm `||ψ||` is computed to verify unitarity. Each rank reads all its chunks, casts to `complex128` for precision, and computes the local sum of `|a_i|^2`:

```python
local_norm_sq = sum(|a_i|^2 for all amplitudes on this rank)
global_norm_sq = MPI.Allreduce(local_norm_sq, MPI.SUM)
norm = sqrt(global_norm_sq)
```

The cast to `complex128` is necessary because summing ~67 million `complex64` values per chunk accumulates floating-point error. For 40 qubits, each rank sums over 2,048 chunks × 67M values = ~137 billion values. The resulting norm error of 2.16e-07 is well within the `1e-6` tolerance.

### Threading Configuration

```bash
NUMBA_NUM_THREADS=$(nproc)   # numba gets all cores
OMP_NUM_THREADS=1            # disable BLAS threading (dead code with numba)
MKL_NUM_THREADS=1
OPENBLAS_NUM_THREADS=1
```

Numba releases the GIL during `prange` execution, so the reader/writer threads run concurrently with compute. This is critical for the pipeline overlap: while numba processes a chunk across 4 cores, the reader thread (Python, GIL-free during I/O) loads the next chunk from NVMe, and the writer thread flushes the previous chunk. The 1-thread BLAS settings prevent numpy (used in the fallback path) from competing with numba for CPU cores.

### End-to-End Flow

```
mpirun -np 8 python -m wenbo_engine.mpi.mpi_benchmark --qubits 40 --gates 50

1. Each rank initializes:
   - Validate circuit, levelize, compile steps
   - Check WAL for recovery
   - Init |0...0> with sparse files (512 MB real write, rest sparse)
   - Barrier: all ranks ready

2. For each step (6 steps for 40q/50g):
   a. Phase 1: Rank-nonlocal groups
      - Split into batches if group exceeds RAM
      - Pipeline: group_reader → group_worker → group_writer
      - Worker applies local + nonlocal gates to each group
   b. Phase 2: Local-only chunks
      - Pipeline: chunk_reader → chunk_worker → chunk_writer
      - Skip chunks already processed in phase 1
   c. Phase 3: MPI-nonlocal gates (0 in 40q benchmark)
      - Pipeline: reader → sendrecv+compute → writer
   d. Barrier + WAL commit + swap buffers

3. Norm computation:
   - Each rank: read all chunks, cast to complex128, sum |a_i|^2
   - MPI.Allreduce to get global norm
   - Verify |norm - 1.0| < 1e-6

4. Write JSON results (rank 0)
```

---

## Write-Ahead Log (WAL)

### Format

```json
{
  "circuit_hash": "5f67e5e7363aa4b1",
  "committed_buf": "a",
  "done_steps": 3
}
```

| Field | Purpose |
|-------|---------|
| circuit_hash | SHA-256 prefix — detects accidental reuse with a different circuit |
| committed_buf | Which buffer (a or b) holds the latest committed state |
| done_steps | Number of fully committed steps |

### Atomicity

Every WAL write: write to `.tmp` -> `fsync` -> `os.replace`. Atomic on POSIX, works on NFS/BeeGFS/Lustre.

### Recovery Flow

1. Read `wal.json` -> get `done_steps` and `committed_buf`
2. Resume from `done_steps` using `committed_buf` as source
3. The destination buffer may contain partial data from the crash — it gets fully overwritten
4. Source is never modified, so it's always a valid complete state

### Limitation

WAL + state buffers are on local NVMe. If the NVMe is lost (e.g., EC2 spot termination), all data is gone. WAL only recovers from process crashes where disk survives. For spot instances, use on-demand.

---

## Fencing Lock

Prevents concurrent runs on the same work directory. Atomic lock file (`run.lock`) containing PID, hostname, timestamp. Checks if holder is alive via `kill(pid, 0)` on same host; assumes alive on different host with 24-hour staleness timeout.

---

## Correctness

### Numerical Precision: complex64 vs complex128

The simulator deliberately uses two different precisions at different stages:

**complex64 (8 bytes per amplitude)** is used for all state storage on disk and all gate computation. In an in-memory simulator like QuEST, switching from complex128 to complex64 saves RAM but doesn't fundamentally change feasibility. In our out-of-core simulator, it's the single highest-impact decision: every step reads and writes the entire state from NVMe, so halving the amplitude size halves the I/O time, halves the disk capacity needed, and halves the number of nodes:

| Metric | complex64 (our choice) | complex128 |
|---|---|---|
| 40q state size | **8 TB** | 16 TB |
| Double buffer total | 16 TB | 32 TB |
| Nodes needed (2 TB NVMe) | **8** | 16 |
| I/O per step | **16 TB** (read + write) | 32 TB |
| Benchmark time | **6h 52min** | ~13h 44min |
| Cost | **~$18** | ~$36 |

Using complex64 halves the state size, which halves the I/O time, disk requirements, and number of nodes needed. For an I/O-bound simulator, this is the single highest-impact optimization.

**Why float32 precision is sufficient for gate application:** Every quantum gate is a unitary matrix with condition number 1 — it is norm-preserving by definition, so numerical errors do not amplify multiplicatively from one gate to the next. Float32 provides ~7 decimal digits of relative precision per arithmetic operation. After applying 50 gates, the accumulated error in the state vector norm is ~2e-07 (about 6.5 correct digits), well within our verification tolerance of 1e-6.

For comparison, most in-memory simulators (Qiskit Statevector, QuEST) default to complex128 — but they can afford to because state size is not their bottleneck. For us, that extra precision would double the runtime for negligible accuracy improvement.

**Where complex64 would break:** For very deep circuits (thousands of gates), float32 rounding errors would accumulate past the 1e-6 tolerance. For 50-gate circuits this is not a concern.

**complex128 (16 bytes per amplitude)** is used in exactly two places:

1. **Gate matrix definitions** (`kernel/gates.py`): Gate unitaries are defined in complex128 for mathematical precision, then cast to complex64 when compiled into simulation steps (`gate_matrix().astype(DTYPE)`). This ensures the gate matrix itself is computed accurately before the downcast.

2. **Norm computation** (`mpi_runner.compute_norm()`): After simulation, verifying ||ψ|| ≈ 1.0 requires summing |a_i|^2 across all 2^40 ≈ 10^12 amplitudes. Each |a_i|^2 term is approximately 2^-40 ≈ 10^-12, while the running sum grows toward 1.0. In float32, adding 10^-12 to a ~1.0 accumulator is below machine epsilon (~10^-7) — the small values simply vanish, and the sum converges to a wrong answer. In float64 (machine epsilon ~10^-16), the addition is still exact.

    ```python
    # Each chunk cast to complex128 one at a time — no extra memory pressure
    for chunk_file in sorted(chunks_dir.glob("chunk_*.bin")):
        data = read_chunk(chunk_file)                              # complex64
        local_norm_sq += np.sum(np.abs(data.astype(np.complex128)) ** 2)
    ```

3. **Reference simulator** (`kernel/ref_dense.py`): The in-memory reference simulator uses complex128 throughout because it serves as the ground-truth oracle for testing. Precision matters more than size at small qubit counts (n <= 20).

### Endianness

**Little-endian throughout**: qubit 0 = LSB of the state vector index. Locked by `test_endianness_lock.py`.

### Gate Matrices

Defined in `kernel/gates.py`. 2Q gates use big-endian subspace order: |qa=0,qb=0>, |qa=0,qb=1>, |qa=1,qb=0>, |qa=1,qb=1>.

### Verification

- **ref_dense.py**: In-memory oracle for small circuits (complex128 ground truth)
- **Qiskit Statevector**: Independent simulator comparison
- **Norm check**: |norm - 1.0| < 1e-6 after simulation (requires complex128 accumulation)
- **153+ pytest tests** covering kernels, end-to-end, crash recovery, fusion, staging

---

## Benchmark Circuit

The benchmark uses a **random non-stabilizer circuit** — a circuit that cannot be efficiently simulated classically using stabilizer (Clifford) shortcuts and requires full state-vector simulation.

The circuit is generated by the following algorithm (deterministic with `seed=42`):

```
Input: N qubits, G gates
For each gate in G:
    Flip an unbiased coin:
      Heads → apply CZ (controlled phase flip) to two randomly chosen qubits
      Tails → pick a random 1-qubit gate from {RX, RY, RZ, H}:
              if H: apply Hadamard
              if RX/RY/RZ: choose random angle θ ∈ [0, π), apply rotation
```

This produces a mix of entangling 2-qubit gates (CZ) and non-Clifford 1-qubit rotations (RX, RY, RZ with irrational angles). The irrational rotation angles are what make the circuit non-stabilizer — they generate states that cannot be represented compactly in the stabilizer formalism, forcing exponential-cost exact simulation.

This is the same circuit generation scheme used in the AWS HPC benchmark for simulating 44-qubit circuits (Baruffa et al., 2022).

---

## Benchmark Results

### 40-Qubit / 50-Gate MPI Benchmark

**Circuit**: Random non-stabilizer circuit (40 qubits, 50 gates, seed=42) with qubit reordering.

**Hardware**: 8x i3en.xlarge (4 vCPU, 32 GB RAM, 2.5 TB NVMe each), us-east-1.

**Configuration**:
- chunk_size = 2^26 (512 MB)
- 16,384 total chunks, 2,048 per rank
- State = 1 TB per rank, 8 TB total (+ 8 TB for double buffer)
- Sparse file init (near-instant)

**Gate classification after reordering**:
- 42 local + 8 rank-nonlocal + **0 MPI-nonlocal** = 50 gates
- Zero cross-node communication during compute

**Step-by-step results**:

| Step | Local | Rank-NL | MPI-NL | Time |
|------|-------|---------|--------|------|
| 1/6 | 9 | 4 | 0 | 98.4 min |
| 2/6 | 8 | 2 | 0 | 73.1 min |
| 3/6 | 8 | 1 | 0 | 64.7 min |
| 4/6 | 7 | 1 | 0 | 58.2 min |
| 5/6 | 6 | 0 | 0 | 42.6 min |
| 6/6 | 4 | 0 | 0 | 31.8 min |
| **Total** | **42** | **8** | **0** | **368.8 min** |

**Results**:
- Total wall time: **6h 52min** (24,735s including norm computation)
- Norm: 0.9999997843 (error: 2.16e-07)
- Pass: true
- Cost: **~$18** (8x $0.326/hr x 6.87 hrs)

### Prior Result: 38-Qubit / 50-Gate

Same setup at smaller scale (4x i3en.xlarge, k=24, 2 TB state). Completed in **3h 38min** for **~$4.75**. Gate split: 40 local + 10 rank-nonlocal + 0 MPI-nonlocal. Norm error: 1.78e-07. This run validated the architecture before scaling to 40 qubits.

### Comparison with AWS QuEST (Baruffa et al., 2022)

The AWS HPC blog benchmark simulates 40-44 qubit random circuits using QuEST on c5.18xlarge instances. The comparison below is **not apples-to-apples** — the workloads differ in gate count, precision, and circuit depth. It illustrates the architectural trade-off, not a direct performance contest.

**Key differences:**
- **Gate count**: We ran 50 gates; their smallest tested gate count is 100 (they tested 100-1000 gates).
- **Precision**: We use complex64 (8 bytes/amplitude, 8.8 TB state); they use complex128 (16 bytes/amplitude, 17.6 TB state). Our state is half the size.
- **Circuit structure**: Same generation algorithm (50/50 CZ vs random 1Q gates), but different seeds and gate counts produce different circuits.

| Metric | wenbo_engine | AWS QuEST (40q) |
|---|---|---|
| Qubits | 40 | 40 |
| Gates | 50 | 100–1000 |
| Precision | complex64 | complex128 |
| State size | 8.8 TB | 17.6 TB |
| Nodes | 8x i3en.xlarge | 256x c5.18xlarge |
| RAM per node | 32 GB | 144 GB |
| Total RAM | 256 GB | 36.8 TB |
| Approach | Out-of-core (NVMe) | In-memory (distributed RAM) |
| Wall time | 6h 52min (50 gates) | ~7 min for 100 gates (from Figure 2) |
| Cost/hr | ~$2.61 | ~$783 (256 × $3.06) |
| Total cost | ~$18 (50 gates) | ~$91 (100 gates, estimated) |

The core trade-off: wenbo_engine uses **32x fewer nodes** but takes **~60x longer** (comparing our 50 gates to their 100 gates — not the same workload). Total cost is in the same order of magnitude (~$18 vs ~$91 estimated), but the hardware barrier to entry is dramatically lower: 8 commodity NVMe instances vs 256 high-memory nodes.

For their deeper circuits (1000 gates, 40q), QuEST takes ~75 minutes at a cost of ~$978. Our architecture would take proportionally longer at depth 1000 as well, since each additional circuit level requires a full I/O pass over the state vector.

### Estimated Spark Performance (40q / 50 gates)

| Metric | MPI (measured) | Spark (estimated) |
|---|---|---|
| I/O medium | Local NVMe (300 MB/s) | NFS (~125 MB/s) |
| Pipeline | Yes (overlapped) | No (sequential) |
| Nonlocal processing | Per-rank, pipelined | Driver-only, sequential |
| Wall time | **6h 52min** | **~210 hours** (numba) / ~360 hours (numpy) |
| Feasibility | Works on 32 GB nodes | OOM on 32 GB (needs 136 GB groups) |

MPI is ~30x faster than Spark for this workload, and Spark cannot run on 32 GB nodes due to nonlocal group sizes exceeding available RAM.

---

## Scaling Characteristics

| Qubits | State Size | Chunks (512 MB) | Nodes (2 TB NVMe each) |
|--------|-----------|-----------------|----------------------|
| 30 | 8 GB | 16 | 1 |
| 34 | 128 GB | 256 | 1 |
| 38 | 2 TB | 4,096 | 4 |
| 40 | 8 TB | 16,384 | 8 |
| 42 | 32 TB | 65,536 | 32 |
| 45 | 256 TB | 524,288 | 256 |

### Bottleneck Progression

| Scale | Bottleneck | Mitigation |
|-------|-----------|------------|
| <= 30q | None (fits in RAM) | Use ref_dense |
| 30-36q | Disk I/O | Pipeline runner, larger chunks |
| 36-40q | Disk I/O + capacity | MPI across nodes, numba kernels |
| 40-42q | Network + I/O | Qubit reordering, batch splitting |
| 42-50q | Everything | Larger cluster, faster NVMe |

### Key Tuning Parameters

| Parameter | Default | Effect |
|-----------|---------|--------|
| chunk_size | 2^26 (512 MB) | Larger = more local qubits, fewer nonlocal exchanges, more RAM per chunk |
| buffer_depth | auto | Pipeline queue depth, auto-scaled to fit RAM |
| use_wal | true | Crash recovery (slight write overhead for WAL commits) |
| NUMBA_NUM_THREADS | nproc | Cores available to numba prange |

---

## Scaling Beyond 40 Qubits

Each additional qubit doubles the state vector:

| Qubits | State (complex64) | Double Buffer | Min Nodes (2 TB NVMe) |
|--------|-------------------|---------------|----------------------|
| 40 | 8 TB | 16 TB | 8 |
| 42 | 32 TB | 64 TB | 32 |
| 45 | 256 TB | 512 TB | 256 |
| 48 | 2 PB | 4 PB | 2,048 |
| 50 | 8 PB | 16 PB | 8,192 |

The architecture (MPI runner, qubit reordering, batch splitting, pipelining) scales without code changes — only the cluster size and vCPU quota need to increase. The next practical target is 45 qubits (256 TB state, ~128 NVMe nodes on spot instances).

---

## Dependencies

| Package | Required | Purpose |
|---------|----------|---------|
| numpy | yes | Array operations, data types |
| numba | recommended | JIT kernels (14-16x speedup) |
| mpi4py | for MPI runner | MPI communication |
| pyspark | for Spark runners | Distributed task scheduling |
| psutil | optional | RAM auto-detection (fallback: os.sysconf) |
| PuLP | optional | ILP solver for Atlas staging |
| qiskit | optional | Circuit import from Qiskit |
