# Recovery Strategies for Out-of-Core State Vector Simulation

## The core problem

A quantum state vector of n qubits has 2^n complex amplitudes.
At 40 qubits that is **8 TB**; at 50 qubits, **8 PB**.
Simulating a circuit means applying gate operations one level (or batch)
at a time — each "step" reads and rewrites the entire state.
A single step can take **minutes (40q)** to **days (50q)**.

If the process crashes mid-step, we need to resume without:
- corrupting the state, or
- redoing more work than necessary.

## Why recovery is hard

Gate application is **not idempotent**.  Applying a gate twice gives the
wrong result (U applied twice = U², not U).
This means: if a chunk was already overwritten in-place and we crash,
we cannot simply re-run it — the input is gone.

Any safe recovery scheme must preserve the **original input** for every
chunk until we are certain the output is durable.

---

## Strategy 1 — Double-buffer + step-level WAL

**How it works:**
Two directories (`state_a`, `state_b`) alternate as source and destination.
Each step reads every chunk from src, computes, writes to dst.
After all chunks: write manifest, then commit the WAL
(`committed_buf`, `done_steps`).
Source is never modified.

**On crash:** WAL says which buffer is committed.
Wipe dst, redo the step from src.

```
Disk:  [state_a]  ←read    [state_b]  ←write
WAL:   { committed_buf: "a", done_steps: 4 }
```

| Aspect | Rating |
|--------|--------|
| Disk space | **2×** (two full copies of the state) |
| I/O per step | **1×** (read src + write dst = 2 × state size) |
| Recovery cost | **Redo entire step** (minutes at 40q, days at 50q) |
| Code complexity | **Very low** — ~20 lines of WAL logic |
| Correctness | **Trivially correct** — src is never touched |

**Best for:** ≤ 40 qubits where a step takes < 1 hour.
**This is the current implementation.**

---

## Strategy 2 — Double-buffer + intra-step checkpointing

**How it works:**
Same as Strategy 1, but the WAL also records a **checkpoint counter**
within the current step: how many work items (chunks or non-local groups)
have been durably written to dst.

On crash: resume from the checkpoint — skip already-written items,
re-compute only the remainder.  Items after the checkpoint may be stale
(from a previous use of the same buffer, 2 steps ago) or partially
current; either way they get overwritten atomically via `os.replace`.

The checkpoint is updated every N items (configurable, default 10,000).
Each checkpoint write is one atomic `fsync` + `os.replace` on a small
JSON file (~1–10 ms on SSD, ~50–100 ms on network FS).

```
WAL: {
  committed_buf: "a",
  done_steps: 4,
  checkpoint: { step: 4, dst: "b", items_done: 450000 }
}
```

**On crash:** Read WAL → skip items 0..(items_done−1) → redo from
items_done onward.  At most `checkpoint_interval` items of work
are lost (~80 GB at 10k × 8 MB → ~8 seconds at 10 GB/s).

| Aspect | Rating |
|--------|--------|
| Disk space | **2×** |
| I/O per step | **1× + negligible** (WAL writes every N chunks) |
| Recovery cost | **Redo at most N items** (seconds, regardless of qubit count) |
| Code complexity | **Low** — adds ~30 lines over Strategy 1 |
| Correctness | **Correct** — src untouched; items are idempotent because input always comes from src |

**Best for:** ≥ 45 qubits where a step takes hours/days.
Not yet implemented — planned upgrade when moving to cluster-scale runs.

---

## Strategy 3 — In-place + per-chunk bitmap

**How it works:**
Single state directory.  Overwrite each chunk in place after computing
the new value.  Maintain a bitmap (or progress counter) tracking which
chunks have been processed.

**The fatal flaw:** We cannot cheaply distinguish "already processed" from
"not yet processed" chunks.  Options:
- Per-chunk WAL update (1 `fsync` per chunk → 1 billion fsyncs at 50q → **days** of WAL overhead alone).
- Bitmap file flushed periodically → gap between flush and crash is ambiguous (some chunks between last flush and crash may or may not have been overwritten; re-applying a gate to an already-overwritten chunk gives the **wrong result**).

| Aspect | Rating |
|--------|--------|
| Disk space | **1×** |
| I/O per step | **1× + massive WAL overhead** (if exact tracking) |
| Recovery cost | Depends on tracking granularity |
| Code complexity | **High** — bitmap management, edge cases |
| Correctness | **Fragile** — any tracking error → silent data corruption |

**Verdict:** Not recommended.  The non-idempotency problem makes this
dangerous unless tracking is perfectly precise, which is prohibitively
expensive.

---

## Strategy 4 — In-place + batch journal

**How it works:**
Process chunks in batches of N.  Before modifying a batch, copy the
original chunk data to a **journal** (backup).  After the batch completes,
advance the checkpoint and discard the journal.

On crash: restore the incomplete batch from the journal (all chunks
in the batch revert to pre-batch state), then redo the batch.

```
Step flow:
  for each batch of N chunks:
    1. Write originals to journal (N × 8 MB)
    2. Overwrite chunks in place
    3. Advance checkpoint, truncate journal
```

| Aspect | Rating |
|--------|--------|
| Disk space | **1× + journal** (N × chunk_size, e.g. 80 GB for N=10k) |
| I/O per step | **1.5×** (every chunk is written twice: journal + in-place) |
| Recovery cost | Redo at most one batch |
| Code complexity | **Medium** — journal management, restore logic |
| Correctness | **Correct** if journal write is durable before in-place write |

**Trade-off vs Strategy 2:**
Saves ~50% disk (1× vs 2×) but costs ~50% more I/O (1.5× vs 1×).
At 50 qubits: saves **8 PB** of disk but adds **~22 hours** of extra
write time per step (at 100 GB/s).

**Best for:** When disk space is the hard constraint and I/O time is
acceptable.

---

## Strategy 5 — In-memory (no disk recovery needed)

**How it works:**
Keep the entire state vector in RAM / GPU VRAM across multiple nodes.
Use MPI or NCCL for inter-node communication (butterfly exchange).
No disk I/O during simulation.

Recovery is via periodic **snapshots** to disk (optional) or simply
restarting from scratch (acceptable if total runtime is < job time limit).

This is how Qiskit Aer on LUMI achieved 44 qubits (2024).

| Qubits | RAM needed | Nodes (256 GB each) | Nodes (A100 80GB) |
|--------|-----------|--------------------|--------------------|
| 35 | 256 GB | 1 | 4 |
| 40 | 8 TB | 32 | 128 |
| 45 | 256 TB | 1,024 | 4,096 |
| 50 | 8 PB | 32,768 | 131,072 |

| Aspect | Rating |
|--------|--------|
| Disk space | **0** during simulation (state in RAM) |
| I/O per step | **0** (unless snapshotting) |
| Recovery cost | Restart from scratch (or from last snapshot) |
| Code complexity | **Low** for simulation, **high** for MPI/NCCL orchestration |
| Correctness | Trivial (no partial state on disk) |

**Best for:** When you have enough aggregate RAM/VRAM.
Not viable for 50 qubits unless you have >30,000 nodes.

---

## Comparison summary

| | Disk | I/O overhead | Max recovery loss | Complexity | Correctness risk |
|---|---|---|---|---|---|
| **1. Double-buf + step WAL** | 2× | 0% | Entire step | Very low | None |
| **2. Double-buf + checkpoint** | 2× | ~0% | N chunks (~secs) | Low | None |
| **3. In-place + bitmap** | 1× | High (fsyncs) | Varies | High | **Dangerous** |
| **4. In-place + journal** | ~1× | +50% | One batch | Medium | Low |
| **5. In-memory (MPI/GPU)** | 0 | 0% | Full restart | Medium | None |

---

## Recommendation by scale

| Qubits | State size | Recommended strategy |
|--------|-----------|---------------------|
| ≤ 30 | ≤ 8 GB | In-memory, no recovery needed |
| 31–40 | 16 GB – 8 TB | Strategy 1 (simple double-buffer) |
| 41–45 | 16 TB – 256 TB | Strategy 2 (double-buffer + checkpoint) |
| 46–50 | 512 TB – 8 PB | Strategy 2 if disk allows 2×; otherwise Strategy 4 |
| 40–44 (cluster) | 8 TB – 128 TB | Strategy 5 (in-memory MPI) if nodes available |

## What we implement

**Strategy 1** — double-buffer with step-level WAL.
It covers development (small circuits on laptop) through moderate
production (≤ 40 qubits on cluster storage), with minimal code
complexity and zero correctness risk.

Upgrading to **Strategy 2** (intra-step checkpointing) for ≥ 45 qubit
runs is a straightforward extension: add a `checkpoint` field to the WAL
and skip already-written items on recovery.  The double-buffer layout
is already in place — only the WAL granularity changes.
