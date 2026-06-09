# Chunk Size Tuning

## Summary

The chunk size is the most important hyperparameter in the out-of-core runner.
It determines how many qubits are "local" (k = log2(chunk_size)) and directly
affects both compute and I/O performance.

**Optimal on this machine (MacBook Pro, Apple Silicon, 48 GB RAM, SSD):**

| Setting | Value | Rationale |
|---------|-------|-----------|
| chunk_size | **2^16 – 2^18** (0.5 – 2 MB) | Best wall-clock time at n=24 and n=26 |
| fusion | **True** | Always helps (20-40% fewer I/O steps) |
| runner | **pipeline** (buf=4) | ~10-15% faster than single_node at optimal chunk size |

**Default recommendation: `chunk_size=2^16`, `use_fusion=True`, pipeline runner.**

## Why NOT 8 MB (2^20)?

Our initial default was 2^20 (8 MB chunks), reasoning that larger chunks keep
more qubits local. The benchmark shows this is **wrong** — 2^20 is 1.5-3x
slower than the optimal 2^16-2^18:

```
n=26, pipeline, fusion=True:
  2^16 (0.5 MB):   40.2s
  2^18 (2.1 MB):   40.0s  ← best
  2^20 (8.4 MB):  118.9s  ← 3x slower!
  2^22 (33.6 MB): 116.9s
```

The reason: **at 8 MB chunks, the gather/scatter kernel working set exceeds
CPU cache.** The kernel builds index arrays and temporary matrices for each
gate. At 2^20 amplitudes per chunk, the working set for a 1Q gate is ~16 MB
(indices + temp matrix), which overflows L2 cache (~4 MB per core on Apple
Silicon). The matmul_vs_io benchmark confirmed this — throughput drops 7x
when chunks exceed ~4 MB.

Smaller chunks (0.5-2 MB) stay cache-friendly, and the extra non-local gate
overhead from having fewer local qubits is offset by much faster per-chunk
processing.

## The trade-off

```
Smaller chunks ←──────────────── Trade-off ──────────────────→ Larger chunks
  ✓ Cache-friendly kernels          ✓ More qubits local (fewer butterfly passes)
  ✓ Good pipeline overlap           ✓ Fewer files to manage
  ✗ More non-local gates            ✗ Working set overflows CPU cache
  ✗ More I/O steps                  ✗ Pipeline stalls (chunk too big to overlap)
```

The sweet spot is where the chunk fits comfortably in L2/L3 cache AND enough
qubits are local to keep the non-local step count reasonable.

## Measured results

### n=24 (16M amplitudes, 128 MB state, 142 gates, 29 levels)

Fusion analysis:
- 2^14 (k=14): 29→17 steps (41% fewer)
- 2^16 (k=16): 29→15 steps (48% fewer)
- 2^18 (k=18): 29→13 steps (55% fewer)
- 2^20 (k=20): 29→11 steps (62% fewer)
- 2^22 (k=22): 29→9 steps (69% fewer)
- 2^24 (k=24, all local): 29→1 step (97% fewer)

| Runner | Chunk | MB | k | Fusion | Time (s) |
|--------|-------|----|---|--------|----------|
| pipeline | 2^16 | 0.5 | 16 | yes | **8.7** |
| pipeline | 2^18 | 2.1 | 18 | yes | 8.8 |
| single_node | 2^16 | 0.5 | 16 | yes | 9.3 |
| single_node | 2^24 | 134 | 24 | yes | 9.8 |
| pipeline | 2^16 | 0.5 | 16 | no | 10.5 |
| single_node | 2^18 | 2.1 | 18 | yes | 9.4 |
| single_node | 2^16 | 0.5 | 16 | no | 12.2 |
| single_node | 2^18 | 2.1 | 18 | no | 11.9 |
| pipeline | 2^14 | 0.1 | 14 | yes | 14.6 |
| single_node | 2^20 | 8.4 | 20 | yes | 14.3 |
| single_node | 2^22 | 33.6 | 22 | yes | 14.8 |
| single_node | 2^20 | 8.4 | 20 | no | 16.2 |
| pipeline | 2^14 | 0.1 | 14 | no | 17.8 |
| single_node | 2^14 | 0.1 | 14 | no | 20.8 |
| pipeline | 2^20 | 8.4 | 20 | yes | 27.5 |
| pipeline | 2^22 | 33.6 | 22 | yes | 29.3 |
| pipeline | 2^20 | 8.4 | 20 | no | 32.6 |
| pipeline | 2^22 | 33.6 | 22 | no | 34.6 |

### n=26 (67M amplitudes, 537 MB state, 154 gates, 31 levels)

Fusion analysis:
- 2^14 (k=14): 31→19 steps (39% fewer)
- 2^16 (k=16): 31→17 steps (45% fewer)
- 2^18 (k=18): 31→15 steps (52% fewer)
- 2^20 (k=20): 31→13 steps (58% fewer)
- 2^22 (k=22): 31→11 steps (65% fewer)

| Runner | Chunk | MB | k | Fusion | Time (s) |
|--------|-------|----|---|--------|----------|
| pipeline | 2^18 | 2.1 | 18 | yes | **40.0** |
| pipeline | 2^16 | 0.5 | 16 | yes | 40.2 |
| single_node | 2^16 | 0.5 | 16 | yes | 43.8 |
| pipeline | 2^16 | 0.5 | 16 | no | 45.2 |
| pipeline | 2^18 | 2.1 | 18 | no | 45.2 |
| single_node | 2^18 | 2.1 | 18 | yes | 45.5 |
| single_node | 2^16 | 0.5 | 16 | no | 52.4 |
| single_node | 2^18 | 2.1 | 18 | no | 55.2 |
| single_node | 2^20 | 8.4 | 20 | yes | 59.1 |
| single_node | 2^22 | 33.6 | 22 | yes | 59.9 |
| pipeline | 2^14 | 0.1 | 14 | yes | 63.5 |
| single_node | 2^14 | 0.1 | 14 | yes | 67.1 |
| single_node | 2^20 | 8.4 | 20 | no | 68.7 |
| pipeline | 2^14 | 0.1 | 14 | no | 75.6 |
| single_node | 2^14 | 0.1 | 14 | no | 106.4 |
| pipeline | 2^22 | 33.6 | 22 | yes | 116.9 |
| pipeline | 2^20 | 8.4 | 20 | yes | 118.9 |
| pipeline | 2^20 | 8.4 | 20 | no | 137.5 |

## Key observations

1. **Fusion always helps** — 10-40% speedup from merging consecutive local
   levels into single I/O passes and pre-multiplying 1Q gate matrices.

2. **Optimal chunk size is 2^16 – 2^18 (0.5 – 2 MB)**, NOT the larger 2^20
   (8 MB) we initially defaulted to. The cache effect dominates the
   local-qubit benefit.

3. **Pipeline helps at small chunks** (2^14-2^18) where there are many chunks
   and I/O can overlap with compute. At large chunks (2^20+), the pipeline
   overhead (thread coordination, queues) actually hurts.

4. **Larger chunks are NOT always better** despite making more qubits local.
   At n=26, going from 2^18 (15 steps) to 2^20 (13 steps) saves 2 I/O
   passes but triples wall time due to cache misses.

5. **Machine-dependent** — these results are for Apple Silicon with ~4 MB L2
   per core. On x86 with 32 MB L3, the optimal chunk might shift to 2^18-2^20.
   Always re-run the sweep on a new machine.

## How to run the sweep

```bash
python3 -m wenbo_engine.bench.hyperparam_sweep
```

Or for larger scale (edit the script to change circuit size).

## Hardware context

- Machine: MacBook Pro, Apple M3 Pro, 48 GB RAM
- CPU: 14 cores, ~4 MB L2 per core, ~36 MB shared L3
- Disk: NVMe SSD, ~3.5 GB/s seq read, ~3 GB/s seq write
