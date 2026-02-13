# wenbo_engine vs v3_hisvsim_spark — Comparison Notes

## The core problem with v3

v3 uses **Spark DataFrames for amplitude math**: each quantum state is a
DataFrame of `(idx, real, imag)` rows, and gate application is done via
DataFrame joins, groupBy, and shuffle operations.

This means every single gate application pays:
- Spark job scheduling overhead (~200-500ms per level)
- Serialization/deserialization of all amplitudes
- Parquet read/write for state checkpoints between levels
- Shuffle files for join operations
- For parallel gate fusion: tensor product matrices that grow as 2^(gates_in_group),
  serialized and broadcast to workers

## Benchmark results (MQT Bench + custom circuits, n=3..12)

All results collected on MacBook Pro (48 GB RAM, 14 cores, Apple Silicon).
Correctness verified: all state vectors match between engines (overlap = 1.0).

| Circuit            |  n | Gates |   v3 (s) | wenbo (s) |  Speedup |
|--------------------|----|-------|----------|-----------|----------|
| GHZ-3              |  3 |     3 |     3.64 |     0.003 |   1,097x |
| GHZ-10             | 10 |    10 |     2.65 |     0.009 |     310x |
| GHZ-12             | 12 |    12 |     2.98 |     0.011 |     278x |
| QFT-6              |  6 |    21 |     3.51 |     0.008 |     417x |
| QFT-8              |  8 |    36 |     5.52 |     0.015 |     362x |
| MQT-dj-10          | 10 |    62 |    14.82 |     0.025 |     585x |
| MQT-graphstate-10  | 10 |    20 |     5.80 |     0.004 |   1,318x |
| H-wall-4           |  4 |     4 |     2.80 |     0.002 |   1,356x |
| H-wall-8           |  8 |     8 |     0.86 |     0.002 |     418x |
| H-wall-12          | 12 |    12 |    94.87 |     0.009 |  10,087x |

**Median speedup: ~320x.  Average: ~400x.**


## Scaling limits (wenbo_engine, single machine, 48 GB RAM, 258 GB free disk)

Tested with non-stabilizer circuits (H + T + CNOT layers).

### ref_dense (in-memory, complex128)

| n  | Amplitudes | State   | Time  | Peak RAM |
|----|-----------|---------|-------|----------|
| 24 | 16M       | 268 MB  | 15s   | 1.8 GB   |
| 26 | 67M       | 1.07 GB | 69s   | 6.9 GB   |
| 28 | 268M      | 4.3 GB  | 412s  | 23 GB    |

**Limit: n=28** (peak RAM ~23 GB; n=29 would need ~46 GB).

### Out-of-core (on-disk, complex64)

| n  | Amplitudes | Disk    | Chunks | Time  |
|----|-----------|---------|--------|-------|
| 24 | 16M       | 0.27 GB | 16     | 15s   |
| 26 | 67M       | 1.07 GB | 64     | 61s   |
| 28 | 268M      | 4.3 GB  | 256    | 139s  |
| 29 | 536M      | 8.6 GB  | 512    | 285s  |
| 30 | 1.07B     | 17.2 GB | 1,024  | 631s  |

**Limit: n=30** on this machine (disk: 17 GB × 2 buffers = 34 GB; n=31 would need 34 GB).
All norm checks pass. Correctness cross-validated against ref_dense at n=20.