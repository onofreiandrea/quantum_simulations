# Final Evidence Tables

All numbers are from real runs on the merged `integration/fleet` stack (HEAD
`f5106f2`). Cluster results are 8 × i3en.xlarge (4 vCPU, 30 GiB RAM, 2.3 TiB
instance-store NVMe), us-east-1d, 1 MPI rank per node, node-local `/mnt/nvme`.
Local results are `mpirun -np 4` on the dev host.

---

## A. Correctness and recovery

| property | result | evidence |
|---|---|---|
| final_norm (all workloads, local n=24) | 1.0000000 ± ~1e-7 | `final_summary.json` per run |
| final_norm (cluster n=30 comm_light) | 1.0000001 | planner-v2 cluster run |
| final_norm (cluster n=30 mixing_heavy) | 0.9999997 | planner-v2 / window-executor runs (identical) |
| source_of_truth | `global_commit_record` (all runs) | `recovery_events.json` |
| wal_json_present | `false` (all generation runs) | `recovery_events.json` |
| crash **before** window/step commit | rolls back to previous committed generation, re-executes, correct | window-executor crash test (run1 exit 137 → run2 correct, re-ran window) |
| crash **after** commit | resumes from the committed (window) generation, does NOT re-run | window-executor crash test (resume from gen 11, windows_executed=0) |
| node-local work_dir | each node holds only its `rank_NNNN`; only rank 0 has `commits/` + `run.json` | verified on 8 nodes every cluster session |
| window state == per-step state (n=30) | `max|OFF − SAFE| = 2.4e-8` over all 1,073,741,824 amplitudes | window-executor cluster A/B full-state compare |

---

## B. Scaling

| item | result |
|---|---|
| 2→4 node wall-time speedup | ~1.8–2.0× per workload; per-rank state halves with rank count; MPI bytes constant, sendrecv calls rise with partner count (expected) |
| 8-node n=28 | completed; correct; recovery invariants intact |
| NVMe ladder — communication_light (extents+compute_unit+direct) | completed n=30, 32, **34**; first OOM at **n=36** (pre-RAM-aware) |
| NVMe ladder — mpi_nonlocal_heavy (chunks+step+gate_aware) | completed n=30, 32; first OOM at **n=34** (pre-RAM-aware) |
| NVMe ladder — mixed_staged | completed n=30, 32 |
| stage-time scaling | ~4× per +2 qubits (kernel-bound on numpy / 4 vCPU) |
| NVMe utilization throughout ladder | >93% free — **the wall was RAM, not storage** |
| n=38 / 45q / 50q | **NOT run, NOT claimed** (projected stage > 2 h cap and would OOM) |
| OOM limits & fixes | (1) `compute_norm` end-of-run OOM → streaming norm; (2) unbounded compute-unit overlay (n=36) → bounded overlay (max_overlay_chunks); (3) unbounded gate-aware remote cache (n=34) → bounded LRU cache; (4) NUMPY_TEMP_FACTOR recalibrated 1.0→3.0 from observed peak |

---

## C. Optimization results

| optimization | measured effect |
|---|---|
| compute_unit + direct extent I/O | `communication_light` 2-node: read 939→134 MB, wall 7.11→4.23 s; temporary chunk files 17→**0** |
| RAM-aware execution (n=36 comm_light) | default n=36 `ram_feasible=False` (est peak 211 GiB) → `--auto-chunk-bits` picks chunk_bits=29 (est peak 19.8 GiB ≤ 21) → **n=36 completes, no OOM** (compute committed gen 1 in 2853 s, final_norm≈1) |
| numba cluster A/B | numba **absent** on the cluster image → runs were numpy-bound on 4 vCPU; backend selection falls back safely to numpy and records the reason. No numba speedup is claimed on i3en. |
| diagonal fast path (phase-heavy) | MPI-nonlocal pairwise traffic → **0** (sendrecv 0, mpi_bytes 0); norm 1.0; exact vs dense |
| diagonal fast path (mpi_nonlocal_heavy) | diagonal gates skipped: sendrecv 80→12, pairwise 2.68 GB→0.40 GB (local n=24) |
| MPI window executor (mixing, n=30) | pairwise Sendrecv 160→0, total comm volume **171.8 GB → 15.0 GB (~11.4×)**; **but wall time higher** (collective scatter + leader compute + segment overhead dominate). Local n=24: segment 6.05 s vs per-step pairwise 0.34 s + nonlocal kernel 0.17 s |
| Planner v2 | makes the **correct adaptive choice** per workload (see section D); selects window only when predicted wall time is lower |

---

## D. Planner v2 decisions (per workload)

Local `-np 4`, n=24, generation, `--auto-chunk-bits --ram-budget-gib 21`,
i3en-calibrated defaults. (comm_light / mixing also confirmed on 8-node n=30.)

| workload | selected strategy | reason | rejected risky strategy | reason rejected |
|---|---|---|---|---|
| communication_light | extents + compute_unit + **direct** + gate_aware + window_off (numba) | lowest predicted wall time; real local runs to fuse; direct extent I/O avoids chunk-file round trip | window_safe | no executable true-mixing window (all-local) |
| mpi_nonlocal_phase_heavy | chunks + step + gate_aware + window_off | diagonal fast path drives MPI to 0; nothing to fuse | window_safe | no executable true-mixing window (all diagonal) |
| mpi_nonlocal_heavy | chunks + step + gate_aware + window_off | lowest predicted wall time; permutation/mixed gates | window_safe | no useful consecutive true-mixing window |
| mpi_nonlocal_mixing_heavy | chunks + step + gate_aware + window_off | window predicted **slower** despite ~11× fewer bytes | **window_safe** | "MPI window predicted slower: collective + leader + segment cost > per-step pairwise — bytes saved ≠ wall-time win" |
| mixed_staged | chunks + step + gate_aware + window_off | step preferred; avoids inefficient tiny compute units | extents+compute_unit | "compute_unit fusion yields only tiny local fragments — step preferred" |

**Synthetic control:** with an injected calibration where collective MPI is
cheap and pairwise is slow, v2 **flips** mixing_heavy to `window_safe` — proving
the decision is cost-model-driven, not hardcoded.
