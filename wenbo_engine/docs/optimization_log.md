# Optimization & Scaling Log — out-of-core MPI state-vector simulator

A consolidated engineering record of the optimization arc on `integration/fleet`:
what was built, why, how it was validated on a real 8-node EC2 cluster, the bugs
and lessons found, and what remains. Each feature was developed on its own
branch/worktree, tested, and merged `--no-ff` into `integration/fleet` (never
`main`).

Merge history (newest first):

```
2b27b19  feat: RAM-aware execution control + automatic chunk_bits   (agent/ram-aware-execution, not yet merged)
2d801b2  merge: recovery-aware hierarchical planner v1
485ce0c  fix(planner): recovery-aware v1 cleanup (candidate/aggregation/MPI/run_experiment)
1ef53b0  feat(planner): recovery-aware hierarchical planner v1
e8cc322  merge: deterministic non-stabilizer communication_light
0f1bb5f  fix(bench): make communication_light deterministically non-stabilizer
f1cdbdd  merge: direct extent-backed overlay I/O + non-stabilizer workloads
2d1f214  feat: direct extent-backed overlay I/O (--extent-io-mode direct)
856bca4  test/bench: make communication workloads non-stabilizer
544476f  merge: memory-overlay compute-unit executor + adaptive fallback
```

Throughout, four invariants were preserved (verified on every cluster run):
`recovery_mode=generation`, `source_of_truth=global_commit_record`,
`wal_json_present=false`, and **node-local work_dir** (each node holds only its
own `rank_NNNN` partition; rank 0 additionally holds `commits/` + `run.json`).

---

## 1. Features implemented

### 1.1 Memory-overlay + compute-unit executor (`--execution-mode compute_unit`)
- **Why:** the step path does one read+write pass of the partition per circuit
  step. Consecutive *local-only* steps act entirely within a chunk, so they can
  be fused: load a chunk once, apply all the run's local gates, write once.
- **How:** `runtime/memory_overlay.py` (RAM overlay over logical chunks, LRU
  eviction under a chunk budget), `runtime/compute_unit.py`
  (`execute_local_unit`), `runtime/overlay_scheduler.py` (`build_compute_units`
  groups consecutive local-only steps into one `local` unit; nonlocal steps stay
  as `step` units). Every unit commits exactly one generation → recovery cadence
  unchanged.
- **Result (2-node cluster):** `communication_light` read 939→134 MB, wall
  7.11→4.23 s.

### 1.2 Adaptive compute-unit fallback (`--compute-unit-min-gates N`, default 4)
- **Why:** fusing a *short* local run into a compute unit adds overlay overhead
  + a commit for little benefit; `mixed_staged` regressed because of tiny units.
- **How:** a local run shorter than `min_gates` falls back to one `step` unit per
  step (`fallback=True`). Default `step` execution unchanged.

### 1.3 Gate-aware MPI exchange (`--mpi-exchange-mode gate_aware`)
- **Why:** naive does one `Sendrecv` per chunk per gate; many gates in a step
  share the same remote partner/chunk.
- **How:** `mpi/exchange_planner.py` resolves each MPI gate to its partner +
  chunk pairs and groups by partner; `mpi/remote_buffer_cache.py` fetches a
  remote chunk once per partner per step and reuses it. Same kernel math → same
  final state; fewer/batched Sendrecv calls.
- **Measured:** mpi-heavy n=24, 4 ranks — sendrecv 320 (naive) → 80 (gate_aware)
  at identical bytes (gates need distinct chunks here, so the win is call-count,
  not bytes).

### 1.4 Extent storage layout (`--storage-layout extents`)
- **Why:** one file per chunk explodes the file count; pack many chunks into a
  few `extent_NNNN.dat` files + an `ExtentManifest` mapping
  `chunk_id → (extent_id, offset, length, checksum)`.
- **How:** `storage/extent_store.py`, `storage/extent_manifest.py`; on-disk
  format is identical whether written by pack or by direct I/O, so recovery /
  validator / durable restore need no special-casing.

### 1.5 Direct extent-backed overlay I/O (`--extent-io-mode direct`)
- **Why:** the materialize path round-trips every committed generation through
  temporary chunk files (unpack extents→chunks, run overlay, repack
  chunks→extents). Direct reads logical chunks straight from source extent
  slices and writes dirty chunks straight into destination extents.
- **How:** `extent_store.read_chunk_from_extent` + `ExtentWriter` (fsync + atomic
  rename before manifest publication); `memory_overlay` gained reader/writer
  callables; `compute_unit.execute_local_unit_direct`.
- **Measured:** `communication_light` (all-local) — temp chunk files 17→**0**,
  byte-identical final state; faster as state grows (n=28/2 GB: ~5.0–5.5 s
  materialize vs ~2.2–3.6 s direct). Benefit is proportional to the local
  fraction of the circuit.

### 1.6 Non-stabilizer benchmark workloads
- **Why:** `rank_nonlocal_heavy` / `mpi_nonlocal_heavy` were built from CNOT/CZ
  only → pure **stabilizer** circuits (Gottesman–Knill simulable), a weak test
  for a full state-vector engine.
- **How:** inject `CR(k≥3)` controlled-phase gates on the *same* qubit pairs the
  generators already used (preserving the locality class exactly), and force
  `communication_light`'s first gate to a deterministic non-Clifford
  (`RZ(π/7)`/`CR(3)`) so it's non-stabilizer for **every** seed.
- **Classifier subtlety:** single-qubit `R(k)` is Clifford for `k≤2` (Z, S);
  controlled `CR(k)` is Clifford only for `k=1` (CZ) — `CR(2)=controlled-S` is
  **not** Clifford. `final_summary.json` carries `is_stabilizer`,
  `non_clifford_gate_count`, `non_clifford_gate_types`.

### 1.7 Recovery-aware hierarchical planner v1 (`--planner recovery_aware_v1`)
- **Why:** make the strategy choice (layout × execution × extent-I/O × MPI mode)
  before a run, deterministically and explained, then compare predicted vs
  actual after.
- **How:** `planner/{strategy_candidate,strategy_selector,recovery_aware_planner,
  cost_report,stage_plan}.py`. Candidates: `chunks+step+naive`,
  `chunks+step+gate_aware`, `extents+step+gate_aware`,
  `extents+compute_unit+{materialize,direct}+gate_aware`. MPI estimates are
  **deterministic** (derived from the real `exchange_planner`, not a heuristic) —
  predicted sendrecv/bytes matched measured to 0%.
- **Selections (n=24, 4 ranks):** light → `extents+compute_unit+direct+gate_aware`;
  mixed/mpi → `chunks+step+gate_aware` (not forced into extents).

### 1.8 RAM-aware execution control + automatic chunk_bits
- **Why (the headline finding):** the 8-node ladder OOM'd while NVMe stayed >93%
  free. The limit is per-rank **RAM working set**, which the capacity planner
  (storage-only) didn't model.
- **How:**
  - `capacity_planner`: `estimate_peak_ram` (overlay/step chunks + numpy temp +
    MPI sendrecv + remote cache + metadata, × safety margin),
    `recommend_chunk_bits`, and **separate** `storage_feasible` / `ram_feasible`.
  - `memory_overlay`: bounded overlay streams a local unit chunk-by-chunk
    (peak ≈ 1 chunk), with a `peak_resident_bytes` high-water mark.
  - `remote_buffer_cache`: optional `max_bytes` LRU eviction (a post-eviction
    use just re-issues a Sendrecv — correctness preserved).
  - `mpi_runner.run`: `--ram-budget-gib / --auto-chunk-bits / --max-overlay-chunks
    / --max-remote-buffer-gib`; `_apply_ram_budgets` converts the budget to an
    overlay chunk budget + remote byte cap and **fails early** (clear
    `RuntimeError`) before allocation if one chunk + temp can't fit.
  - `final_summary.json` gains `ram_budget_gib`, `estimated_peak_ram_gib`,
    `chunk_bits`, `chunk_bytes`, `auto_chunk_bits_enabled`,
    `recommended_chunk_bits`, `max_overlay_chunks`, `max_remote_buffer_gib`,
    `overlay_peak_ram_gib`, `remote_buffer_peak_gib`, `ram_feasible`.
- **Defaults unchanged** unless a RAM option is set.

---

## 2. Cluster validation (8 × i3en.xlarge, us-east-1d, 1 rank/node)

Hardware per node: 4 vCPU, 30 GiB RAM, 2.3 TiB instance-store NVMe. numba was
absent → the numpy kernel path (same as CI). Code synced as a `git archive`
tarball, md5-verified identical on every node.

### 2.1 Correctness / recovery / scaling (n=24, then n=28)
- 2→4-node wall-time speedup ~1.8–2.0× per workload; per-rank state halves with
  rank count; MPI bytes constant, sendrecv calls rise with partners (expected).
- Crash/recovery (fault injection): crash **after** global commit → resume from
  the committed gen; crash **before** commit → partial generation **rejected**,
  resume from the last committed gen. final_norm=1.0 both ways.

### 2.2 NVMe-only larger-scale ladder (n=30…)
| workload | completed | first OOM |
|---|---|---|
| communication_light (extents+compute_unit+direct) | n=30, 32, 34 | **n=36** |
| mpi_nonlocal_heavy (chunks+step+gate_aware) | n=30, 32 | **n=34** |
| mixed_staged | n=30, 32 | — |

- Stage-time scaled a clean ~4× per +2 qubits (kernel-bound on numpy/4 vCPU);
  NVMe stayed >93% free throughout. **The wall was RAM, not storage.**
- n=38 was **not** run (projected stage ≈3.6 h > the 2-h cap, and it would OOM).
  No 38q/45q/50q claim is made.

### 2.3 RAM-aware fix validation
- Preflight: default n=36 → `ram_feasible=False` (est peak 211 GiB) with a
  recommended chunk_bits; `--auto-chunk-bits` → `ram_feasible=True` at chunk_bits
  29 (est peak 19.8 GiB ≤ 21 GiB budget).
- **n=36 communication_light now completes (no OOM)** at auto chunk_bits=29; the
  streamed compute committed gen 1 (2853 s, no OOM); final_norm≈1; invariants
  intact.
- n=34 mpi_nonlocal_heavy: auto chunk_bits=27; **2/9 steps committed with no
  OOM** (past the point the old default-chunk_bits run was killed); not run to
  full completion (stopped to save cost — per-step memory is constant so the tail
  carries no new OOM risk).
- **All 8 instances were stopped after every session** (verified 0 running).
  Approx cost: ~$1.7 (2/4-node validation), ~$6.3 (NVMe ladder), ~$8 (RAM-aware).

---

## 3. Bugs found & fixed (each surfaced by real runs)

1. **`compute_norm` end-of-run OOM (RAM-aware session).** It built the whole
   per-rank partition as a list, then upcast each chunk to complex128 — 64 GiB →
   128 GiB at n=36. The 47-min compute succeeded, then norm OOM'd. **Fix:** a
   streaming generator (`_iter_committed_logical_chunks`) + float64-accumulated
   `|amp|²` (abs on complex64 → float32, half the bytes); peak ≈ one chunk.
2. **Unbounded compute-unit overlay (the n=36 OOM).** The overlay held all
   `n_chunks_per_rank` chunks resident — the whole partition. **Fix:** bound it
   (stream) via `ram_budget_chunks`, sized from the RAM budget.
3. **Unbounded gate-aware remote-buffer cache (the n=34 mpi OOM).** It grew to
   the whole remote partition. **Fix:** `max_bytes` LRU eviction.
4. **`NUMPY_TEMP_FACTOR` under-modeled (mid-validation).** Initial factor 1.0
   said chunk_bits=30 (8 GiB) fit a 21 GiB budget, but it OOM'd (real peak
   ≈4× chunk: raw read + frombuffer copy + 2q-gate in/out/intermediate +
   `tobytes` writeback). **Fix:** calibrated to 3.0 (total 4×) from the observed
   8 GiB×4≈32 GiB > 30 GiB OOM → auto then picks chunk_bits=29.
5. **`_compile_steps` requires `g["params"]`.** Standalone planner calls must run
   `validate_circuit_dict` first (it normalizes `params={}`); the runner already
   does.
6. **`final_summary` fields silently `None`.** Three sinks must each carry a
   field: the result dict, `config.json`, and the `final_summary` required dict.

---

## 4. Lessons learned

### Engineering / architecture
- **RAM, not storage, is the scaling wall on these nodes.** i3en gives 2.3 TiB
  NVMe but only 30 GiB RAM; the binding constraint is the per-rank working set =
  resident chunks × chunk_bytes + kernel temporaries (+ remote cache for MPI).
  A capacity planner that models only storage is dangerously optimistic.
- **Smaller chunk_bits alone does NOT fix a compute-unit OOM** — it raises the
  chunk count but keeps the total per-rank state the same. You must *bound the
  overlay so it streams* AND size chunks so one chunk + temps fit RAM. Both
  levers are needed.
- **The kernel temporary is the hidden multiplier.** A 2-qubit gate on a chunk
  transiently needs ~3–4× the chunk (input + output + einsum intermediate, plus
  the read/writeback copies). Model peak as `(resident + temp_factor) ×
  chunk_bytes`, not just `chunk_bytes`.
- **gate_aware MPI reduces call count, not necessarily bytes.** When each gate
  needs distinct remote chunks, total bytes are unchanged; the win is fewer
  blocking Sendrecv round trips. Price call latency separately if you want the
  cost model to prefer it.
- **Deterministic, planner-derived estimates beat heuristics.** Reusing the real
  `exchange_planner` for the MPI prediction gave 0% error vs a fixed reuse factor
  that was +100%/−50% off. Calibrate cost-model constants against measured runs.
- **The on-disk format being layout-agnostic paid off repeatedly** — direct vs
  pack extents, durable restore, and recovery validation all share one format,
  so new write paths needed no recovery changes.

### Operational (EC2 / MPI)
- **i3en instance-store NVMe is ephemeral** — wiped on stop/start and *not*
  auto-remounted. Re-`mkfs`/mount every session. The root EBS (home, code, SSH
  config, private IPs) persists; only public IPs change.
- **Instance-store device name is not stable** (`nvme0n1` vs `nvme1n1` across
  identical instances). Auto-detect the ~2.3 TB disk; don't hardcode.
- **Private IPs persist across stop/start; public IPs don't.** Put private IPs in
  the MPI hostfile (intra-VPC), use public IPs only for the controller's SSH.
- **Validate the medium, not just the path.** Early levels accidentally ran on
  the 20 GB root EBS because the instance NVMe wasn't mounted — `df -h /mnt/nvme`
  before trusting it.
- **Stop instances in every exit path.** Treated as a hard requirement; verified
  `0 running` after each session.

### Shell / tooling traps (cost real debugging time)
- **zsh does not word-split unquoted `$var`.** `for ip in $LIST` iterates once
  with the whole string. Use an explicit literal list or `${=LIST}`.
- **Nested single quotes inside `ssh '... \"...\"'` collapse.** Don't inline
  multi-quote scripts over SSH — write a local `.sh` and pipe it: `ssh host
  'bash -s' < script.sh`.
- **Long-running cluster jobs:** launch with `nohup ... &` on the node and poll
  the log/artifacts; don't hold a foreground SSH for an hour.
- **`git archive HEAD` + md5 on every node** is a clean, verifiable code-sync for
  a private repo without deploy keys.

### Scientific-validity
- A Clifford-only benchmark is a stabilizer circuit and **not** a real stress
  test. Force non-Clifford content while preserving the locality class. Mind the
  controlled-gate Clifford boundary (`CR(2)` is *not* Clifford).

---

## 5. Improvements / future work
- **Measure RAM peak as RSS**, not just overlay/remote high-water marks; fold a
  measured per-gate-arity temp factor back into `NUMPY_TEMP_FACTOR`.
- **numba on the cluster:** the runs were numpy-bound on 4 vCPU; the kernel path
  is the dominant cost at large n. A numba/threaded kernel would move the
  frontier far more than any I/O change.
- **Planner v2:** search over chunk_bits / rank count / durable policy jointly;
  add a `chunks+compute_unit` candidate; model call-latency vs bytes for MPI.
- **Finish the n=34 mpi RAM-safe run** to capture final_norm + measured peaks;
  consider an n=36 single end-to-end run (compute + norm) once cluster time
  allows, to capture `overlay_peak_ram_gib` from the compute (the validated run
  resumed, so that field read 0).
- **Auto-detect node RAM more precisely** (cgroup limits, not just SC_PHYS_PAGES)
  and default the budget from available rather than total RAM.
- **Per-stage RAM telemetry** in `ram_metrics.json` (peak per unit/step) to spot
  the worst stage rather than a single run-wide peak.

---

## 6. Quick reference — flags added this arc
```
--mpi-exchange-mode {naive,gate_aware}
--storage-layout {chunks,extents}
--execution-mode {step,compute_unit}
--compute-unit-min-gates N            # default 4
--extent-io-mode {materialize,direct} # default materialize
--planner recovery_aware_v1
--ram-budget-gib FLOAT                # default: 70% of node RAM when auto
--auto-chunk-bits
--max-overlay-chunks INT
--max-remote-buffer-gib FLOAT
```
All default to the pre-existing behavior (`chunks` / `step` / `naive` /
`materialize`, unbounded RAM) so nothing changes unless explicitly enabled.
