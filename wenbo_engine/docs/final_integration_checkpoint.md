# Final Integration Checkpoint

**Date:** 2026-06-12
**Integration branch:** `integration/fleet`
**HEAD commit:** `f5106f26a8ce26a24983df6035a5b4b04af98cbb`
**`main` status:** UNTOUCHED — `origin/main` is still `89d355a` (`feat: MPI distributed execution phase`). Every feature was merged `--no-ff` into `integration/fleet`; nothing was ever pushed to `main`.

## Test result
```
python -m pytest wenbo_engine/tests/
703 passed, 43 skipped, 8 warnings in ~69s
```
0 failures, 0 regressions. The 43 skips are environment-gated (no `mpirun`/`mpi4py`/optional deps on a given host); they run when MPI is present.

## Verdict
**SAFE FINAL CHECKPOINT.** The stack is clean, additive, and validated: defaults are unchanged, every optimization is opt-in, recovery invariants hold across the whole stack, and the adaptive planner v2 selects among already-validated mechanisms by predicted wall time.

---

## Merged feature stack (oldest → newest)

Each line is a `--no-ff` merge into `integration/fleet`.

| order | merge | feature |
|---|---|---|
| 1 | `5aa1441` | MPI-nonlocal benchmark suite (base integration) |
| 2 | `f3b3853` / harness | observability / experiment harness + profiling |
| 3 | `fa012b3`,`21cef82`,`16bae1c` | node-local distributed generation recovery (scanner fix) |
| 4 | `445ae81` | fault injection + crash/resume proofs |
| 5 | `56bd4f8` | durable checkpoint / restore (R4) |
| 6 | `4f6f637` | capacity planner integration |
| 7 | `1a60968` | optimizer-v2 (static ablation planner) |
| 8 | `97f5b73` | gate-aware MPI exchange + remote-buffer reuse |
| 9 | `83e6a44` | extent-container storage layout |
| 10 | `544476f` | memory-overlay compute-unit executor + adaptive fallback |
| 11 | `f1cdbdd` | direct extent-backed overlay I/O + non-stabilizer workloads |
| 12 | `e8cc322` | deterministic non-stabilizer `communication_light` |
| 13 | `2d801b2` | recovery-aware hierarchical planner **v1** |
| 14 | `2be8a41` | RAM-aware execution control + automatic `chunk_bits` |
| 15 | `c76370f` | selectable numerical backend (numpy/numba/auto) |
| 16 | `f6ce94a` | MPI-nonlocal telemetry + **diagonal fast path** + true-mixing benchmark |
| 17 | `6498fcb` | MPI-exchange **window feasibility** planner (analysis only) |
| 18 | `4d8f401` | MPI true-mixing **window executor** (off by default) |
| 19 | `e23804c` | calibrated cost-model **telemetry** |
| 20 | `f5106f2` | recovery-aware adaptive planner **v2** (wall-time decision layer) |

## Dependency graph (what builds on what)
- **Recovery (generation + global commit record)** is the foundation; fault injection, durable checkpoint, and every executor commit through it.
- **gate-aware MPI** → required by the diagonal fast path, window feasibility, and window executor.
- **extent storage** → required by direct extent I/O; both consumed by compute-unit execution.
- **memory overlay / compute units** → required by direct extent I/O and RAM-aware overlay bounding.
- **diagonal fast path** + **window feasibility** → prerequisites for the **window executor**.
- **calibrated telemetry** → prerequisite for **planner v2** (it ranks by measured wall-time constants).
- **planner v1** (byte-cost selection) → generalized by **planner v2** (wall-time selection).
- **RAM-aware execution** → consumed by both planners (feasibility gating + auto chunk_bits).

## Defaults — UNCHANGED
The default run (no flags) is byte-for-byte the legacy path:
`storage_layout=chunks`, `execution_mode=step`, `extent_io_mode=materialize`,
`mpi_exchange_mode=naive`, `mpi_window_execution=off`, `kernel_backend=auto`
(safe numpy fallback), `recovery` derived from `--no-wal`. No optimization
activates unless explicitly requested or selected by a planner.

## Opt-in features (only active when requested)
`--storage-layout extents`, `--execution-mode compute_unit`,
`--extent-io-mode direct`, `--mpi-exchange-mode gate_aware`,
`--mpi-window-execution safe`, `--kernel-backend numba`, `--auto-chunk-bits`,
`--ram-budget-gib`, `--max-overlay-chunks`, `--max-remote-buffer-gib`,
`--mpi-window-analysis report`, `--durable.enabled`, `--planner recovery_aware_v1|v2`.
The **diagonal fast path** is the one always-on behavior change, but it is exact
(bit-for-bit vs dense) and only removes provably-unnecessary MPI exchange.

## Features now selected automatically by Planner v2
When `--planner recovery_aware_v2` is used, v2 chooses: `kernel_backend`,
`chunk_bits` (auto), `execution_mode`, `storage_layout`, `extent_io_mode`,
`mpi_exchange_mode`, `mpi_window_execution`, RAM budgets, and commit policy —
ranking candidates by **predicted wall time + recovery risk**, never bytes alone.

---

## Part 2 — Feature stack table

| feature | merge commit | purpose | depends on | key files | tests | cluster validated? | presentation claim |
|---|---|---|---|---|---|---|---|
| observability / experiment harness | `f3b3853` | reproducible, measured runs; artifact bundle (final_summary, profiles, recovery_events) | — | `experiments/run_experiment.py`, `profiling/*` | `test_profiling.py` | yes (every run) | "every result is a reproducible measured artifact" |
| generation recovery | `fa012b3`+`21cef82`+`16bae1c` | immutable `gen_*` + global commit record; node-local distributed scan | harness | `recovery/generation_manager.py`, `recovery/recovery_scanner.py` | `test_recovery*`, `test_generation*` | yes (2-node) | "crash-consistent recovery with a single committed-progress marker, node-local" |
| durable checkpoint / restore | `56bd4f8` | promote committed gens to S3/durable; restore before run | generation recovery | `durable/*` | `test_durable*` | partial (S3 via moto) | "optional off-NVMe durability for committed generations" |
| fault injection | `445ae81` | deterministic crash points; crash/resume proofs | generation recovery | `faults/fault_injector.py`, `faults/fault_points.py` | `test_fault*` | yes (crash before/after commit) | "recovery proven by injected crashes, not assumed" |
| capacity planner | `4f6f637` | offline feasibility: largest exact sim under hw/recovery | — | `planner/capacity_planner.py` | `test_capacity*` | n/a (analytic) | "size a run before writing a byte" |
| gate-aware MPI | `97f5b73` | batch Sendrecv per partner/step + remote-buffer reuse | recovery | `mpi/exchange_planner.py`, `mpi/exchange_batch.py`, `mpi/remote_buffer_cache.py` | `test_gate_aware_mpi.py` | yes | "fewer, larger MPI messages; bit-identical state" |
| extent storage | `83e6a44` | pack many chunks into few extent files | recovery | `storage/extent_store.py`, `storage/extent_manifest.py` | `test_extent*` | yes | "fewer files at rest; same logical chunks" |
| memory overlay / compute units | `544476f` | fuse consecutive local steps; load/apply/write once | extent storage | `runtime/memory_overlay.py`, `runtime/compute_unit.py`, `runtime/overlay_scheduler.py` | `test_memory_overlay*`, `test_compute_unit*` | yes (2-node: 939→134 MB, 7.1→4.2 s) | "fuse local work to cut read/write passes" |
| direct extent I/O | `f1cdbdd` | read/write extent slices directly (no chunk-file round trip) | extent + overlay | `storage/extent_store.py`, `runtime/memory_overlay.py` | `test_direct_extent*` | yes (temp files 17→0) | "zero temporary chunk files on the local path" |
| non-stabilizer benchmarks | `e8cc322`,`856bca4` | workloads are genuinely non-Clifford (real simulation cost) | harness | `bench/communication_workloads.py` | `test_communication_workloads.py` | yes | "benchmarks are non-stabilizer, not trivially classical" |
| RAM-aware execution / auto chunk_bits | `2be8a41` | bound overlay + remote cache to a RAM budget; pick safe chunk_bits; fail early | overlay, capacity planner | `planner/capacity_planner.py`, `mpi/mpi_runner.py`, `mpi/remote_buffer_cache.py` | `test_ram_aware*`, `test_capacity*` | yes (n=36 comm_light no longer OOM) | "RAM, not storage, is the wall — and we bound it" |
| numerical backend selection | `c76370f` | numpy/numba/auto with safe fallback; separate compile time | kernels | `kernel/backend.py`, `kernel/numba_kernels.py` | `test_kernel_backend*` | A/B attempted (numba absent on image) | "pluggable kernel backend, honest fallback" |
| diagonal nonlocal fast path | `f6ce94a` | apply diagonal MPI gates locally (no exchange) | gate-aware MPI | `mpi/diagonal_nonlocal.py`, `mpi/mpi_runner.py` | `test_mpi_diagonal_fast_path.py`, `test_mpi_nonlocal_telemetry.py` | yes (phase-heavy MPI → 0) | "diagonal MPI-nonlocal gates need no remote exchange" |
| MPI window feasibility | `6498fcb` | predict if multi-step windows help (analysis only) | telemetry, diagonal | `mpi/window_planner.py`, `mpi/window_cost_model.py`, `planner/mpi_window_report.py` | `test_mpi_window_planner.py`, `test_mpi_window_cost_model.py` | analytic (matches telemetry) | "measure before building the executor" |
| MPI window executor | `4d8f401` | gather/apply/scatter fused true-mixing window; off by default | feasibility, recovery | `mpi/window_executor.py`, `mpi/mpi_runner.py` | `test_mpi_window_executor.py` | yes (n=30: states match 2.4e-8; bytes 11×↓, slower) | "fuse true-mixing windows correctly; ~11× fewer bytes" |
| calibrated cost-model telemetry | `e23804c` | measure per-phase wall time → cost_model.json | window executor, all paths | `profiling/runtime_timers.py`, `planner/stage_cost_model.py`, `planner/cost_report.py` | `test_calibrated_cost_model.py`, `test_profiling.py` | yes | "wall-time calibration, null-with-reason when unmeasured" |
| recovery-aware planner v1 | `2d801b2` | choose layout/exec/io/MPI by recovery-aware **byte** cost | all execution modes | `planner/recovery_aware_planner.py`, `planner/strategy_candidate.py`, `planner/strategy_selector.py` | `test_recovery_aware_planner*` | yes | "deterministic, grounded strategy selection" |
| recovery-aware planner v2 | `f5106f2` | choose strategy (incl. window+backend) by predicted **wall time** + recovery risk | v1, telemetry, window | `planner/recovery_aware_planner_v2.py`, `planner/cost_model_v2.py`, `planner/cost_report_v2.py` | `test_recovery_aware_planner_v2.py` | yes (8-node n=30 sanity) | "adaptive: optimizes wall time, not bytes; rejects slow windows" |
