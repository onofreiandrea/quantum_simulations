# Final Presentation Outline (14 slides)

Each slide: **title · bullets · figure/table · what to say.**
Figures referenced exist under `wenbo_engine/docs/` (architecture diagrams,
`results_tables.md`, `optimization_log.md`) or are produced by the artifact
bundle (`final_summary.json`, `decision_report_v2.json`).

---

### 1. Problem and motivation
- Exact state-vector simulation is memory-bound: 2^n complex amplitudes.
- A single node cannot hold large n; we need out-of-core + multi-node.
- Must stay **exact** and **non-stabilizer** (not classically cheatable).
- Must be **crash-recoverable** on commodity cloud (spot/instance-store).
- **Figure:** state-vector size vs n (8 GB at n=30, 128 GB at n=34).
- **Say:** "We simulate the full wavefunction exactly; the challenge is doing it across nodes, out-of-core, and surviving crashes — without pretending it's a Clifford circuit."

### 2. Baseline architecture
- State partitioned into chunks on node-local NVMe; double buffer.
- MPI: local / rank-nonlocal / MPI-nonlocal gate classification.
- One rank per node; node-local work_dir (no shared FS).
- **Figure:** `docs/system_architecture.md` / `mpi_architecture.pdf`.
- **Say:** "Each rank owns its slice on local NVMe and exchanges only at the gate boundaries that cross a rank bit."

### 3. Recovery model
- Generation = immutable `gen_*` dir + a single **global commit record**.
- `source_of_truth = global_commit_record`; no `wal.json`.
- Node-local distributed scan (each rank validates only its partition).
- Crash before commit → rollback; after commit → resume.
- **Table:** Evidence A (correctness/recovery).
- **Say:** "Progress is marked by exactly one record. We proved recovery by injecting crashes before and after commit on real nodes."

### 4. Observability and experiment harness
- Every run emits an artifact bundle (final_summary, profiles, recovery_events).
- Calibrated timing telemetry → `cost_model.json` (null-with-reason if unmeasured).
- Reproducible: config + circuit + git commit captured per run.
- **Figure:** an example `final_summary.json` timing block.
- **Say:** "Nothing here is a hand-wave — every claim is a measured artifact, and unmeasured quantities are reported as null with a reason, never faked."

### 5. Storage optimizations
- Extent layout: pack many chunks into few extent files.
- Direct extent I/O: read/write slices, no chunk-file round trip (temp files 17→0).
- **Table:** Evidence C row 1 (939→134 MB, 7.11→4.23 s).
- **Say:** "We cut the local read/write traffic and eliminated temporary files on the all-local path."

### 6. Communication optimizations
- Gate-aware MPI: batch Sendrecv per partner/step + remote-buffer reuse.
- Diagonal fast path: diagonal MPI gates need **no** exchange (phase-heavy → 0).
- MPI window executor (true-mixing): gather/apply/scatter, ~11× fewer bytes.
- **Table:** Evidence C diagonal + window rows.
- **Say:** "We separated MPI gates by type. Diagonal ones we apply locally — zero exchange. True-mixing windows move 11× fewer bytes — but watch the next point."

### 7. Compute optimizations
- Memory-overlay compute units: fuse consecutive local steps, load/apply/write once.
- Adaptive fallback (`--compute-unit-min-gates`) avoids tiny fragments.
- Selectable numerical backend (numpy/numba/auto, safe fallback).
- **Figure:** compute-unit fusion diagram (`docs/slide5_diagrams.pdf`).
- **Say:** "Local runs get fused into one pass; we never fuse so aggressively that we create tiny useless units."

### 8. RAM-aware execution
- The real wall on i3en is **RAM (30 GiB), not NVMe (2.3 TiB)**.
- Bound overlay + remote cache; auto `chunk_bits`; fail early if infeasible.
- n=36 communication_light: OOM → **completes** at chunk_bits=29.
- **Table:** Evidence B (OOM limits & fixes) + C (n=36 fix).
- **Say:** "Storage feasibility lied to us. The binding constraint is the resident working set, so we bound it explicitly and size chunks against that."

### 9. Adaptive Planner v2
- Ranks candidate strategies by predicted **wall time + recovery risk**, not bytes.
- Calibrated cost model; models window collective/leader/segment cost.
- Chooses backend, chunk_bits, layout, execution, MPI mode, window, commit policy.
- **Table:** Evidence D (decisions per workload).
- **Say:** "This is the thesis's adaptive layer: it priced the window honestly and chose NOT to use it for mixing because it would be slower despite fewer bytes."

### 10. Cluster validation
- 8 × i3en.xlarge, us-east-1d, node-local /mnt/nvme, code md5-verified per node.
- 2→4 node ~1.8–2.0×; NVMe ladder n=30/32/34; planner-v2 n=30 sanity.
- Window executor n=30: states match dense-equivalent to 2.4e-8 over 1.07B amps.
- **Table:** Evidence B + A (node-local, state-match).
- **Say:** "Everything was validated on real multi-node EC2, not a laptop — and we stopped the instances after every session."

### 11. Bugs found and fixed
- `compute_norm` end-of-run OOM → streaming norm.
- Unbounded overlay (n=36) and remote cache (n=34) → bounded.
- Window leader gather OOM at n=30 → segmented gather (the cluster caught it).
- Node-local generation recovery (shared-FS assumption) → distributed scanner.
- **Say:** "The cluster found bugs single-host testing never could — the window OOM only appeared at n=30 with 8 ranks."

### 12. Results summary
- Exact, non-stabilizer, recoverable, multi-node, adaptive.
- n=36 communication_light completed after the RAM-aware fix.
- Planner v2 makes the correct per-workload choice.
- **Table:** Evidence C + D.
- **Say:** "The contribution is a clean additive stack where an adaptive planner picks the right mechanism per workload, all proven on real hardware."

### 13. Limitations (honest)
- No 45q/50q; no universal speedup; window executor not faster on i3en.
- No dense reference verification at n≥30 (norm + cross-run match instead).
- Not production-grade against all cloud failure modes; no GPU.
- **Table:** `final_claims_and_limitations.md`.
- **Say:** "I'm explicit about what this is not — every claim has a measured basis, and I don't claim what I didn't run."

### 14. Future work & contribution
- Numba/threaded kernel (runs were numpy-bound on 4 vCPU).
- Joint planner search (chunk_bits × ranks × durable policy); permutation window path.
- **Contribution:** an out-of-core, crash-recoverable, multi-node exact simulator with a *measured, wall-time-aware* adaptive planner — validated on real EC2.
- **Say:** "The frontier here is the kernel, not the I/O — and the planner is the piece that makes the whole stack usable, because no single optimization wins everywhere."
