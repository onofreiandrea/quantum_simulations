# What we can claim / what we cannot claim

This is the honesty contract for the thesis and presentation. Every "can claim"
has a measured artifact behind it on `integration/fleet` (`f5106f2`); every
"cannot claim" is something we deliberately did **not** run or prove.

## ✅ What we CAN claim

| claim | basis |
|---|---|
| **Exact state-vector simulation** | full 2^n complex64 wavefunction; window/diagonal paths verified bit-for-bit vs the dense reference at small n; norm = 1.0 |
| **Non-stabilizer workloads** | benchmarks use deterministic non-Clifford gates (RX/RY); `is_stabilizer=False` recorded in every `final_summary.json` |
| **Generation recovery with global commit records** | immutable `gen_*` + single global commit record; `source_of_truth=global_commit_record`, `wal_json_present=false` on every generation run |
| **Node-local multi-node recovery** | distributed per-rank scanner; verified on 8 real nodes (each node only its `rank_NNNN`, rank 0 holds `commits/`); crash-before/after-commit proven by fault injection |
| **Adaptive strategy selection** | Planner v2 ranks candidates by predicted wall time + recovery risk and selects the correct strategy per workload; decision + reasons emitted to `decision_report_v2.json` |
| **Real 8-node EC2 validation** | 8 × i3en.xlarge, us-east-1d; 2→4 node speedup, NVMe ladder to n=34, planner-v2 n=30 sanity, window-executor n=30 state match (2.4e-8 over 1.07B amplitudes) |
| **n=36 communication_light completed after the RAM-aware fix** | default n=36 OOM'd (est peak 211 GiB); `--auto-chunk-bits` → chunk_bits=29 (est 19.8 GiB ≤ 21) → completed, committed gen 1 in 2853 s, final_norm≈1 |
| **Planner v2 chooses based on wall-time tradeoffs** | mixing_heavy: window rejected because predicted slower despite ~11× fewer bytes; a synthetic cheap-collective calibration flips the choice — proving it is cost-driven |

## ❌ What we CANNOT claim

| not claimed | why |
|---|---|
| **45-qubit or 50-qubit run** | never executed; n=38 projected > 2 h stage cap and would OOM. Largest completed: n=36 communication_light (RAM-aware), n=34 ladder (comm_light), n=30/32 across MPI workloads |
| **Universal speedup for every workload** | optimizations are workload-dependent; the whole point of Planner v2 is that no single recipe wins everywhere |
| **MPI window executor is faster on i3en** | it reduces communication ~11× but is **slower** in wall time on this hardware (collective scatter + leader compute dominate); v2 correctly leaves it off |
| **Dense-reference verification at n ≥ 30** | `ref_dense` is infeasible at 30q on a 30 GiB node; correctness at scale is shown via norm≈1 + bit-close cross-run state match (window vs per-step), not a dense oracle |
| **Production-grade fault tolerance against all cloud failures** | we prove crash/resume via deterministic injection and node-local recovery; we do **not** claim coverage of byzantine faults, network partitions, silent disk corruption beyond checksums, or coordinator loss |
| **GPU acceleration** | CPU only (numpy, optional numba); no GPU kernels |

## Calibration caveat (state it if asked)
Planner v2's predicted wall-time **magnitudes** carry error (i3en-default
calibration constants, not per-run-measured); the **ordering** of candidates —
which drives the decision — is correct, and the cost report shows the prediction
error honestly rather than hiding it.
