# Lessons learned from real cluster validation

Every lesson below was forced by a real 8-node EC2 run, not a thought
experiment. They are ordered to tell the story the presentation should tell.

## 1. Storage feasibility is not enough — RAM working set can be the real limit
i3en.xlarge gives 2.3 TiB of NVMe but only **30 GiB of RAM** per node. The NVMe
ladder stayed >93% free the whole way, yet runs were killed by the OOM killer at
n=34 (mpi) and n=36 (comm_light). The binding constraint is the **per-rank
working set** (resident chunks + kernel temporaries + MPI/overlay buffers), not
disk. We added a `ram_feasible` check that is *separate* from
`storage_feasible`, because they fail independently.

## 2. Reducing chunk_bits alone is insufficient unless overlays and caches are bounded
The intuitive fix for an OOM — "use smaller chunks" — does **not** fix a
compute-unit OOM. The overlay held *all* of a unit's chunks resident, and the
gate-aware remote-buffer cache grew without bound; both scale with the number of
chunks, so smaller chunks just means more of them. The real fix was to **bound**
the overlay (`max_overlay_chunks`) and the remote cache (LRU `max_bytes`), then
size `chunk_bits` against the bounded peak. Also: the leader gather in the MPI
window OOM'd at n=30 until we **segmented** it — same lesson, different buffer.

## 3. Byte reduction does not necessarily mean wall-time speedup
The MPI window executor cut total communication volume ~11.4× (171.8 GB → 15 GB)
and took pairwise Sendrecv to zero — and was **slower** on wall time, because the
leader-serialised collective scatter + the leader re-applying every gate to the
whole group buffer + per-segment latency dominated. This single result is *why
Planner v2 exists*: it must optimize predicted **wall time**, never bytes alone.

## 4. Diagonal MPI-nonlocal gates do not need remote amplitude exchange
A diagonal gate on a rank/MPI qubit only multiplies each amplitude by a phase
determined by its own global basis index — the partner amplitude is never read.
So these gates can be applied **locally**, with no Sendrecv. On the phase-heavy
benchmark this took MPI traffic to exactly zero while staying bit-for-bit exact
against the dense reference. Classify gates (diagonal / permutation / true-mixing)
before deciding whether to exchange.

## 5. Recovery boundaries are performance and correctness boundaries
A "generation" is both the unit of crash recovery (an immutable `gen_*` + a
single global commit record) **and** the unit of work between commits. Fusing
work (compute units, MPI windows) changes the commit cadence, which changes both
the I/O cost and the expected recomputation after a crash. Every executor we
added had to commit through the *same* protocol, and the cost model has to price
the recompute-after-failure term — you cannot reason about throughput and
durability separately.

## 6. Planner v2 is needed because no single optimization is universally best
Across five workloads the best strategy differs: communication_light wants
extents + compute_unit + direct; phase-heavy wants the diagonal fast path with no
window; mixing-heavy wants per-step gate-aware and must **reject** the window;
mixed_staged must avoid tiny compute units. No fixed recipe wins everywhere, so
the system needs an adaptive layer that prices the candidates and chooses.

## 7. Benchmarks must be non-stabilizer and must distinguish phase-heavy from true-mixing MPI stress
A Clifford-only circuit is classically simulable and would make every result
meaningless, so the workloads were made deterministically **non-stabilizer**
(RX/RY with irrational-ish angles). Just as important: "MPI stress" is not one
thing — **diagonal/phase** MPI gates need no exchange (the fast path zeroes them),
while **true-mixing** gates genuinely must move amplitudes. Telemetry that
reported only "MPI bytes" would have hidden this; we report diagonal vs
true-mixing vs permutation counts so a phase-heavy run can never be mistaken for
real communication stress.

## 8. Always use a unique work_dir for correctness experiments
A shared/default work_dir caused a **false `correct=False`**: generation
recovery correctly found a prior completed run and skipped compute, leaving the
wrong circuit's state. The reverse risk (false `correct=True` from a stale
matching state) is just as dangerous. Every benchmark and test now uses a unique
`work_dir` per run; the instance-store is also wiped on stop/start, which is a
related reminder that node-local state is ephemeral.

---

### Operational lessons (cluster mechanics)
- **Instance-store NVMe is wiped on stop/start**, and the device name is not
  stable across reboots — detect the instance-store disk dynamically (largest
  unmounted, no-partition disk), never hardcode `/dev/nvme1n1`.
- **Private IPs persist in-VPC; public IPs rotate** on stop/start — regenerate
  the hostfile from private IPs each session.
- **`git archive HEAD` + md5 on every node** is a clean, verifiable code sync;
  always confirm the same commit hash on all 8 nodes before running.
- **Always stop instances after a session** and verify 0 running — every session
  in this project did, keeping total spend in the low tens of dollars.
