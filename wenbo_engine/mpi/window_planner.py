"""MPI-exchange *window* feasibility planner (analysis only — no execution).

Given a circuit + layout, this enumerates **candidate windows**: maximal runs
of consecutive compiled steps that each carry at least one MPI-nonlocal gate
requiring remote amplitudes (true-mixing or permutation).  For each candidate
it predicts — *separately* — the MPI bytes, Sendrecv calls and commits a
gather/apply/scatter window would save, the extra RAM it would need, and the
extra recomputation a crash would cost.  It then decides, conservatively,
whether such a window would be **safe to execute in the future**.

It does **not** execute anything, does **not** keep any remote cache across
steps, does **not** commit, and does **not** touch recovery.  Candidate
generation is **deterministic** (it walks the real runner's compiled steps in
order and uses pure bit arithmetic).

Rules enforced here (mirrored from the task spec):

1. A window is feasible only if ``estimated_ram_gib <= ram_budget_gib``.
2. Diagonal MPI gates already skipped by the diagonal fast path are **not**
   counted as window work (they need no exchange); they are reported as
   ``diagonal_gates_in_region`` for context only.
3. true-mixing gates are reported separately from permutation / diagonal.
4. A window is not marked executable if it would require stale-cache reuse —
   modelled here as: any non-batchable (``fallback``) MPI gate in the run, for
   which a correct single gather/scatter is not yet defined.
5. Candidate generation is deterministic.
6. Nothing here changes execution.
"""
from __future__ import annotations

from wenbo_engine.kernel import gates as _gates
from wenbo_engine.mpi.diagonal_nonlocal import classify_nonlocal_gate
from wenbo_engine.mpi.exchange_planner import classify_gate as _resolve_exchange
from wenbo_engine.mpi import window_cost_model as cm


def _layout(n: int, chunk_bits: int, num_ranks: int) -> tuple[int, int, int]:
    """(k, p, n_local_bits) — same convention as the runner / bench."""
    if num_ranks < 1 or (num_ranks & (num_ranks - 1)) != 0:
        raise ValueError(f"num_ranks must be a power of two, got {num_ranks}")
    k = chunk_bits
    p = num_ranks.bit_length() - 1
    n_local_bits = n - k - p
    if n_local_bits < 0:
        raise ValueError(
            f"invalid layout: n-chunk_bits-log2(ranks)={n_local_bits} < 0")
    return k, p, n_local_bits


def _compiled_steps(circuit_dict: dict, k: int, n_local_bits: int):
    """The REAL runner compiler's steps (deterministic), reused not re-derived."""
    from wenbo_engine.circuit.io import validate_circuit_dict, levelize
    from wenbo_engine.mpi.mpi_runner import _compile_steps
    cd = validate_circuit_dict(circuit_dict)
    return _compile_steps(levelize(cd), k, n_local_bits)


def _step_remote_profile(mpi_ops, rank: int, k: int, n_local_bits: int,
                         n_chunks_per_rank: int) -> dict:
    """Classify one step's MPI gates and resolve the remote chunks they fetch.

    Returns counts (diagonal / true_mixing / permutation / fallback) plus, for
    the *remote-requiring* gates only, the set of ``(partner_rank, remote_ci)``
    keys fetched and the partner-rank set.  Modelled on a single ``rank`` —
    partitions are homogeneous, so rank 0 is representative.
    """
    diag = mixing = perm = fallback = 0
    remote_keys: set[tuple[int, int]] = set()
    partners: set[int] = set()
    has_fallback = False
    for qs, U in mpi_ops:
        kind, requires_remote = classify_nonlocal_gate(U)
        if kind == "diagonal":
            diag += 1
            continue                       # rule 2: skipped by fast path
        if kind == "permutation":
            perm += 1
        else:
            mixing += 1
        # remote-requiring gate: resolve its concrete exchange
        ge = _resolve_exchange(qs, U, rank, k, n_local_bits, n_chunks_per_rank)
        if not ge.batchable:
            fallback += 1
            has_fallback = True
            # still fetches remotely; model all chunks against its partner
            partner = ge.partner_rank if ge.partner_rank >= 0 else rank ^ 1
            for ci in range(n_chunks_per_rank):
                remote_keys.add((partner, ci))
            partners.add(partner)
            continue
        partners.add(ge.partner_rank)
        for (_lci, rci) in ge.chunk_pairs:
            remote_keys.add((ge.partner_rank, rci))
    return {
        "diagonal": diag, "true_mixing": mixing, "permutation": perm,
        "fallback": fallback, "has_fallback": has_fallback,
        "remote_keys": remote_keys, "partners": partners,
        "n_remote_requiring": mixing + perm,
    }


def analyze_windows(circuit_dict: dict, chunk_bits: int, num_ranks: int,
                    ram_budget_gib: float | None = None,
                    weights: cm.CostWeights | None = None,
                    rank: int = 0, min_window_steps: int = 2) -> dict:
    """Enumerate candidate MPI windows and predict their benefit/cost.

    Pure & deterministic.  Returns ``{"candidates": [...], "layout": {...},
    "summary": {...}}``.  ``candidates`` is a list of dicts with every field the
    task requires.  ``ram_budget_gib`` gates feasibility (rule 1).
    """
    weights = weights or cm.CostWeights()
    n = circuit_dict["number_of_qubits"]
    k, p, n_local_bits = _layout(n, chunk_bits, num_ranks)
    n_chunks_per_rank = (1 << (n - k)) // num_ranks
    cbytes = cm.chunk_bytes(k)

    steps = _compiled_steps(circuit_dict, k, n_local_bits)
    profiles = [
        _step_remote_profile(s["mpi_nonlocal_ops"], rank, k, n_local_bits,
                             n_chunks_per_rank)
        for s in steps
    ]

    # Maximal runs of consecutive steps that each require remote amplitudes.
    runs: list[tuple[int, int]] = []
    start = None
    for i, pr in enumerate(profiles):
        if pr["n_remote_requiring"] > 0:
            if start is None:
                start = i
        else:
            if start is not None:
                runs.append((start, i - 1))
                start = None
    if start is not None:
        runs.append((start, len(profiles) - 1))

    candidates = []
    wid = 0
    total_diag_in_mpi_steps = sum(pr["diagonal"] for pr in profiles
                                  if pr["n_remote_requiring"] > 0)
    for (s0, s1) in runs:
        n_steps = s1 - s0 + 1
        if n_steps < min_window_steps:
            continue                       # single step: nothing to fuse
        wprofiles = profiles[s0:s1 + 1]

        remote_keys: set[tuple[int, int]] = set()
        partners: set[int] = set()
        baseline_fetches = 0
        baseline_sendrecv = 0
        gates = 0
        mixing = perm = diag_region = fallback = 0
        for pr in wprofiles:
            remote_keys |= pr["remote_keys"]
            partners |= pr["partners"]
            baseline_fetches += len(pr["remote_keys"])
            baseline_sendrecv += len(pr["partners"])   # 1 fetch-group / partner
            gates += pr["n_remote_requiring"]
            mixing += pr["true_mixing"]
            perm += pr["permutation"]
            diag_region += pr["diagonal"]
            fallback += pr["fallback"]

        distinct_remote = len(remote_keys)
        local_chunks = n_chunks_per_rank          # local partition the rank owns
        # window communication
        gather_bytes = cm.bytes_for(distinct_remote, cbytes)
        scatter_bytes = cm.bytes_for(distinct_remote, cbytes)
        window_mpi_bytes = gather_bytes + scatter_bytes
        baseline_mpi_bytes = cm.bytes_for(baseline_fetches, cbytes)
        window_sendrecv = 2 * len(partners)        # one gather + one scatter / partner
        sr_reduction = cm.sendrecv_reduction(baseline_sendrecv, window_sendrecv)
        avoided = cm.repeated_fetches_avoided(baseline_fetches, distinct_remote)

        # RAM: remote region + matching local chunks co-resident, +staging.
        co_resident = 2 * distinct_remote
        est_ram = cm.estimate_window_ram_gib(co_resident, cbytes,
                                             overhead_factor=1.5)
        feasible = cm.ram_feasible(est_ram, ram_budget_gib)

        # commits + recovery risk
        commit_baseline = n_steps                  # per-step commit cadence
        commit_window = 1                          # one window commit boundary
        recompute_inc = cm.recomputation_cost_increase(
            gates, commit_baseline, commit_window)

        # safety (rules 1 & 4)
        rejection = None
        if fallback > 0:
            rejection = (f"contains {fallback} non-batchable (fallback) MPI "
                         "gate(s); correct single gather/scatter not yet "
                         "modelled — would risk stale-cache reuse")
        elif not feasible:
            if ram_budget_gib is None:
                rejection = "ram_budget_gib not provided; feasibility unknown"
            else:
                rejection = (f"estimated_ram_gib={est_ram:.4f} exceeds "
                             f"ram_budget_gib={ram_budget_gib}")
        safe = rejection is None and gates > 0

        recovery_note = (
            "window introduces a single commit boundary; a crash inside the "
            f"window replays up to {gates} gate-applications (vs up to "
            f"{gates // max(commit_baseline,1)} per per-step commit). "
            "Gather/apply/scatter re-reads source amplitudes, so no stale "
            "remote cache is reused.")

        candidates.append({
            "window_id": wid,
            "start_step": s0,
            "end_step": s1,
            "mpi_steps_in_window": n_steps,
            "gates_in_window": gates,
            "true_mixing_gates": mixing,
            "permutation_gates": perm,
            "diagonal_gates_in_region": diag_region,
            "partner_ranks": sorted(partners),
            "local_chunks": local_chunks,
            "remote_chunks": baseline_fetches,
            "distinct_remote_chunks": distinct_remote,
            "repeated_remote_fetches_avoided": avoided,
            "estimated_gather_bytes": gather_bytes,
            "estimated_scatter_bytes": scatter_bytes,
            "estimated_baseline_mpi_bytes": baseline_mpi_bytes,
            "estimated_window_mpi_bytes": window_mpi_bytes,
            "estimated_baseline_sendrecv": baseline_sendrecv,
            "estimated_window_sendrecv": window_sendrecv,
            "estimated_sendrecv_reduction": sr_reduction,
            "estimated_mpi_byte_reduction": baseline_mpi_bytes - window_mpi_bytes,
            "estimated_ram_gib": est_ram,
            "ram_budget_gib": ram_budget_gib,
            "ram_feasible": feasible,
            "commit_count_baseline": commit_baseline,
            "commit_count_window": commit_window,
            "commit_reduction": commit_baseline - commit_window,
            "expected_recomputation_cost_increase": recompute_inc,
            "recovery_risk_note": recovery_note,
            "safe_to_execute_future": safe,
            "rejection_reason": rejection,
            "blended_baseline_cost": cm.baseline_cost(
                mpi_bytes=baseline_mpi_bytes, sendrecv_count=baseline_sendrecv,
                commit_count=commit_baseline,
                recompute_units=cm.expected_recomputation_units(
                    gates, commit_baseline),
                weights=weights),
            "blended_window_cost": cm.window_cost(
                gather_bytes=gather_bytes, scatter_bytes=scatter_bytes,
                sendrecv_count=window_sendrecv, commit_count=commit_window,
                recompute_units=cm.expected_recomputation_units(
                    gates, commit_window),
                extra_ram_gib=est_ram, weights=weights),
        })
        wid += 1

    # Full-run baseline over EVERY remote-requiring MPI step (not just the ones
    # that fall inside a multi-step window) — this is what the per-step path
    # actually exchanges across the whole run, modelled on one rank.  Its
    # cluster projection (× num_ranks) equals the measured run telemetry.
    full_fetches = sum(len(pr["remote_keys"]) for pr in profiles)
    full_sendrecv = sum(len(pr["partners"]) for pr in profiles)
    full_mpi_steps = sum(1 for pr in profiles if pr["n_remote_requiring"] > 0)
    full_baseline = {
        "sendrecv_per_rank": full_sendrecv,
        "fetches_per_rank": full_fetches,
        "mpi_bytes_per_rank": cm.bytes_for(full_fetches, cbytes),
        "sendrecv_cluster": full_sendrecv * num_ranks,
        "mpi_bytes_cluster": cm.bytes_for(full_fetches, cbytes) * num_ranks,
        "mpi_steps": full_mpi_steps,
    }

    summary = _summarize(candidates, total_diag_in_mpi_steps,
                         len(steps), cbytes, weights)
    return {
        "candidates": candidates,
        "full_baseline": full_baseline,
        "layout": {
            "n": n, "chunk_bits": k, "num_ranks": num_ranks,
            "n_local_bits": n_local_bits, "n_chunks_per_rank": n_chunks_per_rank,
            "chunk_bytes": cbytes, "n_steps": len(steps),
            "modelled_rank": rank,
        },
        "summary": summary,
    }


def _summarize(candidates: list[dict], diag_in_mpi_steps: int, n_steps: int,
               cbytes: int, weights: cm.CostWeights) -> dict:
    """Aggregate the *separated* tradeoffs across all candidate windows."""
    feasible = [c for c in candidates if c["safe_to_execute_future"]]
    best = None
    if feasible:
        best = max(feasible, key=lambda c: c["estimated_mpi_byte_reduction"])
    return {
        "num_candidate_windows": len(candidates),
        "num_feasible_windows": len(feasible),
        "diagonal_gates_in_mpi_steps": diag_in_mpi_steps,
        # separated, additive across feasible candidates
        "total_estimated_mpi_byte_reduction":
            sum(c["estimated_mpi_byte_reduction"] for c in feasible),
        "total_estimated_sendrecv_reduction":
            sum(c["estimated_sendrecv_reduction"] for c in feasible),
        "total_repeated_remote_fetches_avoided":
            sum(c["repeated_remote_fetches_avoided"] for c in feasible),
        "total_commit_reduction":
            sum(c["commit_reduction"] for c in feasible),
        "max_extra_ram_gib":
            max((c["estimated_ram_gib"] for c in feasible), default=0.0),
        "total_expected_recomputation_cost_increase":
            sum(c["expected_recomputation_cost_increase"] for c in feasible),
        "best_window_id": best["window_id"] if best else None,
        "cost_weights": cm.weights_to_dict(weights),
    }
