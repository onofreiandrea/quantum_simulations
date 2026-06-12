"""Planner v2 calibrated wall-time cost model.

v1 ranked candidate strategies by recovery-aware *byte* cost.  v2 ranks them by
**predicted wall time** built from the calibrated telemetry the previous branch
measures (``cost_model.json`` / final_summary timing fields), plus an explicit
model of the MPI-window gather/apply/scatter path so the planner can tell when a
window that *reduces bytes* would still be *slower*.

The cluster lesson is encoded in the defaults: collective **scatter** is
leader-serialised (low effective bandwidth) and the leader re-applies every
window gate to the whole rank-group buffer, so a true-mixing window's predicted
wall time is dominated by ``collective_scatter`` + ``leader_compute`` +
``segment`` overhead — not by the bytes it saves.  Calibration constants are
injectable, so a machine where collectives are cheap will (correctly) flip the
decision (see ``test_recovery_aware_planner_v2``).

Pure & deterministic: no execution, no I/O.  Every predicted term is returned in
a breakdown dict so the decision report can show *why* a candidate won or lost.
"""
from __future__ import annotations

from dataclasses import dataclass

_GB = 1_000_000_000.0
_GIB = 1 << 30
_ITEMSIZE = 8  # complex64


# Default calibration — representative of the i3en.xlarge cluster this project
# runs on.  Bandwidths GB/s; rates gates/s or amplitudes/s; commit ms.  The
# numbers need not be exact: the *comparison* between candidates is what drives
# the decision, and these encode the measured ordering (pairwise fast, collective
# scatter slow + serialised, leader compute non-trivial).
V2_DEFAULT_CALIBRATION: dict = {
    "nvme_read_gbps": 2.0,
    "nvme_write_gbps": 1.0,
    "direct_extent_read_gbps": 2.5,        # no tmp+rename round trip
    "direct_extent_write_gbps": 1.5,
    "extent_materialize_gbps": 1.0,        # unpack+repack round trip
    "pairwise_mpi_gbps": 3.0,              # batched Sendrecv, parallel across ranks
    "collective_gather_gbps": 2.0,
    "collective_scatter_gbps": 0.05,       # leader-serialised → slow (measured)
    "leader_compute_amps_per_sec": 3.0e8,  # leader re-applies gates to group buffer
    "segment_latency_ms": 1.0,             # per gather/scatter collective call pair
    "local_kernel_gates_per_sec": 6.0e5,
    "nonlocal_kernel_gates_per_sec": 3.0e5,
    "commit_ms": 3.0,
    "norm_scan_gbps": 2.0,
    "numba_speedup_factor": 1.0,           # applied to kernel rates when numba used
    "planner_failure_prob": 0.0,           # expected-recompute weight (0 ⇒ off)
}


def load_v2_calibration(measured: dict | None = None) -> dict:
    """Merge a measured ``cost_model.json`` ``constants`` block over defaults.

    Only non-null measured constants override the defaults (a null+reason
    measured value keeps the default), so a partially-calibrated run still
    yields a complete, usable model.
    """
    cal = dict(V2_DEFAULT_CALIBRATION)
    if measured:
        for k, v in measured.items():
            if k.endswith("_reason"):
                continue
            if v is not None and k in cal:
                cal[k] = v
    return cal


@dataclass
class WindowInfo:
    """Analytic summary of the executable true-mixing window(s) for a circuit."""
    executable: bool = False
    n_windows: int = 0
    n_window_steps: int = 0
    n_window_gates: int = 0
    group_size: int = 1
    gather_bytes: int = 0
    scatter_bytes: int = 0
    n_segments: int = 0
    windowed_amps: int = 0          # group_size * chunk_size * n_chunks summed
    commits_saved: int = 0
    estimated_ram_gib: float = 0.0
    reason: str | None = None       # why no window (when not executable)


def analyze_window_info(steps, *, k: int, n_local_bits: int, num_ranks: int,
                        chunk_size: int, n_chunks_per_rank: int,
                        ram_budget_gib: float | None) -> WindowInfo:
    """Derive :class:`WindowInfo` from the real executable-window planner."""
    from wenbo_engine.mpi.window_executor import plan_executable_windows
    wins, rejs = plan_executable_windows(
        steps, k, n_local_bits, num_ranks, chunk_size, ram_budget_gib)
    if not wins:
        reason = (rejs[0][2] if rejs else
                  "no consecutive pure single-qubit true-mixing MPI steps")
        return WindowInfo(executable=False, reason=reason)
    cb = chunk_size * _ITEMSIZE
    gather = scatter = segs = gates = wsteps = amps = csaved = 0
    gmax = 1
    ram = 0.0
    for w in wins:
        G = w.group_size
        gmax = max(gmax, G)
        # leader receives (G-1) chunks per local chunk index, twice (gather+scatter)
        gather += (G - 1) * n_chunks_per_rank * cb
        scatter += (G - 1) * n_chunks_per_rank * cb
        seg_len = w.seg_len or chunk_size
        n_seg = (chunk_size + seg_len - 1) // max(1, seg_len)
        segs += n_seg * n_chunks_per_rank
        gates += w.n_gates
        wsteps += w.n_steps
        amps += G * chunk_size * n_chunks_per_rank
        csaved += (w.n_steps - 1)
        ram = max(ram, w.estimated_ram_gib)
    return WindowInfo(
        executable=True, n_windows=len(wins), n_window_steps=wsteps,
        n_window_gates=gates, group_size=gmax, gather_bytes=gather,
        scatter_bytes=scatter, n_segments=segs, windowed_amps=amps,
        commits_saved=csaved, estimated_ram_gib=ram)


def _safe_div(num, den):
    return (num / den) if den and den > 0 else 0.0


def predict_wall_time(*, base: dict, ctx, candidate, cal: dict,
                      window: WindowInfo, peak_ram_gib: float | None,
                      ram_feasible: bool, kernel_backend: str = "auto",
                      numba_available: bool = False) -> dict:
    """Predict a candidate's wall-time breakdown (seconds) from calibration.

    ``base`` is the v1 :func:`estimate_candidate` byte/quantity dict for the
    equivalent layout/execution candidate; ``window`` is the circuit's
    :class:`WindowInfo`; ``candidate`` carries ``mpi_window_execution``.
    """
    R = ctx.num_ranks
    C = ctx.n_chunks_per_rank
    B = ctx.chunk_bytes

    # kernel rate, optionally sped up by numba when actually used
    speedup = cal.get("numba_speedup_factor") or 1.0
    use_numba = (kernel_backend == "numba"
                 or (kernel_backend == "auto" and numba_available))
    krate_local = cal["local_kernel_gates_per_sec"] * (speedup if use_numba else 1.0)
    krate_nl = cal["nonlocal_kernel_gates_per_sec"] * (speedup if use_numba else 1.0)

    # ── I/O ──
    io = (_safe_div(base["bytes_read"], cal["nvme_read_gbps"] * _GB)
          + _safe_div(base["bytes_written"], cal["nvme_write_gbps"] * _GB))
    # extent materialize round trip (only when extents+not-direct)
    em_bytes = 0
    if candidate.storage_layout == "extents" and candidate.extent_io_mode != "direct":
        # one unpack + one pack per pass over the partition
        passes = (len(ctx.units) if candidate.execution_mode == "compute_unit"
                  else len(ctx.steps))
        em_bytes = 2 * C * B * R * passes
    extent_materialize_time = _safe_div(em_bytes, cal["extent_materialize_gbps"] * _GB)
    direct_io_time = 0.0
    if candidate.storage_layout == "extents" and candidate.extent_io_mode == "direct":
        # local passes use direct extent slices instead of chunk files
        direct_io_time = (_safe_div(base["bytes_read"],
                                    cal["direct_extent_read_gbps"] * _GB)
                          + _safe_div(base["bytes_written"],
                                      cal["direct_extent_write_gbps"] * _GB))
        io = 0.0   # direct path replaces chunk-file I/O for local passes

    # ── gate counts (cluster total) ──
    local_gates = sum(d["local"] for d in ctx.steps) * R
    rank_nl_gates = sum(d["rank_nl"] for d in ctx.steps) * R
    mpi_nl_gates = sum(d["mpi_nl"] for d in ctx.steps) * R

    window_on = (candidate.mpi_window_execution == "safe" and window.executable)
    # nonlocal kernel gates: rank-nonlocal always; mpi-nonlocal only when NOT
    # executed as a window (a window applies them on the leader instead).
    nl_gates = rank_nl_gates + (0 if window_on else mpi_nl_gates)
    kernel_time = (_safe_div(local_gates, krate_local)
                   + _safe_div(nl_gates, krate_nl))
    numba_compile_time = (cal.get("numba_compile_time", 0.0) or 0.0) if use_numba else 0.0

    # ── MPI ──
    pairwise_time = collective_time = leader_time = segment_time = 0.0
    if window_on:
        # windowed mpi gates go through gather/apply/scatter; any *non-windowed*
        # mpi bytes (none for these workloads) would still be pairwise — model 0.
        collective_time = (_safe_div(window.gather_bytes,
                                     cal["collective_gather_gbps"] * _GB)
                           + _safe_div(window.scatter_bytes,
                                       cal["collective_scatter_gbps"] * _GB))
        # the leader re-applies every window gate to the whole group buffer:
        # work = n_window_gates × (group buffer amplitudes).
        leader_time = _safe_div(window.n_window_gates * window.windowed_amps,
                                cal["leader_compute_amps_per_sec"])
        segment_time = window.n_segments * (cal["segment_latency_ms"] / 1000.0)
    else:
        pairwise_time = _safe_div(base["mpi_bytes_sent"],
                                  cal["pairwise_mpi_gbps"] * _GB)

    # ── commits ──
    commit_count = base["commit_count"]
    if window_on:
        commit_count = max(1, commit_count - window.commits_saved)
    commit_time = commit_count * (cal["commit_ms"] / 1000.0)

    # ── norm scan (one full-state read at the end) ──
    norm_bytes = C * B * R
    norm_time = _safe_div(norm_bytes, cal["norm_scan_gbps"] * _GB)

    # ── expected recomputation after a failure ──
    seg_work = _safe_div(C * B * R, cal["nvme_read_gbps"] * _GB)
    recompute_time = cal.get("planner_failure_prob", 0.0) * seg_work \
        * (window.commits_saved + 1 if window_on else 1)

    # ── RAM penalty / infeasibility ──
    ram_penalty = 0.0 if ram_feasible else 1e9   # infeasible ⇒ never selected

    total = (io + extent_materialize_time + direct_io_time + kernel_time
             + numba_compile_time + pairwise_time + collective_time
             + leader_time + segment_time + commit_time + norm_time
             + recompute_time + ram_penalty)

    return {
        "predicted_wall_time": round(total, 6),
        "predicted_io_time": round(io + extent_materialize_time + direct_io_time, 6),
        "predicted_kernel_time": round(kernel_time + numba_compile_time, 6),
        "predicted_pairwise_mpi_time": round(pairwise_time, 6),
        "predicted_collective_mpi_time": round(collective_time, 6),
        "predicted_window_leader_time": round(leader_time, 6),
        "predicted_window_segment_time": round(segment_time, 6),
        "predicted_commit_time": round(commit_time, 6),
        "predicted_norm_time": round(norm_time, 6),
        "predicted_recomputation_cost": round(recompute_time, 6),
        "predicted_peak_ram_gib": peak_ram_gib,
        "ram_feasible": bool(ram_feasible),
        "ram_penalty_applied": ram_penalty > 0,
        "window_on": window_on,
        "numba_used": use_numba,
    }
