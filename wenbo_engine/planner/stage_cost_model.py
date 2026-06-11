"""Stage cost model.

Estimates the wall-clock cost of one execution *stage* from concrete plan
quantities (chunk bytes, number of chunks touched, number of sendrecv
exchanges, number of commits) and a hardware ``cost_model``.

    stage_cost = estimated_nvme_read
               + estimated_nvme_write
               + estimated_mpi
               + estimated_kernel
               + estimated_commit
               + estimated_recompute_if_failure

The hardware ``cost_model`` reuses the flat keys produced by
:class:`wenbo_engine.profiling.calibration.CalibrationRunner`
(``nvme_read_gbps``, ``nvme_write_gbps``, ``fsync_ms``, ``rename_ms``,
``mpi_sendrecv_gbps``, ...).  When a calibration file is unavailable the
:data:`DEFAULT_COST_MODEL` (sane i3en.xlarge-class defaults) is used, so
the planner works fully offline — the proof in the test-suite never needs
a live machine.

Units: bandwidths in GB/s (1 GB = 1e9 bytes, matching the calibration's
MBps/1000 conversion), latencies in milliseconds, returned costs in
seconds.
"""
from __future__ import annotations

import json
from pathlib import Path

# Defaults chosen to be representative of an i3en.xlarge-class node
# (NVMe SSD + 25 Gbps EFA network).  Conservative round numbers — the
# ablation *comparison* is what matters, and all modes share the model.
DEFAULT_COST_MODEL: dict = {
    "nvme_read_gbps": 2.0,       # GB/s sequential chunk read
    "nvme_write_gbps": 1.0,      # GB/s atomic write (tmp+rename+fsync)
    "fsync_ms": 1.0,             # per commit fsync
    "rename_ms": 0.05,           # per atomic rename
    "mpi_sendrecv_gbps": 3.0,    # GB/s effective Sendrecv bandwidth
    "mpi_barrier_ms": 0.1,       # per-step barrier
    # kernel throughput: complex multiply-add rate per local chunk pass,
    # expressed as effective GB/s over the chunk bytes processed.
    "kernel_gbps": 8.0,
    # probability a stage must be recomputed due to a failure (drives the
    # recompute-if-failure term).  0 by default → that term is 0.
    "failure_prob": 0.0,
    # ── recovery-aware planner v1 extensions ────────────────────────────
    # Durable checkpoint write bandwidth (remote object store / NFS), much
    # slower than local NVMe.  Drives durable_checkpoint_cost.
    "durable_write_gbps": 0.5,
    # Per-chunk seek+read overhead of the direct extent-slice I/O path
    # (seek into a packed extent file instead of opening a chunk file).
    "direct_extent_seek_ms": 0.01,
    # Per-generation failure probability used ONLY by the recovery-aware
    # expected_recomputation_cost term (kept separate from ``failure_prob``
    # so optimizer-v2's stage_cost is unaffected).  Small but non-zero so the
    # term is represented; smaller committed generations ⇒ less work at risk.
    "planner_failure_prob": 0.01,
    # Per-Sendrecv call latency (recovery-aware planner only).  Lets the cost
    # model reward gate_aware's call-count reduction even when the bytes moved
    # are unchanged — fewer blocking Sendrecv round trips is cheaper.
    "mpi_sendrecv_latency_ms": 0.05,
}

# Flat cost-model keys we read; everything else in a calibration file is
# ignored (forward-compatible with richer cost models).
_REQUIRED_KEYS = (
    "nvme_read_gbps", "nvme_write_gbps", "fsync_ms", "rename_ms",
    "mpi_sendrecv_gbps", "mpi_barrier_ms", "kernel_gbps", "failure_prob",
    "durable_write_gbps", "direct_extent_seek_ms", "planner_failure_prob",
    "mpi_sendrecv_latency_ms",
)

_GB = 1e9
_MS = 1e-3


def load_cost_model(path: str | Path | None = None) -> dict:
    """Return a complete cost model, merged over :data:`DEFAULT_COST_MODEL`.

    ``path`` may point at a ``cost_model.json`` written by the calibration
    runner (its flat top-level keys are read).  Missing or unmeasured keys
    fall back to the defaults, so the result always has every required key
    and the planner never crashes on a partial calibration file.
    """
    model = dict(DEFAULT_COST_MODEL)
    if path is None:
        return model
    p = Path(path)
    if not p.exists():
        return model
    try:
        raw = json.loads(p.read_text())
    except (OSError, ValueError):
        return model
    for key in _REQUIRED_KEYS:
        val = raw.get(key)
        if isinstance(val, (int, float)) and val > 0:
            model[key] = float(val)
    return model


def _read_sec(nbytes: int, model: dict) -> float:
    return nbytes / (model["nvme_read_gbps"] * _GB) if nbytes else 0.0


def _write_sec(nbytes: int, model: dict) -> float:
    return nbytes / (model["nvme_write_gbps"] * _GB) if nbytes else 0.0


def _mpi_sec(nbytes: int, model: dict) -> float:
    return nbytes / (model["mpi_sendrecv_gbps"] * _GB) if nbytes else 0.0


def _kernel_sec(nbytes: int, n_ops: int, model: dict) -> float:
    # Kernel work scales with both bytes touched and number of gate ops
    # applied to them (each op is a pass over the chunk amplitudes).
    if nbytes <= 0 or n_ops <= 0:
        return 0.0
    return (nbytes * n_ops) / (model["kernel_gbps"] * _GB)


def stage_cost(*, bytes_read: int, bytes_written: int, mpi_bytes: int,
               n_ops: int, sendrecv_count: int, commits: int,
               model: dict) -> dict:
    """Cost breakdown (seconds) for one stage's concrete quantities.

    Returns a dict with the six additive terms plus ``total``.  The
    ``recompute_if_failure`` term is the failure probability times the
    cost of redoing the stage's read + kernel + write work (commit excluded
    — a failed stage never committed).
    """
    read = _read_sec(bytes_read, model)
    write = _write_sec(bytes_written, model)
    # MPI cost is driven by bytes exchanged; a per-stage barrier is added
    # whenever any MPI traffic occurs (sendrecv_count is tracked separately
    # for reporting and reflected in the byte total).
    mpi = _mpi_sec(mpi_bytes, model)
    if mpi_bytes or sendrecv_count:
        mpi += model["mpi_barrier_ms"] * _MS
    kernel = _kernel_sec(bytes_read, n_ops, model)
    commit = commits * (model["fsync_ms"] + model["rename_ms"]) * _MS
    redo_work = read + kernel + write
    recompute = model["failure_prob"] * redo_work

    total = read + write + mpi + kernel + commit + recompute
    return {
        "estimated_nvme_read": read,
        "estimated_nvme_write": write,
        "estimated_mpi": mpi,
        "estimated_kernel": kernel,
        "estimated_commit": commit,
        "estimated_recompute_if_failure": recompute,
        "total": total,
    }


# ── recovery-aware cost model (planner v1) ──────────────────────────────

def _nvme_rw_avg_gbps(model: dict) -> float:
    return (model["nvme_read_gbps"] + model["nvme_write_gbps"]) / 2.0


def recovery_aware_cost(*, bytes_read: int, bytes_written: int,
                        mpi_bytes: int, sendrecv_count: int,
                        kernel_bytes: int, commits: int,
                        extent_materialize_bytes: int = 0,
                        direct_extent_chunks: int = 0,
                        layout_materialize_bytes: int = 0,
                        durable_bytes: int = 0,
                        recompute_bytes: int = 0,
                        model: dict) -> dict:
    """Recovery-aware cost breakdown (seconds) with the planner-v1 terms.

    ``total_cost = nvme_read_write_cost + mpi_exchange_cost + kernel_cost
                 + extent_materialization_cost + direct_extent_io_cost
                 + layout_materialization_cost + commit_cost
                 + durable_checkpoint_cost + expected_recomputation_cost``

    The terms are mutually exclusive physical contributions:

    * ``nvme_read_write_cost`` — the unavoidable logical state read+write
      passes (fewer for compute-unit fusion).
    * ``extent_materialization_cost`` — the EXTRA temp-chunk round trip the
      materialize extent-I/O path pays (unpack extents→chunks, repack
      chunks→extents).  Zero for chunks layout and for direct extent I/O.
    * ``direct_extent_io_cost`` — the small seek overhead of the direct
      extent-slice path (replaces the materialize round trip).  Zero unless
      direct extent I/O is used.
    * ``layout_materialization_cost`` — one-time cost of converting the input
      state into the strategy's storage layout before stage 0 (zero for a
      fresh run written in-layout; counted so rule 4 is never free).
    * ``durable_checkpoint_cost`` — promoting committed generations to durable
      storage (zero unless a durable policy is active).
    * ``expected_recomputation_cost`` — failure-probability-weighted work at
      risk between commits (smaller/more-frequent commits ⇒ less at risk).
    """
    rd = _read_sec(bytes_read, model)
    wr = _write_sec(bytes_written, model)
    nvme_rw = rd + wr
    mpi = _mpi_sec(mpi_bytes, model)
    if mpi_bytes or sendrecv_count:
        mpi += model["mpi_barrier_ms"] * _MS
    # per-call Sendrecv latency: rewards gate_aware's batched (fewer) calls
    mpi += sendrecv_count * model.get("mpi_sendrecv_latency_ms", 0.0) * _MS
    kernel = kernel_bytes / (model["kernel_gbps"] * _GB) if kernel_bytes else 0.0

    avg = _nvme_rw_avg_gbps(model)
    extent_mat = (extent_materialize_bytes / (avg * _GB)
                  if extent_materialize_bytes else 0.0)
    direct_io = direct_extent_chunks * model["direct_extent_seek_ms"] * _MS
    layout_mat = (layout_materialize_bytes / (avg * _GB)
                  if layout_materialize_bytes else 0.0)
    commit = commits * (model["fsync_ms"] + model["rename_ms"]) * _MS
    durable = (durable_bytes / (model["durable_write_gbps"] * _GB)
               if durable_bytes else 0.0)
    recompute = model["planner_failure_prob"] * (
        recompute_bytes / (avg * _GB) if recompute_bytes else 0.0)

    total = (nvme_rw + mpi + kernel + extent_mat + direct_io + layout_mat
             + commit + durable + recompute)
    return {
        "nvme_read_write_cost": nvme_rw,
        "mpi_exchange_cost": mpi,
        "kernel_cost": kernel,
        "extent_materialization_cost": extent_mat,
        "direct_extent_io_cost": direct_io,
        "layout_materialization_cost": layout_mat,
        "commit_cost": commit,
        "durable_checkpoint_cost": durable,
        "expected_recomputation_cost": recompute,
        "estimated_total_cost": total,
    }
