"""Optimizer v2 — top-level plan construction, metrics, and ablation.

Builds an execution :class:`Plan` for a circuit under a chosen ablation
mode, annotates every stage with concrete data-movement / communication
quantities, prices them with :mod:`.stage_cost_model`, and produces an
ablation report comparing all modes for the SAME circuit + hardware
config.

All metrics are deterministic and computed in-process from the plan — no
cluster run is required.  The plan is also semantically equivalent to the
input circuit; :func:`replay_statevector` reproduces the logical state
(verified against ``ref_dense.simulate`` in the test-suite).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from wenbo_engine.circuit.io import validate_circuit_dict
from wenbo_engine.circuit.reorder import reorder_qubits
from wenbo_engine.kernel import gates as gmod
from wenbo_engine.kernel.ref_dense import _apply_1q, _apply_2q
from wenbo_engine.planner.placement_planner import plan_placement, apply_placement
from wenbo_engine.planner.qubit_activity import qubit_activity
from wenbo_engine.planner.stage_builder import (
    BuiltStage, PlannedOp, build_levelized_stages, build_stage_v2_stages,
    matrix_from_list,
)
from wenbo_engine.planner.stage_cost_model import (
    DEFAULT_COST_MODEL, load_cost_model, stage_cost,
)
from wenbo_engine.storage.block_store import DTYPE

ABLATION_MODES = [
    "current",
    "current_static_reorder",
    "stage_v2",
    "stage_v2_fusion",
    "stage_v2_placement_fusion",
]

_ITEMSIZE = np.dtype(DTYPE).itemsize


# ── hardware config ──────────────────────────────────────────────────

@dataclass
class HardwareConfig:
    """Hardware / partition layout the plan is priced against."""
    n_qubits: int
    chunk_bits: int
    num_ranks: int = 1
    recovery: str = "wal"  # none | wal | generation

    @property
    def p(self) -> int:
        return int(math.log2(self.num_ranks)) if self.num_ranks > 1 else 0

    @property
    def n_local_bits(self) -> int:
        return self.n_qubits - self.chunk_bits - self.p

    @property
    def chunk_bytes(self) -> int:
        return (1 << self.chunk_bits) * _ITEMSIZE

    @property
    def n_chunks_total(self) -> int:
        return 1 << (self.n_qubits - self.chunk_bits)

    @property
    def n_chunks_per_rank(self) -> int:
        return max(1, self.n_chunks_total // self.num_ranks)

    def validate(self) -> None:
        if self.num_ranks < 1 or (self.num_ranks & (self.num_ranks - 1)) != 0:
            raise ValueError(f"num_ranks must be power of 2, got {self.num_ranks}")
        if self.n_local_bits < 0:
            raise ValueError(
                f"invalid layout: n - chunk_bits - p = {self.n_local_bits} < 0 "
                f"(n={self.n_qubits}, chunk_bits={self.chunk_bits}, "
                f"num_ranks={self.num_ranks})")
        if self.recovery not in ("none", "wal", "generation"):
            raise ValueError(f"unknown recovery mode {self.recovery!r}")

    def to_dict(self) -> dict:
        return {
            "n_qubits": self.n_qubits,
            "chunk_bits": self.chunk_bits,
            "num_ranks": self.num_ranks,
            "recovery": self.recovery,
        }


# ── stage / plan ─────────────────────────────────────────────────────

@dataclass
class Stage:
    """A cost-annotated execution stage."""
    index: int
    kind: str
    ops: list[PlannedOp]
    local_ops: int
    rank_nonlocal_ops: int
    mpi_nonlocal_ops: int
    bytes_read: int
    bytes_written: int
    mpi_bytes_sent: int
    sendrecv_count: int
    commits: int
    full_state_pass: bool
    cost: dict = field(default_factory=dict)


@dataclass
class Plan:
    """An execution plan for one ablation mode + hardware config."""
    mode: str
    hardware: HardwareConfig
    perm: dict[int, int] | None
    stages: list[Stage]
    metrics: dict = field(default_factory=dict)
    # net logical->physical bit mapping, used to replay/verify (not
    # serialized into metrics; reconstructable, kept for in-process use).
    log_to_phys: list[int] | None = None

    @property
    def n_stages(self) -> int:
        return len(self.stages)


# ── op-count -> stage cost annotation ────────────────────────────────

def _annotate_stage(index: int, built: BuiltStage, hw: HardwareConfig,
                    model: dict) -> Stage:
    """Turn a :class:`BuiltStage` into a priced :class:`Stage`.

    Byte accounting matches the project's accepted estimate (see
    ``experiments.run_experiment._run_mpi`` and ``compile_plan``):

      * A compute / swap stage reads every chunk once and writes it once
        (double-buffer pass) → one full-state pass.
      * Each MPI-nonlocal op exchanges every (per-rank) chunk with a
        partner: ``mpi_bytes += n_chunks_per_rank * chunk_bytes`` and
        ``sendrecv_count += n_chunks_per_rank``.
      * One commit per stage when the recovery mode is durable
        (``wal`` / ``generation``); ``none`` performs no commit.
    """
    loc, rnl, mnl = built.class_counts()
    n_chunks = hw.n_chunks_per_rank
    chunk_bytes = hw.chunk_bytes

    bytes_read = n_chunks * chunk_bytes
    bytes_written = n_chunks * chunk_bytes
    mpi_bytes = mnl * n_chunks * chunk_bytes
    sendrecv_count = mnl * n_chunks
    commits = 1 if hw.recovery in ("wal", "generation") else 0
    n_ops = len(built.ops)

    cost = stage_cost(
        bytes_read=bytes_read, bytes_written=bytes_written,
        mpi_bytes=mpi_bytes, n_ops=n_ops, sendrecv_count=sendrecv_count,
        commits=commits, model=model,
    )

    return Stage(
        index=index, kind=built.kind, ops=built.ops,
        local_ops=loc, rank_nonlocal_ops=rnl, mpi_nonlocal_ops=mnl,
        bytes_read=bytes_read, bytes_written=bytes_written,
        mpi_bytes_sent=mpi_bytes, sendrecv_count=sendrecv_count,
        commits=commits, full_state_pass=True, cost=cost,
    )


# ── plan construction ────────────────────────────────────────────────

def _perm_to_log_to_phys(perm: dict[int, int], n: int) -> list[int]:
    """``perm`` maps old(logical) -> new(physical) bit; return list form."""
    ltp = [0] * n
    for old_q, new_q in perm.items():
        ltp[old_q] = new_q
    return ltp


def build_plan(circuit_dict: dict, hardware: HardwareConfig, mode: str,
               cost_model: dict | None = None) -> Plan:
    """Build and price a :class:`Plan` for ``mode``.

    The circuit is validated; depending on the mode it may be relabelled by
    a static reorder (``current_static_reorder``) or the activity-based
    placement (``stage_v2_placement_fusion``).  Stages are then built by
    :mod:`.stage_builder` and priced by :mod:`.stage_cost_model`.
    """
    if mode not in ABLATION_MODES:
        raise ValueError(f"unknown ablation mode {mode!r} "
                         f"(choices: {ABLATION_MODES})")
    hardware.validate()
    cd = validate_circuit_dict(circuit_dict)
    n = cd["number_of_qubits"]
    if n != hardware.n_qubits:
        raise ValueError(
            f"circuit has {n} qubits but hardware config expects "
            f"{hardware.n_qubits}")

    k = hardware.chunk_bits
    n_local_bits = hardware.n_local_bits
    model = cost_model if cost_model is not None else dict(DEFAULT_COST_MODEL)

    perm: dict[int, int] | None = None
    phys_cd = cd
    atlas_log_to_phys: list[int] | None = None

    if mode == "current":
        built = build_levelized_stages(phys_cd, k, n_local_bits)

    elif mode == "current_static_reorder":
        phys_cd, perm = reorder_qubits(cd)
        built = build_levelized_stages(phys_cd, k, n_local_bits)

    elif mode in ("stage_v2", "stage_v2_fusion"):
        fusion = (mode == "stage_v2_fusion")
        built, atlas_log_to_phys = build_stage_v2_stages(
            cd, k, n_local_bits, fusion=fusion)

    elif mode == "stage_v2_placement_fusion":
        perm = plan_placement(cd, k=k, p=hardware.p,
                              activity=qubit_activity(cd))
        phys_cd = apply_placement(cd, perm)
        built, atlas_log_to_phys = build_stage_v2_stages(
            phys_cd, k, n_local_bits, fusion=True)
    else:  # pragma: no cover - guarded above
        raise ValueError(mode)

    # Net logical -> physical mapping for replay / verification.
    if atlas_log_to_phys is not None:
        if perm is not None:
            # compose: logical q --perm--> q' --atlas--> phys
            log_to_phys = [atlas_log_to_phys[perm[q]] for q in range(n)]
        else:
            log_to_phys = list(atlas_log_to_phys)
    elif perm is not None:
        log_to_phys = _perm_to_log_to_phys(perm, n)
    else:
        log_to_phys = list(range(n))

    stages = [_annotate_stage(i, b, hardware, model)
              for i, b in enumerate(built)]

    plan = Plan(mode=mode, hardware=hardware, perm=perm, stages=stages,
                log_to_phys=log_to_phys)
    plan.metrics = plan_metrics(plan)
    return plan


# ── metrics ──────────────────────────────────────────────────────────

def plan_metrics(plan: Plan) -> dict:
    """Aggregate deterministic plan metrics for an ablation report row.

    Includes every field the task requires per mode: estimated runtime,
    bytes_read, bytes_written, mpi_bytes_sent, sendrecv_count, number of
    stages, number of commits, plus full_state_passes and the priced cost
    breakdown.  ``final_norm`` is filled in by :func:`ablation_report`
    (it needs the circuit to verify) and defaults to ``None`` here.
    """
    runtime = 0.0
    breakdown = {
        "estimated_nvme_read": 0.0,
        "estimated_nvme_write": 0.0,
        "estimated_mpi": 0.0,
        "estimated_kernel": 0.0,
        "estimated_commit": 0.0,
        "estimated_recompute_if_failure": 0.0,
    }
    bytes_read = bytes_written = mpi_bytes = sendrecv = commits = 0
    full_passes = 0
    swap_stages = 0
    for s in plan.stages:
        runtime += s.cost.get("total", 0.0)
        for key in breakdown:
            breakdown[key] += s.cost.get(key, 0.0)
        bytes_read += s.bytes_read
        bytes_written += s.bytes_written
        mpi_bytes += s.mpi_bytes_sent
        sendrecv += s.sendrecv_count
        commits += s.commits
        if s.full_state_pass:
            full_passes += 1
        if s.kind == "swap":
            swap_stages += 1

    return {
        "mode": plan.mode,
        "estimated_runtime_sec": runtime,
        "bytes_read": bytes_read,
        "bytes_written": bytes_written,
        "mpi_bytes_sent": mpi_bytes,
        "sendrecv_count": sendrecv,
        "n_stages": plan.n_stages,
        "n_steps": plan.n_stages,  # alias: one I/O pass per stage
        "n_commits": commits,
        "full_state_passes": full_passes,
        "swap_stages": swap_stages,
        "cost_breakdown": breakdown,
        "final_norm": None,
    }


# ── replay / equivalence ─────────────────────────────────────────────

def _op_matrix(op: PlannedOp) -> np.ndarray:
    if op.gate == "__matrix__":
        return matrix_from_list(op.params["matrix"])
    return gmod.gate_matrix(op.gate, op.params)


def plan_to_gates(plan: Plan) -> list[PlannedOp]:
    """Flatten a plan to its ordered list of physical-coordinate ops.

    Stage order is preserved and, within a stage, op order is preserved, so
    the result is a valid linear schedule (SWAPs included).
    """
    out: list[PlannedOp] = []
    for s in plan.stages:
        out.extend(s.ops)
    return out


def replay_statevector(plan: Plan) -> np.ndarray:
    """Replay a plan in-process and return the LOGICAL-ordered statevector.

    Applies every physical op (gates + staging SWAPs) to ``|0...0>`` in the
    physical layout, then un-permutes via the plan's net
    ``log_to_phys`` mapping so the result is directly comparable to
    ``ref_dense.simulate`` on the original circuit.
    """
    from wenbo_engine.circuit.staging import permute_state

    n = plan.hardware.n_qubits
    psi = np.zeros(1 << n, dtype=np.complex128)
    psi[0] = 1.0
    for op in plan_to_gates(plan):
        U = _op_matrix(op)
        if len(op.qubits) == 1:
            _apply_1q(psi, op.qubits[0], U)
        else:
            _apply_2q(psi, op.qubits[0], op.qubits[1], U)

    ltp = plan.log_to_phys or list(range(n))
    if ltp != list(range(n)):
        psi = permute_state(psi, ltp)
    return psi


# ── ablation report ──────────────────────────────────────────────────

def ablation_report(circuit_dict: dict, hardware: HardwareConfig,
                    cost_model: dict | None = None,
                    verify_norm: bool = True,
                    modes: list[str] | None = None) -> dict:
    """Build and price ALL ablation modes for one circuit + hardware config.

    Returns ``{"hardware": {...}, "modes": {mode: metrics, ...},
    "order": [...]}`` where each metrics dict has the required fields:
    estimated runtime, bytes_read, bytes_written, mpi_bytes_sent,
    sendrecv_count, n_stages/n_steps, n_commits, full_state_passes, and
    ``final_norm`` (the replayed state's L2 norm when ``verify_norm``).
    """
    modes = modes or ABLATION_MODES
    out_modes: dict[str, dict] = {}
    for mode in modes:
        plan = build_plan(circuit_dict, hardware, mode, cost_model=cost_model)
        metrics = dict(plan.metrics)
        if verify_norm:
            psi = replay_statevector(plan)
            metrics["final_norm"] = float(np.linalg.norm(psi))
        out_modes[mode] = metrics
    return {
        "hardware": hardware.to_dict(),
        "order": list(modes),
        "modes": out_modes,
    }


def format_ablation_table(report: dict) -> str:
    """Render an ablation report as a fixed-width text table (for logs)."""
    cols = [
        ("mode", "mode", 28),
        ("n_steps", "steps", 7),
        ("bytes_read", "bytes_read", 14),
        ("bytes_written", "bytes_writ", 14),
        ("mpi_bytes_sent", "mpi_bytes", 14),
        ("sendrecv_count", "sendrecv", 10),
        ("full_state_passes", "passes", 8),
        ("estimated_runtime_sec", "runtime_s", 12),
    ]
    lines = []
    header = "".join(f"{title:<{w}}" for _key, title, w in cols)
    lines.append(header)
    lines.append("-" * len(header))
    for mode in report["order"]:
        m = report["modes"][mode]
        row = ""
        for key, _title, w in cols:
            val = m.get(key, "")
            if key == "estimated_runtime_sec" and isinstance(val, float):
                val = f"{val:.4f}"
            row += f"{str(val):<{w}}"
        lines.append(row)
    return "\n".join(lines)
