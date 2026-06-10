"""Hardware-bound capacity planner.

Sizes the largest exact state-vector simulation feasible on a given cluster,
or checks the feasibility of one explicit qubit count.  Pure arithmetic — no
I/O, no MPI, no kernels.

Storage model (per rank, in units of one rank's state slice ``S``)
------------------------------------------------------------------
The storage requirement is the *peak* number of full per-rank state slices on
local NVMe during a step, expressed as a sum of explicit, separately-reported
components.  Every component is parameterized — nothing is hardcoded to a
"3S" generation cost:

  * **hot source**            1 * S   — the committed generation a step reads.
  * **destination / temp**    (needs_destination ? 1 : 0) * S
                                      + max_temp_storage_fraction * S
                              — the buffer a step writes, plus atomic
                                tmp+fsync+rename headroom.
  * **retained local recovery** max(committed_generations_retained - 1, 0) * S
                              — rollback targets kept beyond the hot source.
  * **durable checkpoint**    durable_snapshots_retained * state_size
                              — full-state snapshots; on a separate durable
                                store if one is supplied, otherwise this rank's
                                slice folds onto local NVMe.

Defaults per recovery mode are aligned with the generation-recovery package
(:mod:`wenbo_engine.recovery`), whose ``prune`` keeps **3** committed
generations by default (the current one plus two rollback targets):

  * ``none`` / ``wal``        committed=1, destination=yes, durable=0  -> 2 * S
  * ``generation``            committed=3, destination=yes, durable=0  -> 4 * S
  * ``generation+durable``    committed=3, destination=yes, durable=1
                              -> 4 * S local + one full snapshot durable

All defaults are overridable per field, so a different retention policy can be
sized without code changes.

On top of the working set, ``reserved_storage_fraction`` of the *raw* device is
treated as unusable (filesystem reserve / safety margin).

    usable_per_rank = storage_per_rank * (1 - reserved_storage_fraction)
    local_feasible  = total_local_required_per_rank <= usable_per_rank

RAM / MPI model
---------------
The hot path streams chunks; the full state is never resident.  RAM only has
to hold a streaming pipeline plus MPI send/recv buffers, so it is not the
2^n wall that storage is.  We report the RAM buffer budget and the MPI buffer
budget, and flag RAM infeasible only if it cannot hold a minimal pipeline.

State size is exact and precision-driven:

    complex64  : 2^num_qubits * 8  bytes
    complex128 : 2^num_qubits * 16 bytes

Ranks
-----
The MPI runner requires a power-of-two rank count; the planner rejects
non-power-of-two ranks by default (override with ``allow_non_power_of_two``).
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field, replace

# ── units ────────────────────────────────────────────────────────────────
TIB = 1 << 40
GIB = 1 << 30

# ── precision → bytes per complex amplitude ──────────────────────────────
BYTES_PER_AMP = {
    "complex64": 8,
    "complex128": 16,
}

# ── recovery modes ───────────────────────────────────────────────────────
RECOVERY_MODES = ("none", "wal", "generation", "generation+durable")

# Per-mode retention defaults.  Aligned with wenbo_engine.recovery: the
# generation manager's prune() keeps 3 committed generations by default.
RECOVERY_DEFAULTS = {
    "none":               {"committed": 1, "destination": True,  "durable": 0},
    "wal":                {"committed": 1, "destination": True,  "durable": 0},
    "generation":         {"committed": 3, "destination": True,  "durable": 0},
    "generation+durable": {"committed": 3, "destination": True,  "durable": 1},
}

# RAM held back for the OS + numba runtime before any streaming buffers.
RAM_OS_RESERVE_GIB = 8.0
# Fraction of the remaining RAM budget reservable for MPI send/recv buffers.
MPI_BUFFER_FRACTION = 0.25
# A streaming pipeline needs at least this much RAM to make progress.
MIN_PIPELINE_RAM_GIB = 1.0

# Default ceiling for the feasibility search.  2^64 amplitudes is far beyond
# any storage on Earth, so the search always terminates well below this.
DEFAULT_MAX_CANDIDATE_QUBITS = 64


def _is_power_of_two(x: int) -> bool:
    return x >= 1 and (x & (x - 1)) == 0


# ── inputs ────────────────────────────────────────────────────────────────
@dataclass
class PlannerConfig:
    """Hardware + policy inputs for a capacity plan.

    The three retention knobs default to ``None`` and are resolved from
    ``recovery_mode`` via :data:`RECOVERY_DEFAULTS`; set them explicitly to
    model a different retention policy.
    """

    precision: str = "complex64"
    num_ranks: int = 1
    local_storage_per_rank_tib: float = 0.0
    ram_per_rank_gib: float = 0.0
    reserved_storage_fraction: float = 0.15
    max_temp_storage_fraction: float = 0.10
    recovery_mode: str = "wal"
    durable_storage_available_tib: float | None = None
    max_candidate_qubits: int | None = None

    # retention policy (None -> resolved from recovery_mode)
    committed_generations_retained: int | None = None
    needs_destination_generation: bool | None = None
    durable_snapshots_retained: int | None = None

    allow_non_power_of_two: bool = False

    def __post_init__(self) -> None:
        if self.precision not in BYTES_PER_AMP:
            raise ValueError(
                f"precision must be one of {sorted(BYTES_PER_AMP)}, "
                f"got {self.precision!r}")
        if self.recovery_mode not in RECOVERY_MODES:
            raise ValueError(
                f"recovery_mode must be one of {RECOVERY_MODES}, "
                f"got {self.recovery_mode!r}")
        if self.num_ranks < 1:
            raise ValueError("num_ranks must be >= 1")
        if not self.allow_non_power_of_two and not _is_power_of_two(self.num_ranks):
            raise ValueError(
                f"num_ranks must be a power of two for the MPI runner "
                f"(got {self.num_ranks}); pass allow_non_power_of_two=True "
                f"to override")
        if not (0.0 <= self.reserved_storage_fraction < 1.0):
            raise ValueError("reserved_storage_fraction must be in [0, 1)")
        if not (0.0 <= self.max_temp_storage_fraction < 1.0):
            raise ValueError("max_temp_storage_fraction must be in [0, 1)")
        if (self.committed_generations_retained is not None
                and self.committed_generations_retained < 1):
            raise ValueError("committed_generations_retained must be >= 1")
        if (self.durable_snapshots_retained is not None
                and self.durable_snapshots_retained < 0):
            raise ValueError("durable_snapshots_retained must be >= 0")

    # ── resolved retention policy ────────────────────────────────────────
    @property
    def _defaults(self) -> dict:
        return RECOVERY_DEFAULTS[self.recovery_mode]

    @property
    def committed_generations(self) -> int:
        if self.committed_generations_retained is not None:
            return self.committed_generations_retained
        return self._defaults["committed"]

    @property
    def destination_generation(self) -> bool:
        if self.needs_destination_generation is not None:
            return self.needs_destination_generation
        return self._defaults["destination"]

    @property
    def durable_snapshots(self) -> int:
        if self.durable_snapshots_retained is not None:
            return self.durable_snapshots_retained
        return self._defaults["durable"]

    # ── derived hardware capacities (in bytes) ───────────────────────────
    @property
    def storage_per_rank_bytes(self) -> float:
        return self.local_storage_per_rank_tib * TIB

    @property
    def usable_storage_per_rank_bytes(self) -> float:
        return self.storage_per_rank_bytes * (1.0 - self.reserved_storage_fraction)

    @property
    def available_local_storage_bytes(self) -> float:
        return self.storage_per_rank_bytes * self.num_ranks

    @property
    def durable_storage_bytes(self) -> float | None:
        if self.durable_storage_available_tib is None:
            return None
        return self.durable_storage_available_tib * TIB

    @property
    def ram_buffer_budget_bytes(self) -> float:
        return max(self.ram_per_rank_gib - RAM_OS_RESERVE_GIB, 0.0) * GIB

    @property
    def mpi_buffer_budget_bytes(self) -> float:
        return self.ram_buffer_budget_bytes * MPI_BUFFER_FRACTION


# ── per-qubit-count feasibility report ───────────────────────────────────
@dataclass
class QubitFeasibility:
    num_qubits: int
    precision: str
    recovery_mode: str

    state_size_bytes: float
    per_rank_state_bytes: float

    # resolved retention policy used for this evaluation
    committed_generations_retained: int
    needs_destination_generation: bool
    durable_snapshots_retained: int

    # local storage breakdown (per rank, bytes)
    hot_source_bytes: float
    destination_temp_bytes: float
    retained_local_recovery_bytes: float
    durable_local_bytes: float                  # durable folded into local
    wal_recovery_overhead_bytes: float          # wal/commit/manifest metadata
    total_local_required_per_rank_bytes: float
    usable_local_per_rank_bytes: float

    # durable checkpoint (separate budget; aggregate over the whole state)
    durable_checkpoint_required_bytes: float
    durable_storage_bytes: float | None

    # RAM / MPI budgets (per rank, bytes)
    ram_buffer_budget_bytes: float
    mpi_buffer_budget_bytes: float

    # verdicts
    local_feasible: bool
    ram_feasible: bool
    durable_feasible: bool
    feasible: bool
    reasons: list[str] = field(default_factory=list)

    @property
    def total_local_required_bytes(self) -> float:
        """Aggregate across all ranks."""
        return self.total_local_required_per_rank_bytes * _ranks_for(self)

    @property
    def storage_margin_per_rank_bytes(self) -> float:
        return self.usable_local_per_rank_bytes - self.total_local_required_per_rank_bytes


# ranks are not stored on the report; recompute aggregate via state/per-rank.
def _ranks_for(f: "QubitFeasibility") -> float:
    if f.per_rank_state_bytes == 0:
        return 1.0
    return f.state_size_bytes / f.per_rank_state_bytes


# ── core arithmetic ──────────────────────────────────────────────────────
def state_size_bytes(num_qubits: int, precision: str = "complex64") -> int:
    """Exact dense state-vector size: 2^num_qubits * bytes_per_amplitude."""
    if precision not in BYTES_PER_AMP:
        raise ValueError(f"unknown precision {precision!r}")
    if num_qubits < 0:
        raise ValueError("num_qubits must be >= 0")
    return (1 << num_qubits) * BYTES_PER_AMP[precision]


def per_rank_state_bytes(num_qubits: int, num_ranks: int,
                         precision: str = "complex64") -> float:
    """One rank's slice of the state (state evenly partitioned across ranks)."""
    return state_size_bytes(num_qubits, precision) / num_ranks


# WAL/commit/manifest metadata: ~256 bytes per 128 MiB chunk of state, capped.
# Always negligible next to the state, but reported honestly rather than 0.
def _wal_metadata_overhead_bytes(per_rank_state: float, recovery_mode: str) -> float:
    if recovery_mode == "none":
        return 0.0
    chunks = max(per_rank_state / (128 * GIB / 1024), 1.0)
    return min(chunks * 256.0, 64 * GIB)


def evaluate_qubits(config: PlannerConfig, num_qubits: int) -> QubitFeasibility:
    """Score a single qubit count against the configured hardware + policy."""
    precision = config.precision
    mode = config.recovery_mode

    S_total = float(state_size_bytes(num_qubits, precision))
    S = S_total / config.num_ranks

    committed = config.committed_generations
    has_dest = config.destination_generation
    snapshots = config.durable_snapshots

    # ── local storage components (per rank) ──
    hot_source = 1.0 * S
    destination = (1.0 if has_dest else 0.0) * S
    temp_headroom = config.max_temp_storage_fraction * S
    destination_temp = destination + temp_headroom
    retained_recovery = max(committed - 1, 0) * S
    wal_overhead = _wal_metadata_overhead_bytes(S, mode)

    # ── durable checkpoint ──
    durable_checkpoint_required = snapshots * S_total          # aggregate
    durable_storage = config.durable_storage_bytes
    durable_local = 0.0
    if snapshots > 0 and durable_storage is None:
        durable_local = snapshots * S        # this rank's slice folds onto local

    total_local = (hot_source + destination_temp + retained_recovery
                   + wal_overhead + durable_local)
    usable_local = config.usable_storage_per_rank_bytes

    ram_budget = config.ram_buffer_budget_bytes
    mpi_budget = config.mpi_buffer_budget_bytes

    reasons: list[str] = []

    local_feasible = total_local <= usable_local
    if not local_feasible:
        reasons.append(
            f"local storage: need {total_local / TIB:.3f} TiB/rank, "
            f"have {usable_local / TIB:.3f} TiB/rank usable")

    ram_feasible = ram_budget >= MIN_PIPELINE_RAM_GIB * GIB
    if not ram_feasible:
        reasons.append(
            f"RAM: streaming pipeline needs >= {MIN_PIPELINE_RAM_GIB:.1f} GiB "
            f"after a {RAM_OS_RESERVE_GIB:.0f} GiB OS reserve, "
            f"have {ram_budget / GIB:.2f} GiB")

    if snapshots > 0 and durable_storage is not None:
        durable_feasible = durable_checkpoint_required <= durable_storage
        if not durable_feasible:
            reasons.append(
                f"durable checkpoint: need {durable_checkpoint_required / TIB:.3f} "
                f"TiB, have {durable_storage / TIB:.3f} TiB")
    else:
        durable_feasible = True  # not required, or folded into local above

    feasible = local_feasible and ram_feasible and durable_feasible

    return QubitFeasibility(
        num_qubits=num_qubits,
        precision=precision,
        recovery_mode=mode,
        state_size_bytes=S_total,
        per_rank_state_bytes=S,
        committed_generations_retained=committed,
        needs_destination_generation=has_dest,
        durable_snapshots_retained=snapshots,
        hot_source_bytes=hot_source,
        destination_temp_bytes=destination_temp,
        retained_local_recovery_bytes=retained_recovery,
        durable_local_bytes=durable_local,
        wal_recovery_overhead_bytes=wal_overhead,
        total_local_required_per_rank_bytes=total_local,
        usable_local_per_rank_bytes=usable_local,
        durable_checkpoint_required_bytes=durable_checkpoint_required,
        durable_storage_bytes=durable_storage,
        ram_buffer_budget_bytes=ram_budget,
        mpi_buffer_budget_bytes=mpi_budget,
        local_feasible=local_feasible,
        ram_feasible=ram_feasible,
        durable_feasible=durable_feasible,
        feasible=feasible,
        reasons=reasons,
    )


def max_feasible_qubits(config: PlannerConfig) -> int | None:
    """Largest qubit count feasible under ``config``.

    Returns ``None`` if even a 1-qubit state does not fit.  Feasibility is
    monotone in qubit count (state doubles each qubit while hardware is fixed),
    so an ascending scan to the first infeasible count is exact.
    """
    cap = config.max_candidate_qubits or DEFAULT_MAX_CANDIDATE_QUBITS
    best: int | None = None
    for q in range(1, cap + 1):
        if evaluate_qubits(config, q).feasible:
            best = q
        else:
            break
    return best


def recommend_recovery_mode(config: PlannerConfig, num_qubits: int) -> str | None:
    """Strongest recovery mode that keeps ``num_qubits`` feasible on this hardware.

    Tries modes most- to least-protective using each mode's *default* retention
    policy (the explicit retention overrides on ``config`` are cleared so the
    comparison is between policies, not a single hand-tuned one).  Returns
    ``None`` if not even ``none`` fits.
    """
    for mode in reversed(RECOVERY_MODES):  # strongest first
        trial = replace(
            config, recovery_mode=mode,
            committed_generations_retained=None,
            needs_destination_generation=None,
            durable_snapshots_retained=None,
        )
        if evaluate_qubits(trial, num_qubits).feasible:
            return mode
    return None


# ── plan assembly + JSON-friendly output ─────────────────────────────────
def _tib(x: float | None) -> float | None:
    return None if x is None else round(x / TIB, 6)


def _gib(x: float | None) -> float | None:
    return None if x is None else round(x / GIB, 6)


def _feasibility_dict(f: QubitFeasibility, num_ranks: int) -> dict:
    return {
        "num_qubits": f.num_qubits,
        "feasible": f.feasible,
        "state_size_tib": _tib(f.state_size_bytes),
        "per_rank_state_tib": _tib(f.per_rank_state_bytes),
        # retention policy actually used
        "committed_generations_retained": f.committed_generations_retained,
        "needs_destination_generation": f.needs_destination_generation,
        "durable_snapshots_retained": f.durable_snapshots_retained,
        # six separately-reported storage categories
        "hot_source_per_rank_tib": _tib(f.hot_source_bytes),
        "destination_temp_per_rank_tib": _tib(f.destination_temp_bytes),
        "retained_local_recovery_per_rank_tib":
            _tib(f.retained_local_recovery_bytes),
        "durable_checkpoint_total_tib": _tib(f.durable_checkpoint_required_bytes),
        "total_local_required_per_rank_tib":
            _tib(f.total_local_required_per_rank_bytes),
        "total_local_required_tib": _tib(f.total_local_required_bytes),
        "total_durable_required_tib": _tib(f.durable_checkpoint_required_bytes),
        # context
        "wal_recovery_overhead_per_rank_tib": _tib(f.wal_recovery_overhead_bytes),
        "durable_local_per_rank_tib": _tib(f.durable_local_bytes),
        "usable_local_per_rank_tib": _tib(f.usable_local_per_rank_bytes),
        "storage_margin_per_rank_tib": _tib(f.storage_margin_per_rank_bytes),
        "durable_storage_available_tib": _tib(f.durable_storage_bytes),
        "ram_buffer_budget_per_rank_gib": _gib(f.ram_buffer_budget_bytes),
        "mpi_buffer_budget_per_rank_gib": _gib(f.mpi_buffer_budget_bytes),
        "local_feasible": f.local_feasible,
        "ram_feasible": f.ram_feasible,
        "durable_feasible": f.durable_feasible,
        "reasons": f.reasons,
    }


def _recommended_config(config: PlannerConfig,
                        f: QubitFeasibility | None) -> dict | None:
    """A self-consistent, runnable configuration the user can launch as-is."""
    if f is None:
        return None
    warnings: list[str] = []
    margin = f.storage_margin_per_rank_bytes
    usable = f.usable_local_per_rank_bytes
    if usable > 0 and margin / usable < 0.05:
        warnings.append("configuration is tight (local margin < 5%)")
    if f.durable_snapshots_retained > 0 and config.durable_storage_available_tib is None:
        warnings.append("durable checkpoint folded onto local NVMe "
                        "(no separate durable store supplied)")
    return {
        "recommended_num_qubits": f.num_qubits,
        "precision": f.precision,
        "num_ranks": config.num_ranks,
        "recovery_mode": f.recovery_mode,
        "committed_generations_retained": f.committed_generations_retained,
        "needs_destination_generation": f.needs_destination_generation,
        "durable_snapshots_retained": f.durable_snapshots_retained,
        "total_local_required_tib": _tib(f.total_local_required_bytes),
        "total_local_required_per_rank_tib":
            _tib(f.total_local_required_per_rank_bytes),
        "total_durable_required_tib": _tib(f.durable_checkpoint_required_bytes),
        "available_local_storage_tib": _tib(config.available_local_storage_bytes),
        "storage_margin_per_rank_tib": _tib(margin),
        "feasible": f.feasible,
        "warnings": warnings,
    }


def plan(config: PlannerConfig, num_qubits: int | None = None) -> dict:
    """Produce the full capacity plan.

    Reports the maximum-feasible frontier, an optional explicit feasibility
    check, and a self-consistent recommended runnable config.
    """
    max_q = max_feasible_qubits(config)

    out: dict = {
        "precision": config.precision,
        "num_ranks": config.num_ranks,
        "storage_per_rank_tib": config.local_storage_per_rank_tib,
        "ram_per_rank_gib": config.ram_per_rank_gib,
        "available_local_storage_tib": _tib(config.available_local_storage_bytes),
        "reserved_storage_fraction": config.reserved_storage_fraction,
        "max_temp_storage_fraction": config.max_temp_storage_fraction,
        "recovery_mode": config.recovery_mode,
        "committed_generations_retained": config.committed_generations,
        "needs_destination_generation": config.destination_generation,
        "durable_snapshots_retained": config.durable_snapshots,
        "max_feasible_qubits": max_q,
    }

    if max_q is not None:
        f_max = evaluate_qubits(config, max_q)
        out["state_size_at_max_tib"] = _tib(f_max.state_size_bytes)
        out["per_rank_state_at_max_tib"] = _tib(f_max.per_rank_state_bytes)
        out["required_local_storage_at_max_tib"] = _tib(
            f_max.total_local_required_bytes)
        out["total_durable_required_at_max_tib"] = _tib(
            f_max.durable_checkpoint_required_bytes)
        out["storage_margin_at_max_tib"] = _tib(
            f_max.storage_margin_per_rank_bytes * config.num_ranks)
    else:
        for key in ("state_size_at_max_tib", "per_rank_state_at_max_tib",
                    "required_local_storage_at_max_tib",
                    "total_durable_required_at_max_tib",
                    "storage_margin_at_max_tib"):
            out[key] = None

    if num_qubits is not None:
        f = evaluate_qubits(config, num_qubits)
        out["requested_qubits"] = num_qubits
        out["requested_feasible"] = f.feasible
        out["requested"] = _feasibility_dict(f, config.num_ranks)
        out["recommended_recovery_mode"] = recommend_recovery_mode(
            config, num_qubits)
        target = f
    elif max_q is not None:
        out["recommended_recovery_mode"] = recommend_recovery_mode(config, max_q)
        target = evaluate_qubits(config, max_q)
    else:
        out["recommended_recovery_mode"] = None
        target = None

    out["recommended_config"] = _recommended_config(
        config, evaluate_qubits(config, max_q) if max_q is not None else None)
    out["warning"] = _warning(config, target)
    return out


def _warning(config: PlannerConfig, target: QubitFeasibility | None) -> str | None:
    if target is None:
        return "no qubit count is feasible on this hardware"
    if not target.feasible:
        return "; ".join(target.reasons) if target.reasons else "infeasible"

    msgs: list[str] = []
    margin = target.storage_margin_per_rank_bytes
    usable = target.usable_local_per_rank_bytes
    if usable > 0 and margin / usable < 0.05:
        msgs.append("configuration is tight")
    if target.durable_snapshots_retained > 0:
        if config.durable_storage_available_tib is None:
            msgs.append("durable checkpoint requires additional storage")
        else:
            msgs.append("durable checkpoint uses separate durable storage")
    return "; ".join(msgs) if msgs else None


# ── CLI ────────────────────────────────────────────────────────────────────
def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m wenbo_engine.planner.capacity_planner",
        description="Hardware-bound capacity planner for exact state-vector "
                    "simulation: find the largest feasible qubit count (or "
                    "check one explicit count) for the given hardware and "
                    "recovery policy.",
    )
    p.add_argument("--num-qubits", type=int, default=None,
                   help="check feasibility of this explicit qubit count "
                        "(omit to report the maximum feasible count)")
    p.add_argument("--precision", choices=sorted(BYTES_PER_AMP), default="complex64")
    p.add_argument("--num-ranks", type=int, required=True)
    p.add_argument("--storage-per-rank-tib", type=float, required=True,
                   dest="local_storage_per_rank_tib")
    p.add_argument("--ram-per-rank-gib", type=float, default=0.0,
                   dest="ram_per_rank_gib")
    p.add_argument("--reserved-storage-fraction", type=float, default=0.15)
    p.add_argument("--max-temp-storage-fraction", type=float, default=0.10)
    p.add_argument("--recovery-mode", choices=RECOVERY_MODES, default="wal")
    p.add_argument("--durable-storage-available-tib", type=float, default=None)
    p.add_argument("--max-candidate-qubits", type=int, default=None)
    # retention policy overrides (default None -> derived from recovery mode)
    p.add_argument("--committed-generations-retained", type=int, default=None)
    p.add_argument("--needs-destination-generation",
                   action=argparse.BooleanOptionalAction, default=None,
                   help="whether a step writes a separate destination "
                        "generation (default: derived from recovery mode)")
    p.add_argument("--durable-snapshots-retained", type=int, default=None)
    p.add_argument("--allow-non-power-of-two", action="store_true",
                   help="permit a non-power-of-two rank count (the MPI runner "
                        "requires a power of two)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        config = PlannerConfig(
            precision=args.precision,
            num_ranks=args.num_ranks,
            local_storage_per_rank_tib=args.local_storage_per_rank_tib,
            ram_per_rank_gib=args.ram_per_rank_gib,
            reserved_storage_fraction=args.reserved_storage_fraction,
            max_temp_storage_fraction=args.max_temp_storage_fraction,
            recovery_mode=args.recovery_mode,
            durable_storage_available_tib=args.durable_storage_available_tib,
            max_candidate_qubits=args.max_candidate_qubits,
            committed_generations_retained=args.committed_generations_retained,
            needs_destination_generation=args.needs_destination_generation,
            durable_snapshots_retained=args.durable_snapshots_retained,
            allow_non_power_of_two=args.allow_non_power_of_two,
        )
    except ValueError as e:
        print(json.dumps({"error": str(e)}, indent=2))
        return 2

    result = plan(config, num_qubits=args.num_qubits)
    print(json.dumps(result, indent=2))
    if args.num_qubits is not None and not result.get("requested_feasible", False):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
