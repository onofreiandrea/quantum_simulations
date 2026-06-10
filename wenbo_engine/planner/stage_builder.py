"""Build execution stages per ablation mode.

A *stage* is one I/O pass over (a rank's share of) the state vector: it
reads chunks, applies its gates, writes chunks, and (under a durable
recovery mode) commits.  Stages carry the concrete quantities the cost
model consumes plus enough structure to (a) verify gate dependencies and
(b) replay the plan to a state vector for the equivalence test.

Two staging families are produced:

  * **levelized** (modes ``current`` / ``current_static_reorder``):
    one stage per levelized layer, gates classified local / rank-nonlocal
    / mpi-nonlocal exactly as
    :func:`wenbo_engine.mpi.mpi_runner._compile_steps` does.  This is
    *today's behavior* — the baseline.

  * **staged v2** (modes ``stage_v2`` / ``stage_v2_fusion`` /
    ``stage_v2_placement_fusion``): Atlas-style local-set staging
    (reusing :func:`wenbo_engine.circuit.staging.atlas_stages`), which
    rearranges the physical layout via SWAPs so most gates become
    chunk-local — trading a few SWAP passes for many avoided nonlocal
    passes.  The ``*_fusion`` variants additionally merge consecutive
    all-local compute stages into a single I/O pass (the level-batching
    optimisation of :mod:`wenbo_engine.circuit.fusion`), strictly
    reducing the number of full-state passes.

Op classification uses the SAME physical-bit layout as the MPI runner:

    local         bit < k
    rank-nonlocal k <= bit < n - p
    mpi-nonlocal  bit >= n - p
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from wenbo_engine.circuit.io import validate_circuit_dict, levelize
from wenbo_engine.circuit.staging import atlas_stages
from wenbo_engine.kernel import gates as gmod


# ── op record ────────────────────────────────────────────────────────

@dataclass
class PlannedOp:
    """One gate (or SWAP) to apply, in PHYSICAL-bit coordinates.

    ``klass`` is ``local`` / ``rank_nonlocal`` / ``mpi_nonlocal``.

    A named gate carries ``gate`` + ``params``.  A staged stage may emit a
    pre-multiplied / fused matrix that has no gate name; those are encoded
    with ``gate == "__matrix__"`` and ``params["matrix"]`` (a real/imag
    nested list), so the plan stays JSON-serializable and exactly
    replayable without re-deriving the matrix.
    """
    qubits: list[int]
    gate: str
    params: dict = field(default_factory=dict)
    klass: str = "local"


@dataclass
class BuiltStage:
    """A staging-level result before cost annotation (see optimizer_v2)."""
    ops: list[PlannedOp]
    kind: str  # "compute" or "swap"

    def class_counts(self) -> tuple[int, int, int]:
        loc = rnl = mnl = 0
        for op in self.ops:
            if op.klass == "local":
                loc += 1
            elif op.klass == "rank_nonlocal":
                rnl += 1
            else:
                mnl += 1
        return loc, rnl, mnl

    def is_local_only(self) -> bool:
        return all(op.klass == "local" for op in self.ops)


# ── classification (mirrors mpi_runner._compile_steps) ───────────────

def classify(qubits: list[int], k: int, n_local_bits: int) -> str:
    """Classify a gate by its PHYSICAL qubit bits.

    Matches
    :func:`wenbo_engine.bench.communication_workloads.classify_gate` and
    ``mpi_runner._compile_steps``.
    """
    if all(q < k for q in qubits):
        return "local"
    if any((q - k) >= n_local_bits for q in qubits if q >= k):
        return "mpi_nonlocal"
    return "rank_nonlocal"


# ── levelized builder (baseline / static-reorder modes) ──────────────

def build_levelized_stages(circuit_dict: dict, k: int,
                           n_local_bits: int) -> list[BuiltStage]:
    """One stage per levelized layer (today's behavior).

    Gates are kept in PHYSICAL coordinates (the circuit dict has already
    been relabelled by any reorder/placement step).  Each non-empty level
    becomes a single compute stage.
    """
    cd = validate_circuit_dict(circuit_dict)
    levels = levelize(cd)
    stages: list[BuiltStage] = []
    for lv in levels:
        if not lv:
            continue
        ops = [
            PlannedOp(
                qubits=list(g["qubits"]),
                gate=g["gate"],
                params=dict(g.get("params", {})),
                klass=classify(g["qubits"], k, n_local_bits),
            )
            for g in lv
        ]
        stages.append(BuiltStage(ops=ops, kind="compute"))
    return stages


# ── staged-v2 builder ────────────────────────────────────────────────

def build_stage_v2_stages(circuit_dict: dict, k: int, n_local_bits: int,
                          fusion: bool) -> tuple[list[BuiltStage], list[int]]:
    """Atlas-style local-set staging, optionally with local-pass fusion.

    Reuses :func:`wenbo_engine.circuit.staging.atlas_stages`, whose steps
    are dicts ``{"local_ops": [(phys_qubits, U), ...], "nonlocal_ops": ...}``
    already expressed in PHYSICAL coordinates (SWAPs included).  Each runner
    step becomes one :class:`BuiltStage`.

    When ``fusion`` is True, consecutive all-local compute stages are merged
    into a single I/O pass (level batching), strictly reducing the number of
    full-state passes versus the non-fused staging.

    Returns ``(stages, log_to_phys)`` where ``log_to_phys`` is atlas's final
    logical->physical bit mapping (needed to replay/verify the plan).
    """
    steps, log_to_phys = atlas_stages(circuit_dict, k, method="heuristic")

    stages: list[BuiltStage] = []
    for step in steps:
        local_ops = list(step.get("local_ops", []))
        nonlocal_ops = list(step.get("nonlocal_ops", []))
        all_ops = local_ops + nonlocal_ops
        if not all_ops:
            continue
        is_swap_stage = (
            bool(nonlocal_ops) and not local_ops
            and all(_looks_like_swap(U) for _qs, U in nonlocal_ops)
        )
        ops = [
            PlannedOp(
                qubits=list(qs),
                gate="__matrix__",
                params={"matrix": _matrix_to_list(U)},
                klass=classify(qs, k, n_local_bits),
            )
            for qs, U in all_ops
        ]
        stages.append(BuiltStage(
            ops=ops, kind="swap" if is_swap_stage else "compute"))

    if fusion:
        return _merge_local_passes(stages), log_to_phys
    return stages, log_to_phys


def _merge_local_passes(stages: list[BuiltStage]) -> list[BuiltStage]:
    """Merge runs of consecutive all-local compute stages into one pass.

    This is the level-batching optimisation: several local-only passes that
    only read+write the same chunks can be collapsed into a single pass.
    Ordering of non-disjoint ops within the merged stage is preserved
    (stages are concatenated in order), so dependencies are respected.
    """
    out: list[BuiltStage] = []
    buf: list[PlannedOp] = []

    def flush():
        if buf:
            out.append(BuiltStage(ops=list(buf), kind="compute"))
            buf.clear()

    for st in stages:
        if st.kind == "compute" and st.is_local_only():
            buf.extend(st.ops)
        else:
            flush()
            out.append(st)
    flush()
    return out


# ── matrix encoding helpers ──────────────────────────────────────────

def _matrix_to_list(U) -> list:
    arr = np.asarray(U)
    return [[[float(v.real), float(v.imag)] for v in row] for row in arr]


def matrix_from_list(data: list) -> np.ndarray:
    return np.array(
        [[complex(re, im) for re, im in row] for row in data],
        dtype=np.complex128,
    )


def _looks_like_swap(U) -> bool:
    arr = np.asarray(U)
    if arr.shape != (4, 4):
        return False
    return bool(np.allclose(arr, gmod.SWAP(), atol=1e-9))
