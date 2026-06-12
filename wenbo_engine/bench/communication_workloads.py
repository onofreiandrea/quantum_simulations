#!/usr/bin/env python3
"""MPI-nonlocal communication benchmark suite.

The 38-40 qubit benchmark circuits were *reordered* (see
``wenbo_engine.circuit.reorder``) so that, after static qubit reordering,
zero gates landed on rank bits — every gate executed locally or
rank-nonlocally and no MPI traffic was generated.  That is the right
thing to do for production, but it means inter-node communication was
never actually exercised.

This module produces workloads that **deliberately force** the three
gate classes the MPI runner distinguishes, and *measures* the resulting
communication.  Reordering is therefore *off* by default here — the
whole point is to keep MPI-nonlocal gates MPI-nonlocal (see ``--reorder``
to opt into the production reordering path for comparison).

Measured vs estimated
---------------------
All ``mpi_*`` / ``bytes_*`` / ``*_sec`` / ``*_time`` counters in the
output are **measured at runtime**: a ``ProfilingComm`` proxy times the
*actual* ``Sendrecv`` calls and counts bytes from the *actual* buffers
passed to them; the chunk I/O wrappers time and size *actual* reads and
writes; ``partner_rank_pairs`` is the set of *actual* communication
partners observed across ranks.  The only static (non-measured) quantity
is exported under the ``estimated_`` prefix
(``estimated_partner_rank_pairs``) so it is never confused with a
measurement.  ``measured_*_ops`` come from the real runner compiler
(``mpi_runner._compile_steps``), not from this module's static classifier.

Gate classification (matches ``mpi_runner._compile_steps``)
-----------------------------------------------------------
With ``n`` qubits, chunk holding ``k = chunk_bits`` local qubits, and
``p = log2(num_ranks)`` rank bits, ``n_local_bits = n - k - p`` is the
number of chunk-index bits that stay on the same rank:

    local         qubit < k                         within-chunk
    rank-nonlocal k <= qubit < n - p                partner chunk, same rank
    mpi-nonlocal  qubit >= n - p                     partner chunk on another rank

Generators
----------
``communication_light(n, depth, seed)``
    Mostly low-bit (local) gates — minimal MPI traffic.
``rank_nonlocal_heavy(n, depth, chunk_bits, seed, num_ranks=None)``
    Gates touch chunk-index bits on the same rank (no MPI).  Rank bits are
    derived from ``num_ranks`` when given (so it is correct for any rank
    count); otherwise ``DEFAULT_RANK_BITS`` are reserved.
``mpi_nonlocal_heavy(n, depth, chunk_bits, num_ranks, seed)``
    Gates touch rank bits, forcing inter-node Sendrecv.
``mixed_staged(n, depth, chunk_bits, num_ranks, seed)``
    Phased: a block of local, then rank-nonlocal, then MPI-nonlocal gates.

CLI
---
    mpirun -np 4 python -m wenbo_engine.bench.communication_workloads \
        --kind mpi_nonlocal_heavy --n 24 --depth 20 \
        --output-dir /tmp/proof_mpi_bench_artifacts

``--output-dir`` emits the standard experiment artifact bundle
(``config.json``, ``circuit.json``, ``plan.json``, ``cost_model.json``,
``stage_profile.csv``, ``mpi_profile.csv``, ``io_profile.csv``,
``recovery_events.json``, ``final_summary.json``, ``final_norm.txt``,
``git_commit.txt``) using the same CSV/JSON schema as
``wenbo_engine.profiling`` / ``wenbo_engine.experiments`` when present,
and a schema-compatible fallback otherwise.
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import inspect
import json
import logging
import math
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock

import numpy as np

log = logging.getLogger(__name__)

# Durability modes accepted by --recovery.  "generation" requires the
# generation-recovery integration (Agent 2) to be merged — mpi_runner.run
# must accept a ``recovery`` parameter and wenbo_engine.recovery must import.
RECOVERY_MODES = ("none", "wal", "generation")

# Rank bits reserved by ``rank_nonlocal_heavy`` ONLY when it is not told
# ``num_ranks``.  When ``num_ranks`` is supplied the reservation is derived
# exactly from it (p = log2(num_ranks)), so the generator is correct for
# any rank count.  This constant is just the back-compat fallback.
DEFAULT_RANK_BITS = 2

INTENDED_LOCALITY = {
    "communication_light": "local",
    "rank_nonlocal_heavy": "rank_nonlocal",
    "mpi_nonlocal_heavy": "mpi_nonlocal",
    "mpi_nonlocal_phase_heavy": "mpi_nonlocal",
    "mpi_nonlocal_mixing_heavy": "mpi_nonlocal",
    "mixed_staged": "mixed",
}

# CSV schemas — kept identical to wenbo_engine.profiling so artifacts are
# drop-in compatible with the observability summary reader.  Mirrored here
# (not imported) so the bundle can be written even when that branch is not
# merged into the current tree.
STAGE_COLUMNS = [
    "step_or_stage_id", "local_ops", "rank_nonlocal_ops", "mpi_nonlocal_ops",
    "read_sec", "write_sec", "kernel_sec", "mpi_sec", "commit_sec",
    "checksum_sec", "bytes_read", "bytes_written", "mpi_bytes_sent",
    "recovery_mode",
]
MPI_COLUMNS = ["stage_id", "kind", "op", "peer", "bytes_sent", "seconds", "mb_per_s"]
IO_COLUMNS = ["stage_id", "direction", "bytes", "seconds", "mb_per_s"]


# ── gate construction helpers ──────────────────────────────────────────

def _h(q: int) -> dict:
    return {"qubits": [q], "gate": "H"}


def _rot(rng: np.random.RandomState, q: int) -> dict:
    g = str(rng.choice(["RX", "RY", "RZ"]))
    return {"qubits": [q], "gate": g, "params": {"theta": float(rng.uniform(0, np.pi))}}


def _two(rng: np.random.RandomState, qa: int, qb: int) -> dict:
    """Random 2-qubit entangling gate on distinct qubits qa != qb."""
    assert qa != qb, "2q gate needs distinct qubits"
    g = str(rng.choice(["CNOT", "CZ"]))
    return {"qubits": [qa, qb], "gate": g}


# ── non-Clifford injection ─────────────────────────────────────────────
#
# CNOT/CZ/H/S are Clifford; a circuit built only from them is a *stabilizer*
# circuit (efficiently classically simulable via Gottesman–Knill), which is a
# weak stress test for a full state-vector engine.  We deliberately seed every
# workload with non-Clifford gates so the simulated state is genuinely
# non-stabilizer, while preserving each workload's intended locality class.
#
# Non-Clifford primitives used:
#   * RX/RY/RZ(θ), θ ∉ (π/2)ℤ   — arbitrary rotations (non-Clifford); already
#                                 present in ``communication_light``
#   * CR(k), k ≥ 3              — controlled phase 2π/2^k; k=1→CZ, k=2→CS are
#                                 Clifford, k≥3 (controlled-T, …) is non-Clifford
#
# ``_NONCLIFFORD_EVERY`` = 3 makes ~1/3 of a workload's gates non-Clifford on a
# fixed, deterministic cadence (so at least one appears for any depth ≥ 1).
_NONCLIFFORD_EVERY = 3


def _nonclifford_two(rng: np.random.RandomState, qa: int, qb: int) -> dict:
    """A guaranteed non-Clifford 2-qubit entangler (CR with k ≥ 3) on qa, qb.

    Same two-qubit structure as :func:`_two` (a controlled phase on the same
    qubit pair), so it preserves the locality class of the gate it replaces
    while making the operation non-Clifford.
    """
    assert qa != qb, "2q gate needs distinct qubits"
    return {"qubits": [qa, qb], "gate": "CR", "params": {"k": int(rng.choice([3, 4, 5]))}}


def _pick(rng: np.random.RandomState, lo: int, hi: int, exclude: int | None = None) -> int:
    """Random qubit in [lo, hi); raises if the range is empty after exclusion."""
    if hi <= lo:
        raise ValueError(f"empty qubit range [{lo}, {hi})")
    while True:
        q = int(rng.randint(lo, hi))
        if q != exclude:
            return q


def _rank_bits(num_ranks: int) -> int:
    _check_pow2(num_ranks)
    return int(round(math.log2(num_ranks)))


def _available_ram_gib() -> float:
    """Total physical RAM (GiB) on this node, via sysconf (no psutil dep)."""
    try:
        import os as _os
        total = _os.sysconf("SC_PAGE_SIZE") * _os.sysconf("SC_PHYS_PAGES")
        return total / (1 << 30)
    except (ValueError, OSError, AttributeError):
        return 0.0


# ── generators ─────────────────────────────────────────────────────────

def communication_light(n: int, depth: int, seed: int = 42) -> dict:
    """Mostly low-bit gates → low MPI traffic.

    ~70% of gates act on the lowest ``max(1, n//4)`` qubits (which map to
    chunk-local positions under any reasonable partition); the remaining
    gates are local 2-qubit entanglers between adjacent low qubits.  No
    gate is placed on a high qubit, so MPI traffic stays minimal.

    Non-Clifford content: the random single-qubit ``_rot`` gates are arbitrary
    ``RX/RY/RZ(θ)`` rotations on low/local bits, so the circuit is non-stabilizer
    for the default seeds — but that is *probabilistic*.  To guarantee benchmark
    hardness for ANY seed/depth, the first gate is deterministically forced to an
    explicit non-Clifford gate (RZ(π/7) for a 1-qubit slot, CR(3) for a 2-qubit
    slot) on the SAME qubit(s) and with the SAME arity it would otherwise have.
    Reusing the qubits keeps the levelization / locality class / compute-unit
    fusion byte-identical — only the gate's matrix changes — so the workload
    stays local (``mpi_nonlocal == 0``) and the long local run is preserved.
    """
    if n < 1:
        raise ValueError("n must be >= 1")
    rng = np.random.RandomState(seed)
    low = max(1, n // 4)
    gates: list[dict] = []
    for _ in range(depth):
        if low >= 2 and rng.random() < 0.3:
            qa = _pick(rng, 0, low)
            qb = _pick(rng, 0, low, exclude=qa)
            gates.append(_two(rng, qa, qb))
        else:
            q = _pick(rng, 0, low)
            gates.append(_rot(rng, q) if rng.random() < 0.5 else _h(q))

    # Deterministic non-Clifford guarantee: replace the first gate in place with
    # an explicit non-Clifford gate of the same arity on the same qubit(s).  This
    # preserves the qubit-dependency structure exactly (levels, locality counts,
    # compute-unit fusion) while making the circuit non-stabilizer for every seed.
    if gates:
        qs = gates[0]["qubits"]
        if len(qs) == 1:
            gates[0] = {"qubits": list(qs), "gate": "RZ",
                        "params": {"theta": math.pi / 7}}
        else:
            gates[0] = {"qubits": list(qs), "gate": "CR", "params": {"k": 3}}
    return {"number_of_qubits": n, "gates": gates}


def rank_nonlocal_heavy(n: int, depth: int, chunk_bits: int,
                        seed: int = 42, num_ranks: int | None = None) -> dict:
    """Gates touch chunk-index bits on the *same* rank (no MPI).

    Each gate is a 2-qubit entangler between a local qubit (``< chunk_bits``)
    and a chunk-index qubit drawn from the rank-LOCAL chunk-bit range
    ``[chunk_bits, n - p)`` where ``p = log2(num_ranks)``.

    When ``num_ranks`` is given the reserved rank bits ``p`` are derived
    from it exactly, so the generated gates are guaranteed rank-nonlocal
    (never MPI-nonlocal) for *that* rank count — for any rank count.  When
    ``num_ranks`` is ``None`` a conservative ``DEFAULT_RANK_BITS`` are
    reserved (back-compat).
    """
    k = chunk_bits
    p = _rank_bits(num_ranks) if num_ranks is not None else DEFAULT_RANK_BITS
    hi = n - p  # exclusive upper bound of rank-local chunk bits
    if k < 1:
        raise ValueError("chunk_bits must be >= 1 (need a local partner qubit)")
    if hi <= k:
        raise ValueError(
            f"no rank-local chunk bits available: need n > chunk_bits + p "
            f"(got n={n}, chunk_bits={k}, p={p})")
    rng = np.random.RandomState(seed)
    gates: list[dict] = []
    for i in range(depth):
        local_q = _pick(rng, 0, k)
        chunk_q = _pick(rng, k, hi)
        # Every gate is a 2-qubit entangler on a (local, chunk-bit) pair, so the
        # rank-nonlocal class is preserved.  ~1/3 (the _NONCLIFFORD_EVERY
        # cadence) are non-Clifford controlled phases on the SAME pair, so the
        # non-Clifford content is mixed directly into the chunk-bit interactions.
        if i % _NONCLIFFORD_EVERY == 0:
            gates.append(_nonclifford_two(rng, local_q, chunk_q))
        else:
            gates.append(_two(rng, local_q, chunk_q))
    return {"number_of_qubits": n, "gates": gates}


def mpi_nonlocal_heavy(n: int, depth: int, chunk_bits: int,
                       num_ranks: int, seed: int = 42) -> dict:
    """Gates deliberately touch rank bits, forcing MPI Sendrecv.

    The top ``p = log2(num_ranks)`` qubits are rank bits.  Each gate is a
    2-qubit entangler between a rank-bit qubit (``>= n - p``) and a local
    qubit (``< chunk_bits``), guaranteeing an inter-rank exchange.
    """
    k = chunk_bits
    p = _rank_bits(num_ranks)
    if p < 1:
        raise ValueError("num_ranks must be >= 2 for MPI-nonlocal gates")
    if k < 1:
        raise ValueError("chunk_bits must be >= 1 (need a local partner qubit)")
    if n - p < k:
        raise ValueError(
            f"no room for rank bits: need n >= chunk_bits + p "
            f"(got n={n}, chunk_bits={k}, p={p})")
    rng = np.random.RandomState(seed)
    gates: list[dict] = []
    for i in range(depth):
        rank_q = _pick(rng, n - p, n)
        local_q = _pick(rng, 0, k)
        # Every gate is a 2-qubit entangler on a (rank-bit, local) pair, so the
        # MPI-nonlocal stress (count + inter-rank partner structure) is
        # unchanged.  ~1/3 (the _NONCLIFFORD_EVERY cadence) are non-Clifford
        # controlled phases on the SAME pair, adding non-Clifford content
        # without removing any MPI-nonlocal exchange.
        if i % _NONCLIFFORD_EVERY == 0:
            gates.append(_nonclifford_two(rng, rank_q, local_q))
        else:
            gates.append(_two(rng, rank_q, local_q))
    return {"number_of_qubits": n, "gates": gates}


def mpi_nonlocal_phase_heavy(n: int, depth: int, chunk_bits: int,
                             num_ranks: int, seed: int = 42) -> dict:
    """MPI-nonlocal gates that are all DIAGONAL (controlled phases / phases).

    Every gate touches a rank bit but is diagonal — controlled-phase on a
    (rank-bit, local) pair, or a 1-qubit phase on a rank bit — so it needs NO
    remote-amplitude exchange.  Designed to benefit strongly from the diagonal
    fast path (sendrecv → 0).  Non-stabilizer via CR(k>=3) controlled phases
    and RZ(theta) rotations on rank bits.
    """
    k = chunk_bits
    p = _rank_bits(num_ranks)
    if p < 1 or k < 1 or n - p < k:
        raise ValueError(f"bad layout: n={n} chunk_bits={k} p={p}")
    rng = np.random.RandomState(seed)
    gates: list[dict] = []
    for i in range(depth):
        rank_q = _pick(rng, n - p, n)
        local_q = _pick(rng, 0, k)
        if i % _NONCLIFFORD_EVERY == 0:
            # non-Clifford diagonal controlled phase on (rank, local)
            gates.append({"qubits": [rank_q, local_q], "gate": "CR",
                          "params": {"k": int(rng.choice([3, 4, 5]))}})
        elif i % 3 == 1:
            gates.append({"qubits": [rank_q, local_q], "gate": "CZ"})  # diagonal
        else:
            # diagonal 1-qubit phase on the rank bit (non-Clifford theta)
            gates.append({"qubits": [rank_q], "gate": "RZ",
                          "params": {"theta": float(rng.uniform(0.15, np.pi / 2 - 0.15))}})
    return {"number_of_qubits": n, "gates": gates}


def mpi_nonlocal_mixing_heavy(n: int, depth: int, chunk_bits: int,
                              num_ranks: int, seed: int = 42) -> dict:
    """MPI-nonlocal gates that TRULY MIX across the rank boundary.

    Every gate is a 1-qubit gate on a rank bit that is NOT diagonal (H / RX /
    RY), so it mixes amplitudes between partner ranks and MUST exchange — the
    diagonal fast path never applies here.  Non-stabilizer via RX/RY(theta).
    """
    k = chunk_bits
    p = _rank_bits(num_ranks)
    if p < 1 or n - p < k:
        raise ValueError(f"bad layout: n={n} chunk_bits={k} p={p}")
    rng = np.random.RandomState(seed)
    gates: list[dict] = []
    for i in range(depth):
        rank_q = _pick(rng, n - p, n)
        m = i % 3
        if m == 0:
            gates.append({"qubits": [rank_q], "gate": "RX",
                          "params": {"theta": float(rng.uniform(0.15, np.pi / 2 - 0.15))}})
        elif m == 1:
            gates.append({"qubits": [rank_q], "gate": "RY",
                          "params": {"theta": float(rng.uniform(0.15, np.pi / 2 - 0.15))}})
        else:
            gates.append({"qubits": [rank_q], "gate": "H"})   # true mixing (Clifford)
    return {"number_of_qubits": n, "gates": gates}


def mixed_staged(n: int, depth: int, chunk_bits: int,
                 num_ranks: int, seed: int = 42) -> dict:
    """Phased workload: local block, then rank-nonlocal, then MPI-nonlocal.

    ``depth`` gates are split roughly into thirds.  The phase boundaries
    are deterministic for a fixed seed.
    """
    _check_pow2(num_ranks)
    third = max(1, depth // 3)
    n_local = third
    n_rank = third
    n_mpi = depth - n_local - n_rank

    light = communication_light(n, n_local, seed)
    rank = rank_nonlocal_heavy(n, n_rank, chunk_bits, seed + 1, num_ranks=num_ranks)
    mpi = mpi_nonlocal_heavy(n, max(n_mpi, 0), chunk_bits, num_ranks, seed + 2)

    return {
        "number_of_qubits": n,
        "gates": light["gates"] + rank["gates"] + mpi["gates"],
    }


GENERATORS = {
    "communication_light": communication_light,
    "rank_nonlocal_heavy": rank_nonlocal_heavy,
    "mpi_nonlocal_heavy": mpi_nonlocal_heavy,
    "mpi_nonlocal_phase_heavy": mpi_nonlocal_phase_heavy,
    "mpi_nonlocal_mixing_heavy": mpi_nonlocal_mixing_heavy,
    "mixed_staged": mixed_staged,
}


def build_circuit(kind: str, n: int, depth: int, chunk_bits: int,
                  num_ranks: int, seed: int = 42) -> dict:
    """Dispatch to a generator, supplying only the args it accepts."""
    if kind == "communication_light":
        return communication_light(n, depth, seed)
    if kind == "rank_nonlocal_heavy":
        return rank_nonlocal_heavy(n, depth, chunk_bits, seed, num_ranks=num_ranks)
    if kind == "mpi_nonlocal_heavy":
        return mpi_nonlocal_heavy(n, depth, chunk_bits, num_ranks, seed)
    if kind == "mpi_nonlocal_phase_heavy":
        return mpi_nonlocal_phase_heavy(n, depth, chunk_bits, num_ranks, seed)
    if kind == "mpi_nonlocal_mixing_heavy":
        return mpi_nonlocal_mixing_heavy(n, depth, chunk_bits, num_ranks, seed)
    if kind == "mixed_staged":
        return mixed_staged(n, depth, chunk_bits, num_ranks, seed)
    raise ValueError(f"unknown workload kind: {kind!r} (choices: {list(GENERATORS)})")


# ── static gate classification ─────────────────────────────────────────

def _check_pow2(num_ranks: int) -> None:
    if num_ranks < 1 or (num_ranks & (num_ranks - 1)) != 0:
        raise ValueError(f"num_ranks must be a power of 2, got {num_ranks}")


# ── Clifford / stabilizer classification ────────────────────────────────

# Fixed gates that are Clifford (generate the Clifford group with CNOT/CZ).
_CLIFFORD_FIXED = {"H", "X", "Y", "Z", "S", "SDG", "I",
                   "CNOT", "CX", "CZ", "CY", "SWAP"}
# Fixed gates that are NOT Clifford.
_NONCLIFFORD_FIXED = {"T", "TDG"}


def _angle_is_clifford(theta: float) -> bool:
    """True iff ``theta`` is an integer multiple of π/2 (within tolerance)."""
    r = theta / (math.pi / 2.0)
    return abs(r - round(r)) < 1e-9


def is_clifford_gate(gate: dict) -> bool:
    """Whether a single gate entry is a Clifford operation.

    Rotations ``RX/RY/RZ(θ)`` are Clifford only at multiples of π/2.  Phase
    gates by ``2π/2^k`` differ between the 1- and 2-qubit cases:

      * single-qubit ``R(k)`` = diag(1, e^{2πi/2^k}): R(1)=Z, R(2)=S are
        Clifford; R(k≥3) (T, …) is non-Clifford → Clifford iff ``k ≤ 2``.
      * controlled ``CR(k)`` = controlled-R(k): CR(1)=CZ is Clifford, but
        CR(2)=controlled-S is **non-Clifford** (a controlled-Clifford is not
        itself Clifford) → Clifford iff ``k ≤ 1``.

    Unknown gate names are treated as non-Clifford (the safe, conservative
    assumption for a stabilizer test).
    """
    name = str(gate["gate"]).upper()
    if name in _CLIFFORD_FIXED:
        return True
    if name in _NONCLIFFORD_FIXED:
        return False
    params = gate.get("params") or {}
    if name in ("RX", "RY", "RZ", "PHASE"):
        theta = params.get("theta")
        return theta is not None and _angle_is_clifford(float(theta))
    if name == "R":
        # single-qubit phase 2π/2^k: R(1)=Z, R(2)=S are Clifford.
        k = params.get("k")
        return k is not None and int(k) <= 2
    if name == "CR":
        # controlled phase: only CR(1)=CZ is Clifford; CR(2)=controlled-S is not.
        k = params.get("k")
        return k is not None and int(k) <= 1
    return False


def circuit_clifford_stats(circuit_dict: dict) -> dict:
    """Classify a circuit's gates as Clifford / non-Clifford.

    Returns total / Clifford / non-Clifford counts, the sorted set of
    non-Clifford gate-type names that appear, and whether the circuit is a
    stabilizer circuit (Clifford-only).  Used for both the reporting summary
    and the ``final_summary.json`` stabilizer fields.
    """
    gates = circuit_dict.get("gates", [])
    nc_types: dict[str, int] = {}
    clifford = 0
    for g in gates:
        if is_clifford_gate(g):
            clifford += 1
        else:
            name = str(g["gate"]).upper()
            nc_types[name] = nc_types.get(name, 0) + 1
    non_clifford = sum(nc_types.values())
    return {
        "total_gate_count": len(gates),
        "clifford_gate_count": clifford,
        "non_clifford_gate_count": non_clifford,
        "non_clifford_gate_types": sorted(nc_types),
        "non_clifford_gate_type_counts": dict(sorted(nc_types.items())),
        "is_stabilizer": non_clifford == 0,
    }


def classify_gate(qubits: list[int], k: int, n_local_bits: int) -> str:
    """Classify one gate as local / rank_nonlocal / mpi_nonlocal.

    Mirrors the logic in ``mpi_runner._compile_steps``: a gate is
    MPI-nonlocal if any qubit's partner bit reaches into the rank portion
    (``q - k >= n_local_bits``); rank-nonlocal if any qubit is a chunk bit
    but none reach the rank portion; otherwise local.
    """
    if all(q < k for q in qubits):
        return "local"
    if any((q - k) >= n_local_bits for q in qubits if q >= k):
        return "mpi_nonlocal"
    return "rank_nonlocal"


def _partner_pairs_for_gate(qubits: list[int], k: int, n_local_bits: int,
                            p: int) -> set[frozenset[int]]:
    """Unordered (rank, partner) pairs an MPI-nonlocal gate exchanges over.

    Static prediction only (used for ``estimated_partner_rank_pairs``); the
    reported ``partner_rank_pairs`` comes from observed traffic.
    """
    rank_bits = sorted({(q - k) - n_local_bits
                        for q in qubits if (q - k) >= n_local_bits})
    if not rank_bits:
        return set()
    num_ranks = 1 << p
    pairs: set[frozenset[int]] = set()
    for r in range(num_ranks):
        group = set()
        for combo in range(1 << len(rank_bits)):
            rr = r
            for i, b in enumerate(rank_bits):
                if combo & (1 << i):
                    rr ^= (1 << b)
            group.add(rr)
        group_list = sorted(group)
        for i in range(len(group_list)):
            for j in range(i + 1, len(group_list)):
                pairs.add(frozenset((group_list[i], group_list[j])))
    return pairs


def _layout(n: int, chunk_bits: int, num_ranks: int) -> tuple[int, int, int]:
    """Return (k, p, n_local_bits), validating the layout."""
    _check_pow2(num_ranks)
    k = chunk_bits
    p = _rank_bits(num_ranks)
    n_local_bits = n - k - p
    if n_local_bits < 0:
        raise ValueError(
            f"invalid layout: n - chunk_bits - log2(num_ranks) = "
            f"{n_local_bits} < 0 (n={n}, chunk_bits={k}, num_ranks={num_ranks})")
    return k, p, n_local_bits


def classify_mpi_gates(circuit_dict: dict, chunk_bits: int,
                       num_ranks: int) -> dict:
    """Classify the circuit's MPI-nonlocal gates: diagonal / mixing / permutation.

    Diagonal MPI gates need no remote-amplitude exchange (the diagonal fast
    path applies them locally); true-mixing and permutation gates do.  Lets
    final_summary distinguish skipped diagonal MPI from genuine MPI stress.
    """
    from wenbo_engine.kernel import gates as _g
    from wenbo_engine.mpi.diagonal_nonlocal import classify_nonlocal_gate
    k, p, n_local_bits = _layout(circuit_dict["number_of_qubits"]
                                 if "number_of_qubits" in circuit_dict else 0,
                                 chunk_bits, num_ranks)
    diag = mixing = perm = req = 0
    for gt in circuit_dict["gates"]:
        qs = gt["qubits"]
        if classify_gate(qs, k, n_local_bits) != "mpi_nonlocal":
            continue
        U = _g.gate_matrix(gt["gate"], gt.get("params", {}))
        kind, requires_remote = classify_nonlocal_gate(U)
        if kind == "diagonal":
            diag += 1
        elif kind == "permutation":
            perm += 1
        else:
            mixing += 1
        if requires_remote:
            req += 1
    return {
        "diagonal_mpi_nonlocal_gate_count": diag,
        "true_mixing_mpi_nonlocal_gate_count": mixing,
        "permutation_mpi_nonlocal_gate_count": perm,
        "requires_remote_amplitudes_gate_count": req,
        "skipped_mpi_exchange_gate_count": diag,   # diagonal gates skip exchange
    }


def classify_circuit(circuit_dict: dict, chunk_bits: int,
                     num_ranks: int) -> dict:
    """Statically count gate classes and the communication topology.

    This is the *static* (predicted) classification.  For the authoritative
    counts produced by the real runner compiler use
    :func:`runner_classification`.
    """
    n = circuit_dict["number_of_qubits"]
    k, p, n_local_bits = _layout(n, chunk_bits, num_ranks)

    counts = {"local": 0, "rank_nonlocal": 0, "mpi_nonlocal": 0}
    pairs: set[frozenset[int]] = set()
    for g in circuit_dict["gates"]:
        cls = classify_gate(g["qubits"], k, n_local_bits)
        counts[cls] += 1
        if cls == "mpi_nonlocal":
            pairs |= _partner_pairs_for_gate(g["qubits"], k, n_local_bits, p)

    return {
        "n": n,
        "chunk_bits": k,
        "num_ranks": num_ranks,
        "n_local_bits": n_local_bits,
        "local_gate_count": counts["local"],
        "rank_nonlocal_gate_count": counts["rank_nonlocal"],
        "mpi_nonlocal_gate_count": counts["mpi_nonlocal"],
        "partner_rank_pairs": len(pairs),
        "partner_rank_pair_set": sorted(sorted(pr) for pr in pairs),
    }


def runner_classification(circuit_dict: dict, chunk_bits: int,
                          num_ranks: int) -> dict:
    """Authoritative gate-class counts from the REAL runner compiler.

    Calls ``mpi_runner._compile_steps`` (the exact code path the runner
    uses) on the levelized circuit and sums the op classes.  This is what
    the runner will actually do — not a re-implementation.
    """
    from wenbo_engine.circuit.io import validate_circuit_dict, levelize
    from wenbo_engine.mpi.mpi_runner import _compile_steps

    n = circuit_dict["number_of_qubits"]
    k, p, n_local_bits = _layout(n, chunk_bits, num_ranks)
    cd = validate_circuit_dict(circuit_dict)
    levels = levelize(cd)
    steps = _compile_steps(levels, k, n_local_bits)
    local = sum(len(s["local_ops"]) for s in steps)
    rank_nl = sum(len(s["rank_nonlocal_ops"]) for s in steps)
    mpi_nl = sum(len(s["mpi_nonlocal_ops"]) for s in steps)
    return {
        "n": n, "chunk_bits": k, "num_ranks": num_ranks,
        "n_local_bits": n_local_bits, "n_steps": len(steps),
        "local_ops": local, "rank_nonlocal_ops": rank_nl,
        "mpi_nonlocal_ops": mpi_nl,
    }


# ── runtime metrics (all MEASURED) ─────────────────────────────────────

@dataclass
class Metrics:
    """Per-rank runtime counters, accumulated during a single workload run.

    A fresh instance is created per :func:`run_workload` call, so counters
    never leak between sequential runs.  All increment paths take ``lock``
    because the MPI runner reads/writes chunks and applies kernels from
    background pipeline threads.
    """
    mpi_bytes_sent: int = 0
    mpi_sendrecv_time: float = 0.0
    mpi_wait_time: float = 0.0
    sendrecv_count: int = 0
    barrier_count: int = 0
    bytes_read: int = 0
    bytes_written: int = 0
    read_ops: int = 0          # # of chunk-file read_chunk calls
    write_ops: int = 0         # # of chunk-file write_chunk_atomic calls (== temp chunk files)
    read_sec: float = 0.0
    write_sec: float = 0.0
    kernel_time: float = 0.0
    local_kernel_time: float = 0.0
    nonlocal_kernel_time: float = 0.0
    local_gates_applied: int = 0
    nonlocal_gates_applied: int = 0
    norm_time: float = 0.0
    stage_time: float = 0.0
    # measured communication partners and per-call exchange events
    observed_partner_pairs: set = field(default_factory=set)
    sendrecv_events: list = field(default_factory=list)  # [peer, bytes, seconds]
    lock: Lock = field(default_factory=Lock, repr=False, compare=False)

    def to_dict(self) -> dict:
        # Built by hand (not dataclasses.asdict) because the Lock field is
        # not picklable, and the result is sent over MPI.
        return {
            "mpi_bytes_sent": self.mpi_bytes_sent,
            "mpi_sendrecv_time": self.mpi_sendrecv_time,
            "mpi_wait_time": self.mpi_wait_time,
            "sendrecv_count": self.sendrecv_count,
            "barrier_count": self.barrier_count,
            "bytes_read": self.bytes_read,
            "bytes_written": self.bytes_written,
            "read_ops": self.read_ops,
            "write_ops": self.write_ops,
            "read_sec": self.read_sec,
            "write_sec": self.write_sec,
            "kernel_time": self.kernel_time,
            "local_kernel_time": self.local_kernel_time,
            "nonlocal_kernel_time": self.nonlocal_kernel_time,
            "local_gates_applied": self.local_gates_applied,
            "nonlocal_gates_applied": self.nonlocal_gates_applied,
            "norm_time": self.norm_time,
            "stage_time": self.stage_time,
            "observed_partner_pairs": sorted(
                sorted(pr) for pr in self.observed_partner_pairs),
            "sendrecv_events": list(self.sendrecv_events),
        }


class ProfilingComm:
    """Transparent proxy around an MPI communicator that times Sendrecv.

    Everything except ``Sendrecv`` / ``Barrier`` is forwarded unchanged to
    the wrapped communicator via ``__getattr__``.  ``Sendrecv`` is timed
    and its bytes counted **from the actual send buffer**, and the actual
    partner rank is recorded.  ``Barrier`` time is attributed to
    ``mpi_wait_time`` (the runner uses blocking Sendrecv, so synchronization
    cost shows up at the barriers).
    """

    def __init__(self, comm, metrics: Metrics):
        self._comm = comm
        self._m = metrics
        self._rank = comm.Get_rank()

    def __getattr__(self, name):
        return getattr(self._comm, name)

    def Sendrecv(self, sendbuf, dest, recvbuf, source, **kw):
        nbytes = int(np.asarray(sendbuf).nbytes)
        t0 = time.perf_counter()
        r = self._comm.Sendrecv(sendbuf=sendbuf, dest=dest,
                                recvbuf=recvbuf, source=source, **kw)
        dt = time.perf_counter() - t0
        with self._m.lock:
            self._m.mpi_bytes_sent += nbytes
            self._m.mpi_sendrecv_time += dt
            self._m.sendrecv_count += 1
            self._m.observed_partner_pairs.add(frozenset((self._rank, int(dest))))
            self._m.sendrecv_events.append([int(dest), nbytes, dt])
        return r

    def Barrier(self):
        t0 = time.perf_counter()
        r = self._comm.Barrier()
        with self._m.lock:
            self._m.mpi_wait_time += time.perf_counter() - t0
            self._m.barrier_count += 1
        return r


@contextlib.contextmanager
def _instrument_runner(metrics: Metrics):
    """Monkeypatch the MPI runner's chunk I/O and kernel calls to measure.

    The kernels themselves stay pure — we wrap the *names the runner
    resolves* (module globals), so no kernel, I/O, WAL, or MPI source is
    edited.  Originals are restored on exit even if the run raises
    (try/finally), so nothing leaks into a subsequent run.
    """
    from wenbo_engine.mpi import mpi_runner as mr

    io_names = ("read_chunk", "write_chunk_atomic")
    kernel_names = ("apply_1q", "apply_2q", "apply_1q_pair",
                    "apply_2q_pair_qa_local", "apply_2q_pair_qb_local",
                    "apply_2q_quad")
    saved = {name: getattr(mr, name) for name in io_names + kernel_names}

    def wrap_read(orig):
        def w(path):
            t0 = time.perf_counter()
            data = orig(path)
            dt = time.perf_counter() - t0
            with metrics.lock:
                metrics.bytes_read += int(np.asarray(data).nbytes)
                metrics.read_ops += 1
                metrics.read_sec += dt
            return data
        return w

    def wrap_write(orig):
        def w(path, data):
            nbytes = int(np.asarray(data).nbytes)
            t0 = time.perf_counter()
            r = orig(path, data)
            dt = time.perf_counter() - t0
            with metrics.lock:
                metrics.bytes_written += nbytes
                metrics.write_ops += 1
                metrics.write_sec += dt
            return r
        return w

    # local (within-chunk) vs nonlocal (partner-chunk) kernels — split so the
    # cost model can calibrate them separately.
    _local_kernels = {"apply_1q", "apply_2q"}

    def wrap_kernel(orig, name):
        is_local = name in _local_kernels
        def w(*a, **k):
            t0 = time.perf_counter()
            r = orig(*a, **k)
            dt = time.perf_counter() - t0
            with metrics.lock:
                metrics.kernel_time += dt
                if is_local:
                    metrics.local_kernel_time += dt
                    metrics.local_gates_applied += 1
                else:
                    metrics.nonlocal_kernel_time += dt
                    metrics.nonlocal_gates_applied += 1
            return r
        return w

    # The compute-unit overlay resolves read_chunk/write_chunk_atomic in its
    # own module namespace, so instrument those too (else overlay I/O would be
    # invisible and byte metrics would be wrong in compute_unit mode).
    try:
        from wenbo_engine.runtime import memory_overlay as _ov
    except Exception:
        _ov = None
    ov_saved = {}
    if _ov is not None:
        ov_saved = {n: getattr(_ov, n) for n in io_names}

    # The window executor resolves read_chunk/write_chunk_atomic in its own
    # namespace; instrument those too so window disk I/O is measured (else
    # nvme bandwidth can't be calibrated on a window run).
    try:
        from wenbo_engine.mpi import window_executor as _we
    except Exception:
        _we = None
    we_saved = {}
    if _we is not None:
        we_saved = {n: getattr(_we, n) for n in io_names if hasattr(_we, n)}

    # The gate-aware MPI path runs the nonlocal *pair* kernels inside
    # exchange_batch's namespace; instrument those so nonlocal_kernel_time is
    # measured on gate_aware runs (the naive path uses mr's names, already wrapped).
    try:
        from wenbo_engine.mpi import exchange_batch as _eb
    except Exception:
        _eb = None
    eb_kernel_names = ("apply_1q_pair", "apply_2q_pair_qa_local",
                       "apply_2q_pair_qb_local")
    eb_saved = {}
    if _eb is not None:
        eb_saved = {n: getattr(_eb, n) for n in eb_kernel_names
                    if hasattr(_eb, n)}

    try:
        mr.read_chunk = wrap_read(saved["read_chunk"])
        mr.write_chunk_atomic = wrap_write(saved["write_chunk_atomic"])
        for name in kernel_names:
            setattr(mr, name, wrap_kernel(saved[name], name))
        if _ov is not None:
            _ov.read_chunk = wrap_read(ov_saved["read_chunk"])
            _ov.write_chunk_atomic = wrap_write(ov_saved["write_chunk_atomic"])
        if _we is not None and we_saved:
            _we.read_chunk = wrap_read(we_saved["read_chunk"])
            _we.write_chunk_atomic = wrap_write(we_saved["write_chunk_atomic"])
        for name, fn in eb_saved.items():
            setattr(_eb, name, wrap_kernel(fn, name))
        yield
    finally:
        for name, fn in saved.items():
            setattr(mr, name, fn)
        for name, fn in ov_saved.items():
            setattr(_ov, name, fn)
        for name, fn in we_saved.items():
            setattr(_we, name, fn)
        for name, fn in eb_saved.items():
            setattr(_eb, name, fn)


# ── run + aggregate ────────────────────────────────────────────────────

def _runner_supports_recovery() -> bool:
    """True when the merged mpi_runner.run accepts a ``recovery`` parameter."""
    from wenbo_engine.mpi.mpi_runner import run
    return "recovery" in inspect.signature(run).parameters


def _durable_promote(work_dir, comm, durable: dict) -> dict | None:
    """Promote the final committed generation to durable storage (rank 0 dict).

    Separate, explicit step run after the runner returns — never on the hot
    path.  Uses the durable package's promotion protocol.
    """
    from wenbo_engine.durable import DurableConfig, DurableCheckpointManager
    from wenbo_engine.recovery.generation_manager import MPICoordinator
    from wenbo_engine.recovery.recovery_scanner import RecoveryScanner

    run_id = Path(work_dir).name or "comm_workload"
    dconf = DurableConfig.from_dict(durable)
    backend = dconf.build_backend()
    coord = MPICoordinator(comm)
    cm = DurableCheckpointManager(work_dir, run_id, backend, coord)
    cm.upload_run_metadata()

    final_gen = RecoveryScanner(work_dir).scan(quarantine=False).generation
    final_gen = comm.bcast(final_gen, root=0)
    promoted = []
    if final_gen is not None:
        rec = cm.promote(final_gen)
        if rec is not None:
            promoted.append(final_gen)
    if comm.Get_rank() == 0:
        return {"promoted_generations": promoted, "run_id": run_id,
                "backend": dconf.backend, "root": dconf.root}
    return None


def run_workload(kind: str, n: int, depth: int, chunk_bits: int,
                 work_dir: str | Path, comm=None, seed: int = 42,
                 use_wal: bool = True, verify: bool = False,
                 reorder: bool = False, recovery: str | None = None,
                 output_dir: str | Path | None = None,
                 durable: dict | None = None,
                 planner: str | None = None,
                 mpi_exchange_mode: str = "naive",
                 storage_layout: str = "chunks",
                 execution_mode: str = "step",
                 compute_unit_min_gates: int = 4,
                 extent_io_mode: str = "materialize",
                 ram_budget_gib: float | None = None,
                 auto_chunk_bits: bool = False,
                 max_overlay_chunks: int | None = None,
                 max_remote_buffer_gib: float | None = None,
                 kernel_backend: str = "auto",
                 mpi_window_analysis: str = "off",
                 mpi_window_execution: str = "off") -> dict:
    """Run a communication workload under instrumentation.

    Builds the circuit, optionally applies production reordering
    (``reorder=True``; off by default to preserve MPI stress), runs it on
    the MPI runner with a ``ProfilingComm`` and patched I/O/kernels, gathers
    per-rank measured metrics onto rank 0, and (if ``output_dir`` is given)
    writes the standard artifact bundle.  Returns the result dict on rank 0,
    ``None`` elsewhere.

    ``recovery`` selects the durability mode (``none`` / ``wal`` /
    ``generation``).  ``generation`` requires the generation-recovery
    integration; if that is not merged a clear ``RuntimeError`` is raised.
    When ``recovery`` is ``None`` it is derived from ``use_wal`` (back-compat).
    """
    from mpi4py import MPI
    from wenbo_engine.mpi.mpi_runner import run, collect_state, compute_norm

    if comm is None:
        comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_ranks = comm.Get_size()

    if recovery is None:
        recovery = "wal" if use_wal else "none"
    if recovery not in RECOVERY_MODES:
        raise ValueError(f"unknown recovery mode {recovery!r} "
                         f"(expected {'|'.join(RECOVERY_MODES)})")

    chunk_size = 1 << chunk_bits
    cd = build_circuit(kind, n, depth, chunk_bits, num_ranks, seed)

    if reorder:
        from wenbo_engine.circuit.reorder import reorder_qubits
        cd, _perm = reorder_qubits(cd)

    # Optimizer v2: when a --planner mode is selected, apply the STATIC
    # circuit transform the mode implies before running (the MPI runner
    # levelizes internally, so a static qubit relabelling is what changes
    # its execution).  The deterministic ablation report is written to the
    # artifact bundle regardless.  ``current`` / None are behavior-preserving.
    planner_perm = None
    recovery_aware_plan = None
    # Resolve the RAM working-set budget (70% of node RAM if not given) when any
    # RAM-aware control is active.
    ram_budget_resolved = None
    if auto_chunk_bits or ram_budget_gib is not None:
        ram_budget_resolved = (ram_budget_gib if ram_budget_gib is not None
                               else round(_available_ram_gib() * 0.70, 3))
    if planner == "recovery_aware_v1":
        # Recovery-aware planner v1: evaluate candidate strategies and SELECT
        # the run's (storage_layout, execution_mode, extent_io_mode,
        # mpi_exchange_mode, compute_unit_min_gates).  Deterministic, so every
        # rank computes the same selection independently and runs consistently.
        # It does NOT relabel qubits (no reorder → MPI stress is preserved).
        from wenbo_engine.planner import plan_recovery_aware, selected_run_params
        recovery_aware_plan = plan_recovery_aware(
            cd, n=n, chunk_bits=chunk_bits, num_ranks=num_ranks,
            recovery=recovery, compute_unit_min_gates=compute_unit_min_gates,
            ram_budget_gib=ram_budget_resolved,
            max_overlay_chunks=max_overlay_chunks,
            max_remote_buffer_gib=max_remote_buffer_gib,
            auto_chunk_bits=auto_chunk_bits)
        # RAM-aware auto chunk_bits: adopt the recommended chunk_bits and
        # re-plan (the circuit's gate placement depends on chunk_bits, so the
        # circuit is rebuilt too).
        if auto_chunk_bits:
            rec_cb = recovery_aware_plan["ram"].get("recommended_chunk_bits")
            if rec_cb is None:
                raise RuntimeError(
                    f"RAM-infeasible: no chunk_bits fits the {ram_budget_resolved} "
                    f"GiB/rank budget for {kind} n={n} on {num_ranks} ranks "
                    f"(selected {recovery_aware_plan['decision']['selected_strategy']['name']}).")
            if rec_cb != chunk_bits:
                chunk_bits = rec_cb
                chunk_size = 1 << chunk_bits
                cd = build_circuit(kind, n, depth, chunk_bits, num_ranks, seed)
                recovery_aware_plan = plan_recovery_aware(
                    cd, n=n, chunk_bits=chunk_bits, num_ranks=num_ranks,
                    recovery=recovery,
                    compute_unit_min_gates=compute_unit_min_gates,
                    ram_budget_gib=ram_budget_resolved,
                    max_overlay_chunks=max_overlay_chunks,
                    max_remote_buffer_gib=max_remote_buffer_gib,
                    auto_chunk_bits=auto_chunk_bits)
        _rp = selected_run_params(recovery_aware_plan)
        storage_layout = _rp["storage_layout"]
        execution_mode = _rp["execution_mode"]
        extent_io_mode = _rp["extent_io_mode"]
        mpi_exchange_mode = _rp["mpi_exchange_mode"]
        compute_unit_min_gates = _rp["compute_unit_min_gates"]
        if rank == 0:
            log.info("  recovery_aware_v1 selected: %s (chunk_bits=%d)",
                     recovery_aware_plan["decision"]["selected_strategy"]["name"],
                     chunk_bits)
    elif planner is not None and planner != "current":
        from wenbo_engine.planner import ABLATION_MODES
        from wenbo_engine.planner.placement_planner import (
            plan_placement, apply_placement,
        )
        from wenbo_engine.planner.qubit_activity import qubit_activity
        if planner not in ABLATION_MODES:
            raise ValueError(f"unknown --planner mode {planner!r} "
                             f"(choices: {ABLATION_MODES})")
        if planner == "current_static_reorder":
            from wenbo_engine.circuit.reorder import reorder_qubits
            cd, planner_perm = reorder_qubits(cd)
        elif planner == "stage_v2_placement_fusion":
            p = _rank_bits(num_ranks) if num_ranks >= 1 else 0
            planner_perm = plan_placement(cd, k=chunk_bits, p=p,
                                          activity=qubit_activity(cd))
            cd = apply_placement(cd, planner_perm)
        # stage_v2 / stage_v2_fusion only change the single-node staging
        # schedule, not the MPI runner's levelized execution — no static
        # relabelling, run as-is (the report still captures the gains).

    metrics = Metrics()
    pcomm = ProfilingComm(comm, metrics)

    comm.Barrier()
    t0 = time.perf_counter()
    with _instrument_runner(metrics):
        if _runner_supports_recovery():
            run(cd, work_dir, chunk_size=chunk_size, comm=pcomm,
                recovery=recovery, mpi_exchange_mode=mpi_exchange_mode,
                storage_layout=storage_layout, execution_mode=execution_mode,
                compute_unit_min_gates=compute_unit_min_gates,
                extent_io_mode=extent_io_mode,
                ram_budget_gib=ram_budget_resolved,
                max_overlay_chunks=max_overlay_chunks,
                max_remote_buffer_gib=max_remote_buffer_gib,
                auto_chunk_bits=auto_chunk_bits,
                kernel_backend=kernel_backend,
                mpi_window_execution=mpi_window_execution)
        elif recovery == "generation":
            raise RuntimeError(
                "recovery='generation' requires the generation-recovery "
                "integration (mpi_runner.run has no 'recovery' parameter on "
                "this branch). Run on a branch that merges Agent 2.")
        else:
            run(cd, work_dir, chunk_size=chunk_size, comm=pcomm,
                use_wal=(recovery == "wal"),
                mpi_exchange_mode=mpi_exchange_mode)
    metrics.stage_time = time.perf_counter() - t0
    comm.Barrier()

    # Compute-unit / overlay profiling (written by the runner in compute_unit
    # mode); measured, surfaced into the result + final_summary.
    overlay_metrics = {}
    om_path = Path(work_dir) / "overlay_metrics.json"
    if om_path.exists():
        try:
            overlay_metrics = json.loads(om_path.read_text())
        except (OSError, ValueError):
            overlay_metrics = {}

    # RAM-aware execution metrics (budgets + measured peaks), written by run().
    kernel_metrics = {}
    km_path = Path(work_dir) / "kernel_metrics.json"
    if km_path.exists():
        try:
            kernel_metrics = json.loads(km_path.read_text())
        except (OSError, ValueError):
            kernel_metrics = {}

    mpi_telemetry = {}
    mt_path = Path(work_dir) / "mpi_telemetry.json"
    if mt_path.exists():
        try:
            mpi_telemetry = json.loads(mt_path.read_text())
        except (OSError, ValueError):
            mpi_telemetry = {}

    # MPI true-mixing window-executor metrics (written by run() when enabled).
    mpi_window_exec = {}
    we_path = Path(work_dir) / "mpi_window_exec.json"
    if we_path.exists():
        try:
            mpi_window_exec = json.loads(we_path.read_text())
        except (OSError, ValueError):
            mpi_window_exec = {}

    # Runtime timing accumulators (commit/extent/materialize/pack/rss), written
    # by run() via runtime_timers.
    timing_metrics = {}
    tm_path = Path(work_dir) / "timing_metrics.json"
    if tm_path.exists():
        try:
            timing_metrics = json.loads(tm_path.read_text())
        except (OSError, ValueError):
            timing_metrics = {}

    ram_metrics = {}
    rm_path = Path(work_dir) / "ram_metrics.json"
    if rm_path.exists():
        try:
            ram_metrics = json.loads(rm_path.read_text())
        except (OSError, ValueError):
            ram_metrics = {}

    # Durable R4 (optional): promote committed generations AFTER the run — a
    # separate, explicit step, never during gate execution.  Only valid in
    # generation recovery mode.
    durable_promotion = None
    if durable and durable.get("enabled") and recovery == "generation":
        durable_promotion = _durable_promote(work_dir, comm, durable)

    # Global norm (collective — all ranks participate).  Timed for the cost
    # model's norm-scan bandwidth calibration.
    _norm_t0 = time.perf_counter()
    norm = compute_norm(work_dir, comm)
    metrics.norm_time += time.perf_counter() - _norm_t0

    correct = None
    if verify:
        from wenbo_engine.kernel.ref_dense import simulate
        got = collect_state(work_dir, comm)
        if rank == 0:
            # cd is the (possibly reordered) circuit actually run, so the
            # reference is computed on the same circuit — no extra permute.
            ref = simulate(cd)
            correct = bool(np.allclose(got, ref, atol=1e-5))

    all_metrics = comm.gather(metrics.to_dict(), root=0)
    if rank != 0:
        return None

    aggregate = _aggregate(all_metrics)
    # ── assemble the measured-timing dict (single source for final_summary,
    # cost_report timing block, and calibration) ──
    _km0 = {}
    _kmpath = Path(work_dir) / "kernel_metrics.json"
    if _kmpath.exists():
        try:
            _km0 = json.loads(_kmpath.read_text())
        except (OSError, ValueError):
            _km0 = {}
    _state_bytes = (1 << n) * int(np.dtype(np.complex64).itemsize)
    measured_timing = {
        "local_kernel_time": aggregate.get("local_kernel_time"),
        "nonlocal_kernel_time": aggregate.get("nonlocal_kernel_time"),
        "mpi_pairwise_sendrecv_time": aggregate.get("mpi_pairwise_sendrecv_time"),
        "mpi_collective_gather_time": mpi_window_exec.get("mpi_collective_gather_time"),
        "mpi_collective_scatter_time": mpi_window_exec.get("mpi_collective_scatter_time"),
        "mpi_window_leader_compute_time": mpi_window_exec.get("mpi_window_leader_compute_time"),
        "mpi_window_segment_time": mpi_window_exec.get("mpi_window_segment_time"),
        "direct_extent_read_time": timing_metrics.get("direct_extent_read_time"),
        "direct_extent_write_time": timing_metrics.get("direct_extent_write_time"),
        "extent_materialize_time": timing_metrics.get("extent_materialize_time"),
        "extent_pack_time": timing_metrics.get("extent_pack_time"),
        "commit_time": timing_metrics.get("commit_time"),
        "norm_time": aggregate.get("norm_time"),
        "numba_compile_time": _km0.get("numba_compile_time"),
        "rss_peak_gib": timing_metrics.get("rss_peak_gib"),
        "rss_peak_gib_reason": timing_metrics.get("rss_peak_gib_reason"),
        # raw byte/gate/count inputs for calibration
        "bytes_read": aggregate.get("bytes_read"),
        "read_sec": aggregate.get("read_sec"),
        "bytes_written": aggregate.get("bytes_written"),
        "write_sec": aggregate.get("write_sec"),
        "mpi_bytes_sent": aggregate.get("mpi_bytes_sent"),
        "gather_bytes": mpi_window_exec.get("mpi_window_gather_bytes"),
        "gather_time": mpi_window_exec.get("mpi_collective_gather_time"),
        "scatter_bytes": mpi_window_exec.get("mpi_window_scatter_bytes"),
        "scatter_time": mpi_window_exec.get("mpi_collective_scatter_time"),
        "local_gates_applied": aggregate.get("local_gates_applied"),
        "nonlocal_gates_applied": aggregate.get("nonlocal_gates_applied"),
        "commit_count": timing_metrics.get("commit_count_runtime"),
        "state_bytes": _state_bytes,
        "numba_speedup_factor": None,   # needs numpy-vs-numba A/B
    }
    runner_cls = runner_classification(cd, chunk_bits, num_ranks)
    mpi_gate_classes = classify_mpi_gates(cd, chunk_bits, num_ranks)
    static = classify_circuit(cd, chunk_bits, num_ranks)
    clifford = circuit_clifford_stats(cd)

    # Deterministic Optimizer-v2 ablation report over the ORIGINAL circuit
    # (before any planner/reorder transform), so all modes compare fairly.
    ablation = None
    try:
        from wenbo_engine.planner import HardwareConfig, ablation_report
        orig_cd = build_circuit(kind, n, depth, chunk_bits, num_ranks, seed)
        hw = HardwareConfig(n_qubits=n, chunk_bits=chunk_bits,
                            num_ranks=num_ranks,
                            recovery=recovery if recovery != "none" else "none")
        ablation = ablation_report(orig_cd, hw, verify_norm=False)
    except Exception as e:  # pragma: no cover - defensive
        ablation = {"error": repr(e)}

    result = {
        "workload_kind": kind,
        "kind": kind,  # back-compat alias
        "n": n,
        "depth": depth,
        "seed": seed,
        "chunk_bits": chunk_bits,
        "num_ranks": num_ranks,
        "use_wal": use_wal,
        "recovery_mode": recovery,
        "reorder_applied": reorder,
        "planner_mode": planner or "current",
        "mpi_exchange_mode": mpi_exchange_mode,
        "storage_layout": storage_layout,
        "execution_mode": execution_mode,
        "extent_io_mode": extent_io_mode,
        "overlay_metrics": overlay_metrics,
        "ram_metrics": ram_metrics,
        "kernel_metrics": kernel_metrics,
        "mpi_telemetry": mpi_telemetry,
        "mpi_window_exec": mpi_window_exec,
        "timing_metrics": timing_metrics,
        "measured_timing": measured_timing,
        "mpi_window_execution_mode": mpi_window_execution,
        "mpi_gate_classes": mpi_gate_classes,
        "kernel_backend_requested": kernel_backend,
        "auto_chunk_bits_enabled": bool(auto_chunk_bits),
        "intended_locality": INTENDED_LOCALITY.get(kind, "unknown"),
        # Clifford / stabilizer analysis of the generated circuit
        "total_gate_count": clifford["total_gate_count"],
        "clifford_gate_count": clifford["clifford_gate_count"],
        "non_clifford_gate_count": clifford["non_clifford_gate_count"],
        "non_clifford_gate_types": clifford["non_clifford_gate_types"],
        "non_clifford_gate_type_counts": clifford["non_clifford_gate_type_counts"],
        "is_stabilizer": clifford["is_stabilizer"],
        "n_local_bits": runner_cls["n_local_bits"],
        "n_steps": runner_cls["n_steps"],
        # authoritative counts from the real runner compiler
        "measured_local_ops": runner_cls["local_ops"],
        "measured_rank_nonlocal_ops": runner_cls["rank_nonlocal_ops"],
        "measured_mpi_nonlocal_ops": runner_cls["mpi_nonlocal_ops"],
        # back-compat names (also from the real runner, not estimates)
        "local_gate_count": runner_cls["local_ops"],
        "rank_nonlocal_gate_count": runner_cls["rank_nonlocal_ops"],
        "mpi_nonlocal_gate_count": runner_cls["mpi_nonlocal_ops"],
        # MEASURED communication partners (actual traffic)
        "partner_rank_pairs": aggregate["partner_rank_pairs"],
        "partner_rank_pair_set": aggregate["partner_rank_pair_set"],
        # static prediction, clearly labelled as an estimate
        "estimated_partner_rank_pairs": static["partner_rank_pairs"],
        "final_norm": norm,
        "aggregate": aggregate,
        "per_rank": all_metrics,
        "ablation_report": ablation,
    }
    if correct is not None:
        result["correct"] = correct
    if durable_promotion is not None:
        result["durable_promotion"] = durable_promotion

    # Recovery-aware planner v1: compare the SELECTED candidate's predicted
    # metrics against the measured run, and attach the decision + cost report.
    if recovery_aware_plan is not None:
        from wenbo_engine.planner import build_cost_report
        # Byte metrics at a CONSISTENT cluster aggregation level.  The
        # instrumented agg.bytes_* are summed across all ranks (true cluster
        # total) for the step / materialize paths.  The direct extent path is
        # not chunk-file-instrumented, so agg.bytes_* is 0 there; fall back to
        # the rank-0 overlay bytes scaled by num_ranks (partitions are
        # homogeneous), and expose the per-rank value too — never compare a
        # cluster prediction against a rank-0 actual unlabelled.
        _ov_r0_read = int(overlay_metrics.get("overlay_bytes_read") or 0)
        _ov_r0_written = int(overlay_metrics.get("overlay_bytes_written") or 0)
        if aggregate["bytes_read"] > 0:
            cl_read = aggregate["bytes_read"]
            cl_written = aggregate["bytes_written"]
        else:
            cl_read = _ov_r0_read * num_ranks
            cl_written = _ov_r0_written * num_ranks
        actual = {
            "bytes_read_cluster": cl_read,
            "bytes_written_cluster": cl_written,
            "bytes_read_per_rank": cl_read // num_ranks if num_ranks else cl_read,
            "bytes_written_per_rank": (cl_written // num_ranks
                                       if num_ranks else cl_written),
            "read_ops": aggregate["read_ops"],
            "write_ops": aggregate["write_ops"],
            "temporary_chunk_files_created": aggregate["write_ops"],
            "mpi_bytes_sent": aggregate["mpi_bytes_sent"],
            "sendrecv_count": aggregate["sendrecv_count"],
            "kernel_time": aggregate["kernel_time"],
            "commit_count": _count_commit_records(work_dir),
            "num_ranks": num_ranks,
            "wall_time": aggregate["stage_time"],
            "work_time": (aggregate["read_sec"] + aggregate["write_sec"]
                          + aggregate["kernel_time"]
                          + aggregate["mpi_sendrecv_time"]),
            **measured_timing,   # timing fields for the cost-report timing block
        }
        cost_report = build_cost_report(recovery_aware_plan, actual)
        result["planner_mode"] = "recovery_aware_v1"
        result["selected_strategy"] = \
            recovery_aware_plan["decision"]["selected_strategy"]["name"]
        result["recovery_aware"] = {
            "plan": recovery_aware_plan,
            "actual_cost_after_run": actual,
            "cost_report": cost_report,
        }

    # Calibrated cost-model constants derived from THIS run's measured timing
    # (null-with-reason where a primitive was not exercised).
    from wenbo_engine.planner.stage_cost_model import build_calibration
    result["calibration"] = build_calibration(measured_timing)

    # MPI-window feasibility analysis (rank 0, pure, post-run).  OFF by
    # default; never touches execution / recovery / remote cache.  Reads the
    # measured telemetry only to report the true baseline alongside the
    # analytic prediction.
    if mpi_window_analysis == "report":
        from wenbo_engine.planner.mpi_window_report import build_window_report
        result["mpi_window_analysis"] = build_window_report(
            cd, chunk_bits, num_ranks,
            ram_budget_gib=ram_budget_resolved,
            measured={"sendrecv_count": aggregate["sendrecv_count"],
                      "mpi_bytes_sent": aggregate["mpi_bytes_sent"]},
            mpi_telemetry=mpi_telemetry)
    else:
        result["mpi_window_analysis"] = {"analysis_mode": "off"}

    if output_dir is not None:
        write_artifacts(output_dir, result, cd, work_dir=work_dir)

    return result


def _aggregate(all_metrics: list[dict]) -> dict:
    """Combine per-rank measured metrics: sum counters/bytes, max wall times."""
    def s(key):
        return sum(m[key] for m in all_metrics)

    def mx(key):
        return max(m[key] for m in all_metrics)

    pair_set: set[frozenset[int]] = set()
    for m in all_metrics:
        for pr in m["observed_partner_pairs"]:
            pair_set.add(frozenset(pr))

    return {
        "mpi_bytes_sent": s("mpi_bytes_sent"),
        "mpi_sendrecv_time": mx("mpi_sendrecv_time"),
        "mpi_wait_time": mx("mpi_wait_time"),
        "sendrecv_count": s("sendrecv_count"),
        "barrier_count": mx("barrier_count"),
        "bytes_read": s("bytes_read"),
        "bytes_written": s("bytes_written"),
        "read_ops": s("read_ops"),
        "write_ops": s("write_ops"),
        "read_sec": mx("read_sec"),
        "write_sec": mx("write_sec"),
        "kernel_time": mx("kernel_time"),
        # local/nonlocal kernel time + gate counts: SUM across ranks (so the
        # cost model's gates/sec uses the cluster's total work and total time).
        "local_kernel_time": s("local_kernel_time"),
        "nonlocal_kernel_time": s("nonlocal_kernel_time"),
        "local_gates_applied": s("local_gates_applied"),
        "nonlocal_gates_applied": s("nonlocal_gates_applied"),
        "mpi_pairwise_sendrecv_time": s("mpi_sendrecv_time"),
        "norm_time": mx("norm_time"),
        "stage_time": mx("stage_time"),
        "partner_rank_pairs": len(pair_set),
        "partner_rank_pair_set": sorted(sorted(pr) for pr in pair_set),
    }


# ── artifact bundle (observability-compatible) ─────────────────────────

def _git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _wal_present(work_dir: str | Path | None) -> bool:
    if work_dir is None:
        return False
    return bool(list(Path(work_dir).glob("**/wal.json")))


def _count_commit_records(work_dir: str | Path | None) -> int:
    """Count global commit records written (one per committed generation)."""
    if work_dir is None:
        return 0
    return len(list(Path(work_dir).glob("**/commits/commit_*.json")))


def _recovery_events(work_dir: str | Path | None, recovery: str) -> dict:
    """Build recovery_events.json content for the active durability mode.

    For ``generation`` the durability point is the *global commit record*
    (``commits/commit_*.json``); ``source_of_truth`` is reported as
    ``global_commit_record`` and the recovery scanner's event log is
    serialized.  For ``wal`` / ``none`` a minimal record naming the WAL (or
    nothing) as the source of truth is written.
    """
    wal_present = _wal_present(work_dir)
    if recovery != "generation":
        return {
            "recovery_mode": recovery,
            "source_of_truth": "wal_json" if recovery == "wal" else "none",
            "wal_json_present": wal_present,
            "crashed": False,
            "events": [],
        }

    # generation mode — read the on-disk generation-recovery state.
    out: dict = {
        "recovery_mode": "generation",
        "source_of_truth": "global_commit_record",
        "wal_json_present": wal_present,
        "crashed": False,
        "events": [],
    }
    try:
        from wenbo_engine.recovery import (
            commits_dir, list_commit_files, read_run_metadata, RecoveryScanner,
        )
        wd = Path(work_dir)
        meta = read_run_metadata(wd)
        if meta is not None:
            out["recovery_mode"] = meta.recovery_mode
            out["n_ranks"] = getattr(meta, "n_ranks", None)
        commit_files = list_commit_files(commits_dir(wd))
        out["commits_dir"] = str(commits_dir(wd))
        out["n_commit_records"] = len(commit_files)
        out["commit_files"] = [p.name for p in commit_files]
        res = RecoveryScanner(wd).scan(quarantine=False)
        out["committed_generation"] = res.generation
        events = [e.to_dict() for e in res.events] if res.events else []
        # Merge any durably-recorded injected-fault events (proof #4): a hard
        # os_exit crash persists these to fault_events.jsonl before dying.
        faults = _read_fault_events(wd)
        if faults:
            out["crashed"] = True
            out["injected_faults"] = faults
            events = faults + events
        out["events"] = events
    except Exception as e:  # pragma: no cover - defensive
        out["scan_error"] = repr(e)
    return out


def _read_fault_events(work_dir: Path) -> list[dict]:
    """Read durably-persisted injected-fault events (one JSON object per line)."""
    sink = work_dir / "fault_events.jsonl"
    if not sink.exists():
        return []
    out: list[dict] = []
    for line in sink.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def write_artifacts(output_dir: str | Path, result: dict, circuit: dict,
                    work_dir: str | Path | None = None) -> None:
    """Write the standard experiment artifact bundle (rank-0 only).

    Emits config.json, circuit.json, plan.json, cost_model.json,
    stage_profile.csv, mpi_profile.csv, io_profile.csv,
    recovery_events.json, final_summary.json, final_norm.txt,
    git_commit.txt — schema-compatible with wenbo_engine.profiling /
    wenbo_engine.experiments.summary.  ``work_dir`` is the runner's scratch
    directory, needed to read the generation-recovery commit state.
    """
    d = Path(output_dir)
    d.mkdir(parents=True, exist_ok=True)
    agg = result["aggregate"]
    recovery = result.get("recovery_mode", "wal")

    # config.json
    (d / "config.json").write_text(json.dumps({
        "workload_kind": result["workload_kind"],
        "seed": result["seed"],
        "n": result["n"],
        "depth": result["depth"],
        "chunk_bits": result["chunk_bits"],
        "num_ranks": result["num_ranks"],
        "use_wal": result["use_wal"],
        "recovery_mode": recovery,
        "reorder_applied": result["reorder_applied"],
        "planner_mode": result.get("planner_mode", "current"),
        "mpi_exchange_mode": result.get("mpi_exchange_mode", "naive"),
        "storage_layout": result.get("storage_layout", "chunks"),
        "execution_mode": result.get("execution_mode", "step"),
        "extent_io_mode": result.get("extent_io_mode", "materialize"),
        "intended_locality": result["intended_locality"],
        "runner": "mpi",
    }, indent=2))

    # circuit.json
    (d / "circuit.json").write_text(json.dumps(circuit, indent=2, default=str))

    # plan.json
    (d / "plan.json").write_text(json.dumps({
        "n_qubits": result["n"],
        "chunk_bits": result["chunk_bits"],
        "chunk_size": 1 << result["chunk_bits"],
        "num_ranks": result["num_ranks"],
        "n_local_bits": result["n_local_bits"],
        "n_chunks_total": 1 << (result["n"] - result["chunk_bits"]),
        "n_chunks_per_rank": (1 << (result["n"] - result["chunk_bits"]))
                              // result["num_ranks"],
        "n_steps": result["n_steps"],
        "n_gates": len(circuit["gates"]),
        "intended_locality": result["intended_locality"],
        "op_counts": {
            "local": result["measured_local_ops"],
            "rank_nonlocal": result["measured_rank_nonlocal_ops"],
            "mpi_nonlocal": result["measured_mpi_nonlocal_ops"],
        },
    }, indent=2))

    # Recovery-aware planner v1 artifacts: a richer plan.json (overwriting the
    # simple one above), the candidate table, and the predicted-vs-actual report.
    ra = result.get("recovery_aware")
    if ra is not None:
        from wenbo_engine.planner import (
            serialize_recovery_aware_plan, serialize_candidate_strategies,
        )
        (d / "plan.json").write_text(json.dumps(
            serialize_recovery_aware_plan(ra["plan"]), indent=2, default=str))
        (d / "candidate_strategies.json").write_text(json.dumps(
            serialize_candidate_strategies(ra["plan"]), indent=2, default=str))
        (d / "cost_report.json").write_text(json.dumps(
            ra["cost_report"], indent=2, default=str))

    # cost_model.json — calibration is the observability harness's job; we
    # only guarantee the artifact exists (honest "not calibrated here").
    (d / "cost_model.json").write_text(json.dumps({"calibrated": False}, indent=2))

    # ablation_report.json — deterministic Optimizer-v2 plan metrics for all
    # modes on this circuit + hardware config (predicted, not measured).
    if result.get("ablation_report") is not None:
        (d / "ablation_report.json").write_text(
            json.dumps(result["ablation_report"], indent=2, default=str))

    # stage_profile.csv — one aggregate row of MEASURED values.  Per-step
    # granularity would need a runner hook (see module notes); op counts are
    # the runner's authoritative totals, timings/bytes are measured totals.
    with open(d / "stage_profile.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=STAGE_COLUMNS)
        w.writeheader()
        w.writerow({
            "step_or_stage_id": "aggregate",
            "local_ops": result["measured_local_ops"],
            "rank_nonlocal_ops": result["measured_rank_nonlocal_ops"],
            "mpi_nonlocal_ops": result["measured_mpi_nonlocal_ops"],
            "read_sec": round(agg["read_sec"], 6),
            "write_sec": round(agg["write_sec"], 6),
            "kernel_sec": round(agg["kernel_time"], 6),
            "mpi_sec": round(agg["mpi_sendrecv_time"], 6),
            "commit_sec": 0.0,
            "checksum_sec": 0.0,
            "bytes_read": agg["bytes_read"],
            "bytes_written": agg["bytes_written"],
            "mpi_bytes_sent": agg["mpi_bytes_sent"],
            "recovery_mode": "normal",
        })

    # mpi_profile.csv — one row per ACTUAL Sendrecv exchange (per rank),
    # plus a collective row per rank for the barriers.
    with open(d / "mpi_profile.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MPI_COLUMNS)
        w.writeheader()
        for r_idx, m in enumerate(result["per_rank"]):
            for peer, nbytes, secs in m["sendrecv_events"]:
                mbps = (nbytes / 1e6 / secs) if secs > 0 else 0.0
                w.writerow({
                    "stage_id": r_idx, "kind": "sendrecv", "op": "Sendrecv",
                    "peer": peer, "bytes_sent": nbytes,
                    "seconds": round(secs, 9), "mb_per_s": round(mbps, 3),
                })
            if m["barrier_count"]:
                w.writerow({
                    "stage_id": r_idx, "kind": "collective", "op": "Barrier",
                    "peer": -1, "bytes_sent": 0,
                    "seconds": round(m["mpi_wait_time"], 9), "mb_per_s": 0.0,
                })

    # io_profile.csv — header only; per-event I/O is not separately captured
    # (totals live in stage_profile.csv).  Allowed to be empty.
    with open(d / "io_profile.csv", "w", newline="") as f:
        csv.DictWriter(f, fieldnames=IO_COLUMNS).writeheader()

    # recovery_events.json — mode-aware (generation reads the commit state).
    (d / "recovery_events.json").write_text(
        json.dumps(_recovery_events(work_dir, recovery), indent=2))

    # final_norm.txt
    (d / "final_norm.txt").write_text(f"{result['final_norm']:.12f}\n")

    # git_commit.txt
    (d / "git_commit.txt").write_text(_git_commit() + "\n")

    # final_summary.json — required keys, all measured/authoritative.  When
    # the observability summary writer is importable, use it (reads our
    # stage_profile.csv) and merge our required fields; else write directly.
    required = {
        "workload_kind": result["workload_kind"],
        "seed": result["seed"],
        "n": result["n"],
        "depth": result["depth"],
        "chunk_bits": result["chunk_bits"],
        "num_ranks": result["num_ranks"],
        "recovery_mode": recovery,
        "intended_locality": result["intended_locality"],
        "measured_local_ops": result["measured_local_ops"],
        "measured_rank_nonlocal_ops": result["measured_rank_nonlocal_ops"],
        "measured_mpi_nonlocal_ops": result["measured_mpi_nonlocal_ops"],
        "mpi_bytes_sent": agg["mpi_bytes_sent"],
        "sendrecv_count": agg["sendrecv_count"],
        "partner_rank_pairs": result["partner_rank_pairs"],
        "estimated_partner_rank_pairs": result["estimated_partner_rank_pairs"],
        "final_norm": result["final_norm"],
        "reorder_applied": result["reorder_applied"],
        "mpi_exchange_mode": result.get("mpi_exchange_mode", "naive"),
        "storage_layout": result.get("storage_layout", "chunks"),
        "execution_mode": result.get("execution_mode", "step"),
        "extent_io_mode": result.get("extent_io_mode", "materialize"),
        # measured chunk-file I/O operation counts; write_ops == # temporary
        # chunk files materialized (0 for fully-direct extent runs).
        "read_ops": agg.get("read_ops", 0),
        "write_ops": agg.get("write_ops", 0),
        "temporary_chunk_files_created": agg.get("write_ops", 0),
        # Clifford / stabilizer analysis of the generated circuit
        "is_stabilizer": result.get("is_stabilizer"),
        "total_gate_count": result.get("total_gate_count"),
        "clifford_gate_count": result.get("clifford_gate_count"),
        "non_clifford_gate_count": result.get("non_clifford_gate_count"),
        "non_clifford_gate_types": result.get("non_clifford_gate_types"),
        "metrics_are_measured": True,
    }
    # Recovery-aware planner v1 decision fields in final_summary.
    _ra = result.get("recovery_aware")
    if _ra is not None:
        _dec = _ra["plan"]["decision"]
        _cr = _ra["cost_report"]
        required["planner"] = "recovery_aware_v1"
        required["planner_mode"] = "recovery_aware_v1"
        required["selected_strategy"] = _dec["selected_strategy"]["name"]
        required["rejected_candidates"] = [r["name"] for r in
                                           _dec["rejected_candidates"]]
        required["reason_for_selection"] = _dec["reason_for_selection"]
        required["predicted_cost_by_candidate"] = \
            _dec["predicted_cost_by_candidate"]
        required["actual_cost_after_run"] = _ra["actual_cost_after_run"]
        required["prediction_error_pct"] = {
            k: v["prediction_error_pct"] for k, v in _cr["metrics"].items()
        }

    _om = result.get("overlay_metrics") or {}
    for _kk in ("compute_unit_min_gates", "compute_units_executed",
                "compute_unit_fallbacks", "avg_gates_per_compute_unit",
                "gates_per_compute_unit", "overlay_load_count",
                "overlay_writeback_count", "overlay_bytes_read",
                "overlay_bytes_written"):
        if _kk in _om:
            required[_kk] = _om[_kk]

    # ── RAM-aware execution fields (measured by run() + planner estimate) ──
    _rm = result.get("ram_metrics") or {}
    required["auto_chunk_bits_enabled"] = bool(result.get("auto_chunk_bits_enabled"))
    required["chunk_bits"] = _rm.get("chunk_bits", result.get("chunk_bits"))
    required["chunk_bytes"] = _rm.get("chunk_bytes")
    required["ram_budget_gib"] = _rm.get("ram_budget_gib")
    required["max_overlay_chunks"] = _rm.get("max_overlay_chunks")
    required["max_remote_buffer_gib"] = _rm.get("max_remote_buffer_gib")
    required["overlay_peak_ram_gib"] = _rm.get("overlay_peak_ram_gib")
    required["remote_buffer_peak_gib"] = _rm.get("remote_buffer_peak_gib")
    required["measured_peak_working_set_gib"] = _rm.get("measured_peak_working_set_gib")
    required["ram_feasible"] = _rm.get("ram_feasible")
    _raram = (result.get("recovery_aware") or {}).get("plan", {}).get("ram", {}) \
        if result.get("recovery_aware") else {}
    required["estimated_peak_ram_gib"] = _raram.get("estimated_peak_ram_gib")
    required["recommended_chunk_bits"] = _raram.get("recommended_chunk_bits")

    # ── numerical backend fields ──
    _km = result.get("kernel_metrics") or {}
    required["kernel_backend_requested"] = _km.get(
        "kernel_backend_requested", result.get("kernel_backend_requested"))
    required["kernel_backend_used"] = _km.get("kernel_backend_used")
    required["kernel_backend_available"] = _km.get("kernel_backend_available")
    required["numba_compile_time"] = _km.get("numba_compile_time")
    required["backend_fallback_reason"] = _km.get("backend_fallback_reason")
    required["kernel_time"] = agg["kernel_time"]

    # ── MPI-nonlocal gate classification (analytic, per-circuit) ──
    _mgc = result.get("mpi_gate_classes") or {}
    for _kk in ("diagonal_mpi_nonlocal_gate_count",
                "true_mixing_mpi_nonlocal_gate_count",
                "permutation_mpi_nonlocal_gate_count",
                "requires_remote_amplitudes_gate_count",
                "skipped_mpi_exchange_gate_count"):
        required[_kk] = _mgc.get(_kk)

    # ── MPI remote-cache telemetry (measured + aggregated across ranks) ──
    _mt = result.get("mpi_telemetry") or {}
    for _kk in ("remote_buffer_cache_hits", "remote_buffer_cache_misses",
                "remote_buffer_cache_hit_rate", "distinct_remote_chunks_per_rank",
                "repeated_remote_chunk_fetches",
                "repeated_remote_chunk_fetches_adjacent_steps",
                "mpi_steps", "mpi_gates_per_step", "remote_cache_scope",
                "diagonal_mpi_applications", "exchange_mpi_applications",
                "requires_remote_amplitudes_applications"):
        required[_kk] = _mt.get(_kk)

    # ── MPI-window feasibility analysis (analysis-only; OFF by default) ──
    _wa = result.get("mpi_window_analysis") or {"analysis_mode": "off"}
    required["mpi_window_analysis"] = _wa.get("analysis_mode", "off")
    if _wa.get("analysis_mode") == "report":
        from wenbo_engine.planner.mpi_window_report import (
            report_to_candidates_json, report_to_summary_json,
        )
        (d / "mpi_window_candidates.json").write_text(
            json.dumps(report_to_candidates_json(_wa), indent=2))
        (d / "mpi_window_report.json").write_text(
            json.dumps(report_to_summary_json(_wa), indent=2))
        _to = _wa.get("tradeoffs", {})
        required["mpi_window_num_candidates"] = _wa.get("num_candidate_windows")
        required["mpi_window_num_feasible"] = _wa.get("num_feasible_windows")
        required["mpi_window_bytes_saved"] = _to.get("bytes_saved")
        required["mpi_window_sendrecv_saved"] = _to.get("sendrecv_calls_saved")
        required["mpi_window_commits_saved"] = _to.get("commits_saved")
        required["mpi_window_extra_ram_gib"] = _to.get("extra_ram_gib_required")
        required["mpi_window_extra_recomputation_cost"] = \
            _to.get("extra_recomputation_cost_after_crash")
        required["mpi_window_executor_worth_implementing"] = \
            _wa.get("executor_worth_implementing")
        required["mpi_window_recommendation"] = _wa.get("recommendation")

    # ── MPI true-mixing window executor metrics (off by default) ──
    _we = result.get("mpi_window_exec") or {}
    required["mpi_window_execution_mode"] = _we.get(
        "mpi_window_execution_mode",
        result.get("mpi_window_execution_mode", "off"))
    required["mpi_windows_executed"] = _we.get("mpi_windows_executed", 0)
    required["mpi_window_steps_executed"] = _we.get("mpi_window_steps_executed", 0)
    required["mpi_window_gates_executed"] = _we.get("mpi_window_gates_executed", 0)
    required["mpi_window_gather_bytes"] = _we.get("mpi_window_gather_bytes", 0)
    required["mpi_window_scatter_bytes"] = _we.get("mpi_window_scatter_bytes", 0)
    required["mpi_window_sendrecv_count"] = _we.get("mpi_window_sendrecv_count", 0)
    required["mpi_window_commits_saved"] = _we.get("mpi_window_commits_saved", 0)
    required["mpi_window_estimated_ram_gib"] = _we.get(
        "mpi_window_estimated_ram_gib", 0.0)
    required["mpi_window_fallbacks"] = _we.get("mpi_window_fallbacks", 0)
    required["mpi_window_fallback_reasons"] = _we.get(
        "mpi_window_fallback_reasons", [])
    required["expected_recomputation_cost_increase"] = _we.get(
        "expected_recomputation_cost_increase", 0.0)

    # ── calibrated-cost-model timing telemetry (measured; null+reason) ──
    _mt = result.get("measured_timing") or {}
    _timing_fields = (
        "local_kernel_time", "nonlocal_kernel_time",
        "mpi_pairwise_sendrecv_time", "mpi_collective_gather_time",
        "mpi_collective_scatter_time", "mpi_window_leader_compute_time",
        "mpi_window_segment_time", "direct_extent_read_time",
        "direct_extent_write_time", "extent_materialize_time",
        "extent_pack_time", "commit_time", "norm_time", "numba_compile_time",
        "rss_peak_gib", "overlay_peak_ram_gib", "remote_buffer_peak_gib",
    )
    _rm2 = result.get("ram_metrics") or {}
    for _f in _timing_fields:
        if _f == "overlay_peak_ram_gib":
            required[_f] = _rm2.get("overlay_peak_ram_gib")
        elif _f == "remote_buffer_peak_gib":
            required[_f] = _rm2.get("remote_buffer_peak_gib")
        else:
            required[_f] = _mt.get(_f)
        # explicit null-with-reason: never silently omit an unavailable metric
        if required.get(_f) is None:
            reason = _mt.get(_f + "_reason")
            required[_f + "_reason"] = reason or f"{_f} not measured this run"
    # calibrated constants (also written to cost_model.json below)
    required["calibration"] = result.get("calibration")

    # cost_model.json — the measured calibration constants for Planner v2.
    _cal = result.get("calibration")
    if _cal is not None:
        (d / "cost_model.json").write_text(json.dumps({
            "source": "measured_run",
            "workload_kind": result["workload_kind"],
            "n": result["n"], "num_ranks": result["num_ranks"],
            "chunk_bits": result["chunk_bits"],
            "kernel_backend_used": (result.get("kernel_metrics") or {}).get(
                "kernel_backend_used"),
            "constants": _cal,
        }, indent=2))

    if "correct" in result:
        required["correct"] = result["correct"]
    try:
        from wenbo_engine.experiments.summary import write_summary
        write_summary(d, extra=required)
    except Exception:
        (d / "final_summary.json").write_text(json.dumps(required, indent=2))


# ── CLI ────────────────────────────────────────────────────────────────

def _default_chunk_bits(n: int, num_ranks: int) -> int:
    """Pick chunk_bits leaving rank bits + 2 rank-nonlocal bits (clamped)."""
    p = _rank_bits(num_ranks) if num_ranks >= 1 else 0
    k = n - p - 2
    return max(1, min(k, n - p))


def main(argv: list[str] | None = None) -> None:
    from mpi4py import MPI

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_ranks = comm.Get_size()

    ap = argparse.ArgumentParser(description="MPI-nonlocal communication benchmark")
    ap.add_argument("--kind", required=True, choices=list(GENERATORS))
    ap.add_argument("--n", type=int, required=True, help="number of qubits")
    ap.add_argument("--depth", type=int, required=True, help="number of gates")
    ap.add_argument("--chunk-bits", type=int, default=None,
                    help="log2(chunk_size); default leaves 2 rank-nonlocal bits")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--work-dir", type=str, default="/tmp/wenbo_comm_bench",
                    help="node-local NVMe scratch for chunk storage")
    ap.add_argument("--output", type=str, default=None,
                    help="single JSON profile path (legacy)")
    ap.add_argument("--output-dir", type=str, default=None,
                    help="directory for the standard artifact bundle")
    ap.add_argument("--no-wal", action="store_true",
                    help="disable WAL (shorthand for --recovery none)")
    ap.add_argument("--recovery", choices=list(RECOVERY_MODES), default=None,
                    help="durability mode: none|wal|generation "
                         "(generation requires the generation-recovery merge)")
    ap.add_argument("--reorder", action="store_true",
                    help="apply production qubit reordering (DISABLED by "
                         "default so MPI-nonlocal stress is forced)")
    ap.add_argument("--fault-point", type=str, default=None,
                    help="inject a deterministic crash at this commit-protocol "
                         "fault point (e.g. AFTER_GLOBAL_COMMIT); requires "
                         "--recovery generation")
    ap.add_argument("--fault-rank", type=str, default=None,
                    help="rank to crash (default: any)")
    ap.add_argument("--fault-stage", type=str, default=None,
                    help="stage_id (circuit step) to crash at (default: any)")
    ap.add_argument("--fault-mode", type=str, default="os_exit",
                    help="crash mode: os_exit (default) | exception")
    ap.add_argument("--planner", default=None,
                    help="Planner mode. Optimizer-v2 ablation: current | "
                         "current_static_reorder | stage_v2 | "
                         "stage_v2_fusion | stage_v2_placement_fusion "
                         "(writes ablation_report.json; reorder/placement modes "
                         "apply a static transform).  Or recovery_aware_v1: "
                         "evaluate candidate strategies, select a safe one, and "
                         "write plan.json / candidate_strategies.json / "
                         "cost_report.json (predicted vs actual).")
    ap.add_argument("--mpi-exchange-mode", dest="mpi_exchange_mode",
                    choices=["naive", "gate_aware"], default="naive",
                    help="MPI-nonlocal exchange path: naive (one Sendrecv per "
                         "chunk per gate) or gate_aware (batch per-partner + "
                         "reuse received remote chunks).")
    ap.add_argument("--storage-layout", dest="storage_layout",
                    choices=["chunks", "extents"], default="chunks",
                    help="On-disk layout for committed generations: chunks (one "
                         "file per chunk) or extents (pack many chunks into few "
                         "extent files). Generation recovery only.")
    ap.add_argument("--execution-mode", dest="execution_mode",
                    choices=["step", "compute_unit"], default="step",
                    help="step (one read/write pass per circuit step, default) "
                         "or compute_unit (fuse consecutive local-only steps "
                         "into a RAM overlay: load once, apply many, write once).")
    ap.add_argument("--compute-unit-min-gates", dest="compute_unit_min_gates",
                    type=int, default=4,
                    help="min local gates to form a compute unit; shorter local "
                         "runs fall back to the step path (default 4).")
    ap.add_argument("--extent-io-mode", dest="extent_io_mode",
                    choices=["materialize", "direct"], default="materialize",
                    help="extents + compute_unit only: materialize (default; "
                         "round-trip through temp chunk files) or direct (read "
                         "logical chunks straight from source extent slices and "
                         "write dirty chunks straight into destination extents, "
                         "no temp chunk files).")
    # ── RAM-aware execution control ──
    ap.add_argument("--ram-budget-gib", dest="ram_budget_gib", type=float,
                    default=None,
                    help="per-rank RAM budget for the working set (overlay + "
                         "remote cache + temps). Default: 70%% of node RAM when "
                         "--auto-chunk-bits is set.")
    ap.add_argument("--auto-chunk-bits", dest="auto_chunk_bits",
                    action="store_true",
                    help="pick the largest chunk_bits whose predicted peak RAM "
                         "fits the budget (fails early if none fits).")
    ap.add_argument("--max-overlay-chunks", dest="max_overlay_chunks", type=int,
                    default=None,
                    help="cap resident compute-unit overlay chunks (default: "
                         "derived from the RAM budget).")
    ap.add_argument("--max-remote-buffer-gib", dest="max_remote_buffer_gib",
                    type=float, default=None,
                    help="cap the gate-aware remote-buffer cache (GiB/rank); "
                         "default: 25%% of the RAM budget.")
    ap.add_argument("--mpi-window-analysis", dest="mpi_window_analysis",
                    choices=["off", "report"], default="off",
                    help="analyze (do not execute) multi-step MPI exchange "
                         "windows: 'report' writes mpi_window_candidates.json + "
                         "mpi_window_report.json; 'off' (default) does nothing")
    ap.add_argument("--mpi-window-execution", dest="mpi_window_execution",
                    choices=["off", "safe"], default="off",
                    help="execute RAM-feasible consecutive true-mixing MPI "
                         "windows as one gather/apply/scatter + commit: 'safe' "
                         "enables it; 'off' (default) uses per-step execution")
    ap.add_argument("--kernel-backend", dest="kernel_backend",
                    choices=["numpy", "numba", "auto"], default="auto",
                    help="CPU kernel numerical backend: numpy (baseline) | numba "
                         "(JIT, if installed) | auto (numba when available, else "
                         "numpy). auto/numba fall back to numpy safely.")
    ap.add_argument("--verify", action="store_true",
                    help="collect full state and compare to ref_dense (small n)")
    ap.add_argument("--durable.enabled", dest="durable_enabled",
                    action="store_true",
                    help="promote the committed generation to durable storage "
                         "(requires --recovery generation)")
    ap.add_argument("--durable.backend", dest="durable_backend",
                    choices=["local_path", "s3"], default="local_path")
    ap.add_argument("--durable.root", dest="durable_root", default=None,
                    help="durable storage root (filesystem path / mount)")
    args = ap.parse_args(argv)

    durable_cfg = None
    if args.durable_enabled:
        durable_cfg = {"enabled": True, "backend": args.durable_backend,
                       "root": args.durable_root}

    # --recovery takes precedence; otherwise derive from --no-wal.
    recovery = args.recovery if args.recovery is not None else (
        "none" if args.no_wal else "wal")

    # Thread fault-injection config through the environment so every rank's
    # mpi_runner.run picks it up via FaultInjector.from_env().
    if args.fault_point:
        import os as _os
        _os.environ["WE_FAULT_POINT"] = args.fault_point
        _os.environ["WE_FAULT_MODE"] = args.fault_mode
        if args.fault_rank is not None:
            _os.environ["WE_FAULT_RANK"] = args.fault_rank
        if args.fault_stage is not None:
            _os.environ["WE_FAULT_STAGE"] = args.fault_stage
        if rank == 0:
            log.info("  fault injection: point=%s rank=%s stage=%s mode=%s",
                     args.fault_point, args.fault_rank, args.fault_stage,
                     args.fault_mode)

    chunk_bits = args.chunk_bits
    if chunk_bits is None:
        chunk_bits = _default_chunk_bits(args.n, num_ranks)

    if rank == 0:
        log.info("=" * 60)
        log.info("  communication workload: %s", args.kind)
        log.info("  n=%d depth=%d chunk_bits=%d ranks=%d seed=%d",
                 args.n, args.depth, chunk_bits, num_ranks, args.seed)
        log.info("  recovery: %s", recovery)
        log.info("  reorder: %s", "ON" if args.reorder
                 else "DISABLED (forcing MPI-nonlocal stress)")
        log.info("  planner: %s", args.planner or "current")
        log.info("=" * 60)

    result = run_workload(
        kind=args.kind, n=args.n, depth=args.depth, chunk_bits=chunk_bits,
        work_dir=args.work_dir, comm=comm, seed=args.seed,
        verify=args.verify, reorder=args.reorder, recovery=recovery,
        output_dir=args.output_dir, durable=durable_cfg, planner=args.planner,
        mpi_exchange_mode=args.mpi_exchange_mode,
        storage_layout=args.storage_layout,
        execution_mode=args.execution_mode,
        compute_unit_min_gates=args.compute_unit_min_gates,
        ram_budget_gib=args.ram_budget_gib,
        auto_chunk_bits=args.auto_chunk_bits,
        max_overlay_chunks=args.max_overlay_chunks,
        max_remote_buffer_gib=args.max_remote_buffer_gib,
        extent_io_mode=args.extent_io_mode,
        kernel_backend=args.kernel_backend,
        mpi_window_analysis=args.mpi_window_analysis,
        mpi_window_execution=args.mpi_window_execution,
    )

    if rank == 0:
        agg = result["aggregate"]
        log.info("  measured_mpi_nonlocal_ops : %d", result["measured_mpi_nonlocal_ops"])
        log.info("  measured_rank_nonlocal_ops: %d", result["measured_rank_nonlocal_ops"])
        log.info("  measured_local_ops        : %d", result["measured_local_ops"])
        log.info("  partner_rank_pairs (meas) : %d", result["partner_rank_pairs"])
        log.info("  mpi_bytes_sent            : %d", agg["mpi_bytes_sent"])
        log.info("  sendrecv_count            : %d", agg["sendrecv_count"])
        log.info("  mpi_sendrecv_time         : %.4fs", agg["mpi_sendrecv_time"])
        log.info("  mpi_wait_time             : %.4fs", agg["mpi_wait_time"])
        log.info("  bytes_read / written      : %d / %d",
                 agg["bytes_read"], agg["bytes_written"])
        log.info("  read_sec / write_sec      : %.4fs / %.4fs",
                 agg["read_sec"], agg["write_sec"])
        log.info("  kernel_time               : %.4fs", agg["kernel_time"])
        log.info("  stage_time                : %.4fs", agg["stage_time"])
        log.info("  final_norm                : %.10f", result["final_norm"])
        log.info("  recovery_mode             : %s", result["recovery_mode"])
        if "correct" in result:
            log.info("  correct (vs ref_dense)    : %s", result["correct"])

        if args.output:
            Path(args.output).parent.mkdir(parents=True, exist_ok=True)
            Path(args.output).write_text(json.dumps(result, indent=2))
            log.info("  JSON profile written to %s", args.output)
        if args.output_dir:
            log.info("  artifact bundle written to %s", args.output_dir)
        if not args.output and not args.output_dir:
            out = str(Path(args.work_dir) / f"{args.kind}_profile.json")
            Path(out).parent.mkdir(parents=True, exist_ok=True)
            Path(out).write_text(json.dumps(result, indent=2))
            log.info("  JSON profile written to %s", out)


if __name__ == "__main__":
    main()
