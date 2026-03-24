"""RDD-based distributed runner with data-local scheduling.

State vector lives as a Spark RDD persisted with MEMORY_AND_DISK (small
states) or DISK_ONLY (large states > 64 GB, e.g. 38q = 2 TB).
Spark tracks partition locations; subsequent ops get PROCESS_LOCAL scheduling.

  - Local gates:    mapPartitions(custom_kernel) — zero network I/O
                    Chunks in each partition are concatenated into one big
                    numpy array for vectorised gate application (single GEMM
                    call across all chunks), then split back.
  - Nonlocal gates: Applied ONE AT A TIME via groupByKey + custom kernel.
                    Each gate's group size ≤ 4 chunks (bounded, no OOM).
  - WAL/recovery:   checkpoint RDD every N nonlocal gates (sub-step)

Computation uses custom NumPy kernels (cpu_batched, cpu_nonlocal),
NOT Spark primitives. Spark handles scheduling + data transport only.

Partition coalescing: multiple chunks are packed per Spark partition to
reduce task scheduling overhead. At 38q with chunk_size=2^20, this gives
~1000 partitions instead of 262K.
"""
from __future__ import annotations

import logging
import math
import os
import re
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from wenbo_engine.circuit.io import validate_circuit_dict, levelize
from wenbo_engine.kernel import gates as gmod
from wenbo_engine.storage.block_store import DTYPE
from wenbo_engine.wal.wal import WAL

if TYPE_CHECKING:
    from pyspark import SparkContext, RDD

log = logging.getLogger(__name__)

DEFAULT_MAX_PARTITIONS = 1024
DEFAULT_GATE_CKPT_INTERVAL = 2  # checkpoint every N nonlocal gates


# ── helpers ──────────────────────────────────────────────────

def _crash_after_step() -> int | None:
    val = os.environ.get("WE_CRASH_AFTER_STEP")
    return int(val) if val is not None else None


def _serialise_ops(ops):
    return [(qs, U.tobytes(), U.shape) for qs, U in ops]


def _classify_ops(gates, k):
    local, nonlocal_ = [], []
    for g in gates:
        U = gmod.gate_matrix(g["gate"], g["params"])
        qs = g["qubits"]
        if all(q < k for q in qs):
            local.append((qs, U))
        else:
            nonlocal_.append((qs, U))
    return local, nonlocal_


def _storage_level(state_bytes: int = 0):
    """MEMORY_AND_DISK when state fits mostly in RAM, DISK_ONLY otherwise."""
    from pyspark import StorageLevel
    if state_bytes > 64 * (1024 ** 3):  # > 64 GB → disk only
        return StorageLevel.DISK_ONLY
    return StorageLevel.MEMORY_AND_DISK


def _find_latest_checkpoint(checkpoint_dir):
    """Scan for the latest sub-step checkpoint.

    Checkpoint naming: ckpt_s{step}_g{gate}
    where gate = number of nonlocal gates completed in that step (1-indexed).
    """
    ckpt_dir = Path(checkpoint_dir)
    if not ckpt_dir.exists():
        return None
    best = None
    for p in ckpt_dir.iterdir():
        m = re.match(r"ckpt_s(\d+)_g(\d+)", p.name)
        if m:
            step, gate = int(m.group(1)), int(m.group(2))
            if best is None or (step, gate) > (best[0], best[1]):
                best = (step, gate, str(p))
    return best


# ── RDD operations ───────────────────────────────────────────

def _init_state_rdd(sc: "SparkContext", n: int, chunk_size: int,
                    max_partitions: int, sl) -> "RDD":
    """Create |0...0> as an RDD[(int, bytes)].

    Packs multiple chunks per partition to limit Spark task count.
    """
    n_chunks = (1 << n) // chunk_size
    n_parts = min(n_chunks, max_partitions)

    def make_chunks(part_iter):
        for ci in part_iter:
            arr = np.zeros(chunk_size, dtype=DTYPE)
            if ci == 0:
                arr[0] = 1.0
            yield (ci, arr.tobytes())

    rdd = sc.parallelize(range(n_chunks), numSlices=n_parts) \
            .mapPartitions(make_chunks) \
            .persist(sl)
    rdd.count()  # force materialization
    return rdd


def _apply_local_step(state_rdd: "RDD", ops_ser: list,
                      chunk_size: int, sl) -> "RDD":
    """Apply local-only gates via mapPartitions — fully data-local.

    Concatenates all chunks in each partition into one big numpy array
    and applies gates once on the big array. This works because for
    local gates (q < k), pairs at stride 2^q never cross chunk
    boundaries — so one GEMM call processes all chunks at once.
    """

    def apply_gates_batch(part_iter):
        from wenbo_engine.kernel.cpu_batched import apply_1q, apply_2q

        chunks = list(part_iter)
        if not chunks:
            return

        chunks.sort(key=lambda x: x[0])
        chunk_ids = [ci for ci, _ in chunks]

        # Concatenate into one big array — single GEMM across all chunks
        big = np.concatenate(
            [np.frombuffer(raw, dtype=DTYPE) for _, raw in chunks]
        ).copy()

        for qubits, ubytes, ushape in ops_ser:
            U = np.frombuffer(ubytes, dtype=np.complex128).reshape(ushape)
            if len(qubits) == 1:
                apply_1q(big, qubits[0], U)
            else:
                apply_2q(big, qubits[0], qubits[1], U)

        # Split back into chunks
        for i, ci in enumerate(chunk_ids):
            start = i * chunk_size
            yield (ci, big[start:start + chunk_size].tobytes())

    new_rdd = state_rdd.mapPartitions(apply_gates_batch) \
                       .persist(sl)
    new_rdd.count()
    return new_rdd


def _apply_single_nonlocal_gate(state_rdd: "RDD", qs: tuple, U_bytes: bytes,
                                U_shape: tuple, k: int, sl) -> "RDD":
    """Apply ONE nonlocal gate via groupByKey.

    Group size is bounded: ≤ 2 chunks for 1q gate or 2q gate with 1
    nonlocal qubit, ≤ 4 chunks for 2q gate with both qubits nonlocal.
    At chunk_bits=24, that's max 512 MB per group — guaranteed to fit.
    """
    # Compute mask for this gate's nonlocal qubits (1-2 bits)
    nl_bits = set()
    for q in qs:
        if q >= k:
            nl_bits.add(q - k)
    mask = sum(1 << b for b in nl_bits)

    def process_group(item):
        group_key, chunks_iter = item
        from wenbo_engine.kernel.cpu_nonlocal import (
            apply_1q_pair, apply_2q_pair_qa_local,
            apply_2q_pair_qb_local, apply_2q_quad,
        )

        U = np.frombuffer(U_bytes, dtype=np.complex128).reshape(U_shape)

        data = {}
        for ci, raw in chunks_iter:
            data[ci] = np.frombuffer(raw, dtype=DTYPE).copy()

        if len(qs) == 1:
            q = qs[0]
            pbit = q - k
            done = set()
            for ci in data:
                c0 = ci & ~(1 << pbit)
                if c0 in done:
                    continue
                done.add(c0)
                c1 = c0 | (1 << pbit)
                apply_1q_pair(data[c0], data[c1], U)
        else:
            qa, qb = qs
            if qa < k:
                pbit = qb - k
                done = set()
                for ci in data:
                    c0 = ci & ~(1 << pbit)
                    if c0 in done:
                        continue
                    done.add(c0)
                    c1 = c0 | (1 << pbit)
                    apply_2q_pair_qa_local(data[c0], data[c1], qa, U)
            elif qb < k:
                pbit = qa - k
                done = set()
                for ci in data:
                    c0 = ci & ~(1 << pbit)
                    if c0 in done:
                        continue
                    done.add(c0)
                    c1 = c0 | (1 << pbit)
                    apply_2q_pair_qb_local(data[c0], data[c1], qb, U)
            else:
                pa, pb = qa - k, qb - k
                done = set()
                for ci in data:
                    cb = ci & ~(1 << pa) & ~(1 << pb)
                    if cb in done:
                        continue
                    done.add(cb)
                    apply_2q_quad(
                        data[cb],
                        data[cb | (1 << pb)],
                        data[cb | (1 << pa)],
                        data[cb | (1 << pa) | (1 << pb)],
                        U,
                    )

        return [(ci, data[ci].tobytes()) for ci in data]

    new_rdd = state_rdd \
        .map(lambda item: (item[0] & ~mask, (item[0], item[1]))) \
        .groupByKey() \
        .flatMap(process_group) \
        .persist(sl)
    new_rdd.count()
    return new_rdd


# ── main entry point ─────────────────────────────────────────

def run(
    circuit_dict: dict,
    sc: "SparkContext",
    chunk_size: int = 1 << 20,
    max_partitions: int = DEFAULT_MAX_PARTITIONS,
    checkpoint_dir: str | Path | None = None,
    gate_ckpt_interval: int = DEFAULT_GATE_CKPT_INTERVAL,
    use_wal: bool = True,
) -> dict:
    """Run the circuit simulation.

    Args:
        gate_ckpt_interval: Checkpoint every N nonlocal gates (default: 2).
            Lower = less lost work on crash, more I/O.
    """
    cd = validate_circuit_dict(circuit_dict)
    n = cd["number_of_qubits"]
    N = 1 << n
    if chunk_size > N:
        chunk_size = N
    if N % chunk_size != 0:
        raise ValueError("2^n must be divisible by chunk_size")

    k = int(math.log2(chunk_size))
    n_chunks = N // chunk_size
    n_parts = min(n_chunks, max_partitions)

    levels = levelize(cd)
    steps = []
    for lv in levels:
        if not lv:
            continue
        lo, nlo = _classify_ops(lv, k)
        steps.append({"local_ops": lo, "nonlocal_ops": nlo})

    # WAL setup
    wal_dir = Path(checkpoint_dir) if checkpoint_dir else None
    if wal_dir:
        wal_dir.mkdir(parents=True, exist_ok=True)
    wal = None
    if use_wal and wal_dir:
        wal = WAL(wal_dir / "wal.json", circuit_dict=cd)

    # Find recovery point from checkpoints
    start_step = 0
    start_gate = 0  # nonlocal gate index to resume from within start_step

    if checkpoint_dir:
        latest = _find_latest_checkpoint(checkpoint_dir)
        if latest:
            ckpt_step, ckpt_gate, _ckpt_path = latest
            n_nonlocal = len(steps[ckpt_step]["nonlocal_ops"]) if ckpt_step < len(steps) else 0
            if ckpt_gate >= n_nonlocal:
                # Full step completed, start from next step
                start_step = ckpt_step + 1
                start_gate = 0
            else:
                start_step = ckpt_step
                start_gate = ckpt_gate
            log.info(f"found checkpoint: step {ckpt_step}, gate {ckpt_gate}")

    state_bytes = N * 8
    state_gb = state_bytes / (1024**3)
    sl = _storage_level(state_bytes)
    log.info(f"{n}q ({state_gb:.1f} GB), {n_chunks} chunks in {n_parts} partitions, "
             f"{len(steps)} steps (resume from step {start_step} gate {start_gate}), "
             f"gate checkpoint every {gate_ckpt_interval}, "
             f"storage={'DISK_ONLY' if state_gb > 64 else 'MEMORY_AND_DISK'}")

    # Set Spark checkpoint dir for RDD recovery
    if checkpoint_dir:
        sc.setCheckpointDir(str(Path(checkpoint_dir) / "spark_checkpoints"))

    # Initialize or recover state
    if start_step == 0 and start_gate == 0:
        state_rdd = _init_state_rdd(sc, n, chunk_size, max_partitions, sl)
        log.info("initialized |0...0> RDD")
    else:
        latest = _find_latest_checkpoint(checkpoint_dir)
        state_rdd = sc.pickleFile(latest[2]).persist(sl)
        state_rdd.count()
        log.info(f"recovered RDD from checkpoint at step {start_step} gate {start_gate}")

    crash_after_step = _crash_after_step()

    for step_idx in range(start_step, len(steps)):
        step = steps[step_idx]
        local_ops = step["local_ops"]
        nonlocal_ops = step["nonlocal_ops"]

        # Determine starting gate for this step
        resume_gate = start_gate if step_idx == start_step else 0
        n_local = len(local_ops)
        n_nonlocal = len(nonlocal_ops)

        log.info(f"step {step_idx+1}/{len(steps)}: "
                 f"{n_local} local, {n_nonlocal} non-local"
                 + (f" (resume from gate {resume_gate})" if resume_gate > 0 else ""))

        if not nonlocal_ops:
            # Pure local step
            new_rdd = _apply_local_step(
                state_rdd, _serialise_ops(local_ops), chunk_size, sl)
            state_rdd.unpersist()
            state_rdd = new_rdd
        else:
            # Apply local gates first (skip if recovering past this point)
            if resume_gate == 0 and local_ops:
                log.info(f"  applying {n_local} local gates")
                local_rdd = _apply_local_step(
                    state_rdd, _serialise_ops(local_ops), chunk_size, sl)
                state_rdd.unpersist()
                state_rdd = local_rdd

            # Apply nonlocal gates one at a time
            for gi in range(resume_gate, n_nonlocal):
                qs, U = nonlocal_ops[gi]
                log.info(f"  nonlocal gate {gi+1}/{n_nonlocal}: qubits {qs}")

                new_rdd = _apply_single_nonlocal_gate(
                    state_rdd, qs, U.tobytes(), U.shape, k, sl)
                state_rdd.unpersist()
                state_rdd = new_rdd

                # Checkpoint every N nonlocal gates or on last gate
                is_last = (gi == n_nonlocal - 1)
                should_ckpt = (
                    checkpoint_dir
                    and ((gi + 1) % gate_ckpt_interval == 0 or is_last)
                )
                if should_ckpt:
                    ckpt_path = str(
                        Path(checkpoint_dir) / f"ckpt_s{step_idx}_g{gi+1}")
                    state_rdd.saveAsPickleFile(ckpt_path)
                    log.info(f"  checkpointed step {step_idx} gate {gi+1}")

        # WAL commit full step
        if wal:
            wal.commit_step(step_idx, "rdd")

        if crash_after_step is not None and step_idx + 1 >= crash_after_step:
            log.warning(f"crash injection: dying after step {step_idx + 1}")
            raise SystemExit(1)

    if wal:
        wal.close()

    return {
        "state_rdd": state_rdd,
        "n_chunks": n_chunks,
        "n_partitions": n_parts,
        "n_qubits": n,
        "chunk_size": chunk_size,
    }


def collect_state_rdd(result: dict) -> np.ndarray:
    """Collect full state to driver. Only for testing — don't use at 38q."""
    state_rdd = result["state_rdd"]
    chunks = state_rdd.collect()
    chunks.sort(key=lambda x: x[0])
    arrays = [np.frombuffer(raw, dtype=DTYPE) for _, raw in chunks]
    return np.concatenate(arrays).astype(np.complex128)
