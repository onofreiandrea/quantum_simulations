"""RDD-based distributed runner with data-local scheduling.

State vector lives as a Spark RDD persisted with MEMORY_AND_DISK.
Spark tracks partition locations; subsequent ops get PROCESS_LOCAL scheduling.

  - Local gates:    mapPartitions(custom_kernel) — zero network I/O
                    Chunks in each partition are concatenated into one big
                    numpy array for vectorised gate application (single GEMM
                    call across all chunks), then split back.
  - Nonlocal gates: groupByKey (Spark shuffles partner chunks) + custom kernel
  - WAL/recovery:   checkpoint RDD every N steps (configurable)

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
DEFAULT_CHECKPOINT_INTERVAL = 1  # checkpoint every step by default


# ── helpers ──────────────────────────────────────────────────────

def _crash_after_step() -> int | None:
    val = os.environ.get("WE_CRASH_AFTER_STEP")
    return int(val) if val is not None else None


def _serialise_ops(ops):
    return [(qs, U.tobytes(), U.shape) for qs, U in ops]


def _deserialise_ops(ops_ser):
    return [(qs, np.frombuffer(ub, dtype=np.complex128).reshape(us))
            for qs, ub, us in ops_ser]


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


def _nonlocal_mask(nonlocal_ops, k):
    """Compute the bitmask of nonlocal qubit positions."""
    nl_bits = set()
    for qs, _U in nonlocal_ops:
        for q in qs:
            if q >= k:
                nl_bits.add(q - k)
    return sum(1 << b for b in nl_bits)


def _storage_level():
    from pyspark import StorageLevel
    return StorageLevel.MEMORY_AND_DISK


# ── RDD operations ───────────────────────────────────────────────

def _init_state_rdd(sc: "SparkContext", n: int, chunk_size: int,
                    max_partitions: int) -> "RDD":
    """Create |0...0> as an RDD[(int, bytes)].

    Packs multiple chunks per partition to limit Spark task count.
    """
    n_chunks = (1 << n) // chunk_size
    n_parts = min(n_chunks, max_partitions)
    sl = _storage_level()

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
                      chunk_size: int) -> "RDD":
    """Apply local-only gates via mapPartitions — fully data-local.

    Concatenates all chunks in each partition into one big numpy array
    and applies gates once on the big array. This works because for
    local gates (q < k), pairs at stride 2^q never cross chunk
    boundaries — so one GEMM call processes all chunks at once.
    """
    sl = _storage_level()

    def apply_gates_batch(part_iter):
        from wenbo_engine.kernel.cpu_batched import apply_1q, apply_2q

        chunks = list(part_iter)
        if not chunks:
            return

        chunks.sort(key=lambda x: x[0])
        chunk_ids = [ci for ci, _ in chunks]
        n_ch = len(chunk_ids)

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


def _apply_nonlocal_step(state_rdd: "RDD", local_ops_ser: list,
                          nonlocal_ops_ser: list, k: int,
                          mask: int) -> "RDD":
    """Apply a step with nonlocal gates via groupByKey + custom kernel."""
    sl = _storage_level()

    def process_group(item):
        group_key, chunks_iter = item
        from wenbo_engine.kernel.cpu_batched import apply_1q, apply_2q
        from wenbo_engine.kernel.cpu_nonlocal import (
            apply_1q_pair, apply_2q_pair_qa_local,
            apply_2q_pair_qb_local, apply_2q_quad,
        )

        local_ops = [(qs, np.frombuffer(ub, dtype=np.complex128).reshape(us))
                     for qs, ub, us in local_ops_ser]
        nonlocal_ops = [(qs, np.frombuffer(ub, dtype=np.complex128).reshape(us))
                        for qs, ub, us in nonlocal_ops_ser]

        data = {}
        for ci, raw in chunks_iter:
            data[ci] = np.frombuffer(raw, dtype=DTYPE).copy()

        group = sorted(data.keys())

        # Apply local gates to each chunk
        for ci in group:
            for qubits, U in local_ops:
                if len(qubits) == 1:
                    apply_1q(data[ci], qubits[0], U)
                else:
                    apply_2q(data[ci], qubits[0], qubits[1], U)

        # Apply nonlocal gates across chunk pairs
        for qs, U in nonlocal_ops:
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

        return [(ci, data[ci].tobytes()) for ci in group]

    # Re-key chunks by group, shuffle, process, persist
    new_rdd = state_rdd \
        .map(lambda item: (item[0] & ~mask, (item[0], item[1]))) \
        .groupByKey() \
        .flatMap(process_group) \
        .persist(sl)
    new_rdd.count()
    return new_rdd


# ── main entry point ─────────────────────────────────────────────

def run(
    circuit_dict: dict,
    sc: "SparkContext",
    chunk_size: int = 1 << 20,
    max_partitions: int = DEFAULT_MAX_PARTITIONS,
    checkpoint_dir: str | Path | None = None,
    checkpoint_interval: int = DEFAULT_CHECKPOINT_INTERVAL,
    use_wal: bool = True,
) -> dict:
    """Run the circuit simulation.

    Args:
        checkpoint_interval: Save RDD checkpoint every N steps.
            1 = every step (safest, most I/O).
            Higher = less I/O, but lose more work on crash.
            At 38q, each checkpoint = 2 TB write.
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

    wal_done = wal.done_steps if wal else 0

    # Snap to last available checkpoint (may be < wal_done if interval > 1)
    start_step = 0
    if wal_done > 0 and checkpoint_dir:
        for s in range(wal_done, 0, -1):
            if Path(checkpoint_dir, f"ckpt_step_{s}").exists():
                start_step = s
                break

    state_gb = N * 8 / (1024**3)
    log.info(f"{n}q ({state_gb:.1f} GB), {n_chunks} chunks in {n_parts} partitions, "
             f"{len(steps)} steps (resume from step {start_step}), "
             f"checkpoint every {checkpoint_interval} steps")

    # Set Spark checkpoint dir for RDD recovery
    if checkpoint_dir:
        sc.setCheckpointDir(str(Path(checkpoint_dir) / "spark_checkpoints"))

    # Initialize or recover state
    if start_step == 0:
        state_rdd = _init_state_rdd(sc, n, chunk_size, max_partitions)
        log.info("initialized |0...0> RDD")
    else:
        # Recover from checkpoint
        sl = _storage_level()
        state_rdd = sc.pickleFile(
            str(Path(checkpoint_dir) / f"ckpt_step_{start_step}")
        ).persist(sl)
        state_rdd.count()
        log.info(f"recovered RDD from checkpoint at step {start_step}")

    crash_after_step = _crash_after_step()

    for step_idx in range(start_step, len(steps)):
        step = steps[step_idx]
        local_ops = step["local_ops"]
        nonlocal_ops = step["nonlocal_ops"]

        log.info(f"step {step_idx+1}/{len(steps)}: "
                 f"{len(local_ops)} local, {len(nonlocal_ops)} non-local")

        local_ops_ser = _serialise_ops(local_ops)
        nonlocal_ops_ser = _serialise_ops(nonlocal_ops)

        if not nonlocal_ops:
            new_rdd = _apply_local_step(state_rdd, local_ops_ser, chunk_size)
        else:
            mask = _nonlocal_mask(nonlocal_ops, k)
            new_rdd = _apply_nonlocal_step(
                state_rdd, local_ops_ser, nonlocal_ops_ser, k, mask)

        # Checkpoint for crash recovery (only every N steps, or on last step)
        is_last_step = (step_idx == len(steps) - 1)
        should_checkpoint = (
            checkpoint_dir
            and ((step_idx + 1) % checkpoint_interval == 0 or is_last_step)
        )
        if should_checkpoint:
            ckpt_path = str(Path(checkpoint_dir) / f"ckpt_step_{step_idx + 1}")
            new_rdd.saveAsPickleFile(ckpt_path)
            log.info(f"  checkpointed step {step_idx + 1}")

        # Release old state
        state_rdd.unpersist()
        state_rdd = new_rdd

        # WAL commit
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
