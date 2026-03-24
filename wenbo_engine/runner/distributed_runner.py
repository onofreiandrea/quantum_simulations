"""Distributed runner with shared-filesystem chunk storage (double-buffer).

All nodes access chunks via a shared filesystem (NFS mount from master's
NVMe). Spark distributes compute tasks across executors; any executor can
read/write any chunk. Same double-buffer + WAL recovery as single_node.
"""
from __future__ import annotations

import logging
import math
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from wenbo_engine.circuit.io import validate_circuit_dict, levelize
from wenbo_engine.circuit.fusion import batch_levels
from wenbo_engine.kernel import gates as gmod
from wenbo_engine.storage.block_store import DTYPE, chunk_filename
from wenbo_engine.wal.wal import WAL

if TYPE_CHECKING:
    from pyspark import SparkContext

log = logging.getLogger(__name__)


def _crash_after_step() -> int | None:
    val = os.environ.get("WE_CRASH_AFTER_STEP")
    return int(val) if val is not None else None


def _chunk_owner(ci, n_chunks, n_partitions):
    return ci * n_partitions // n_chunks


def _partition_range(part_id, n_chunks, n_partitions):
    start = part_id * n_chunks // n_partitions
    end = (part_id + 1) * n_chunks // n_partitions
    return start, end


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


def _init_partition(args):
    part_id, start, end, chunk_size, local_dir, buf = args
    import numpy as np
    from wenbo_engine.storage.block_store import DTYPE, chunk_filename, write_chunk_atomic

    buf_dir = Path(local_dir) / f"state_{buf}" / "chunks"
    buf_dir.mkdir(parents=True, exist_ok=True)

    for ci in range(start, end):
        arr = np.zeros(chunk_size, dtype=DTYPE)
        if ci == 0:
            arr[0] = 1.0
        write_chunk_atomic(buf_dir / chunk_filename(ci), arr)
    return part_id, end - start


def _process_local_partition(args):
    part_id, start, end, ops_ser, local_dir, src_buf, dst_buf = args
    from wenbo_engine.kernel.cpu_batched import apply_1q, apply_2q
    from wenbo_engine.storage.block_store import read_chunk, write_chunk_atomic, chunk_filename
    import numpy as np

    src_dir = Path(local_dir) / f"state_{src_buf}" / "chunks"
    dst_dir = Path(local_dir) / f"state_{dst_buf}" / "chunks"
    dst_dir.mkdir(parents=True, exist_ok=True)

    ops = [(qs, np.frombuffer(ub, dtype=np.complex128).reshape(us))
           for qs, ub, us in ops_ser]

    for ci in range(start, end):
        cname = chunk_filename(ci)
        data = read_chunk(src_dir / cname)
        for qubits, U in ops:
            if len(qubits) == 1:
                apply_1q(data, qubits[0], U)
            else:
                apply_2q(data, qubits[0], qubits[1], U)
        write_chunk_atomic(dst_dir / cname, data)

    return part_id, end - start


def _process_local_chunks(args):
    chunk_indices, ops_ser, local_dir, src_buf, dst_buf = args
    from wenbo_engine.kernel.cpu_batched import apply_1q, apply_2q
    from wenbo_engine.storage.block_store import read_chunk, write_chunk_atomic, chunk_filename
    import numpy as np

    src_dir = Path(local_dir) / f"state_{src_buf}" / "chunks"
    dst_dir = Path(local_dir) / f"state_{dst_buf}" / "chunks"
    dst_dir.mkdir(parents=True, exist_ok=True)

    ops = [(qs, np.frombuffer(ub, dtype=np.complex128).reshape(us))
           for qs, ub, us in ops_ser]

    for ci in chunk_indices:
        cname = chunk_filename(ci)
        data = read_chunk(src_dir / cname)
        for qubits, U in ops:
            if len(qubits) == 1:
                apply_1q(data, qubits[0], U)
            else:
                apply_2q(data, qubits[0], qubits[1], U)
        write_chunk_atomic(dst_dir / cname, data)

    return len(chunk_indices)


def _read_chunks_for_exchange(args):
    chunk_indices, local_dir, src_buf = args
    from wenbo_engine.storage.block_store import read_chunk, chunk_filename

    src_dir = Path(local_dir) / f"state_{src_buf}" / "chunks"
    return [(ci, read_chunk(src_dir / chunk_filename(ci)).tobytes())
            for ci in chunk_indices]


def _write_chunks_after_exchange(args):
    chunks_data, local_dir, dst_buf = args
    from wenbo_engine.storage.block_store import write_chunk_atomic, chunk_filename, DTYPE
    import numpy as np

    dst_dir = Path(local_dir) / f"state_{dst_buf}" / "chunks"
    dst_dir.mkdir(parents=True, exist_ok=True)

    for ci, raw in chunks_data:
        data = np.frombuffer(raw, dtype=DTYPE).copy()
        write_chunk_atomic(dst_dir / chunk_filename(ci), data)
    return len(chunks_data)


def _wipe_local_buf(args):
    part_id, local_dir, buf, n_chunks, n_partitions = args
    from wenbo_engine.storage.block_store import chunk_filename

    buf_dir = Path(local_dir) / f"state_{buf}" / "chunks"
    if not buf_dir.exists():
        return part_id
    start = part_id * n_chunks // n_partitions
    end = (part_id + 1) * n_chunks // n_partitions
    for ci in range(start, end):
        p = buf_dir / chunk_filename(ci)
        if p.exists():
            p.unlink()
    return part_id


def _copy_src_to_dst(args):
    chunk_indices, local_dir, src_buf, dst_buf = args
    from wenbo_engine.storage.block_store import read_chunk, write_chunk_atomic, chunk_filename

    src_dir = Path(local_dir) / f"state_{src_buf}" / "chunks"
    dst_dir = Path(local_dir) / f"state_{dst_buf}" / "chunks"
    dst_dir.mkdir(parents=True, exist_ok=True)

    for ci in chunk_indices:
        cname = chunk_filename(ci)
        data = read_chunk(src_dir / cname)
        write_chunk_atomic(dst_dir / cname, data)
    return len(chunk_indices)


def _compute_nonlocal_groups(n_chunks, k, nonlocal_ops):
    nl_bits = set()
    for qs, U in nonlocal_ops:
        for q in qs:
            if q >= k:
                nl_bits.add(q - k)
    nl_bits_sorted = sorted(nl_bits)
    mask = sum(1 << b for b in nl_bits_sorted)

    processed_bases = set()
    groups = []
    for c in range(n_chunks):
        base = c & ~mask
        if base in processed_bases:
            continue
        processed_bases.add(base)

        group = []
        for combo in range(1 << len(nl_bits_sorted)):
            idx = base
            for i, b in enumerate(nl_bits_sorted):
                if combo & (1 << i):
                    idx |= (1 << b)
            group.append(idx)
        groups.append(group)

    return groups


def _process_nonlocal_group(args):
    group, chunks_data_raw, local_ops_ser, nonlocal_ops_ser, k, chunk_size = args
    from wenbo_engine.kernel.cpu_batched import apply_1q, apply_2q
    from wenbo_engine.kernel.cpu_nonlocal import (
        apply_1q_pair, apply_2q_pair_qa_local,
        apply_2q_pair_qb_local, apply_2q_quad,
    )
    from wenbo_engine.storage.block_store import DTYPE
    import numpy as np

    local_ops = [(qs, np.frombuffer(ub, dtype=np.complex128).reshape(us))
                 for qs, ub, us in local_ops_ser]
    nonlocal_ops = [(qs, np.frombuffer(ub, dtype=np.complex128).reshape(us))
                    for qs, ub, us in nonlocal_ops_ser]

    data = {}
    for ci, raw in chunks_data_raw:
        data[ci] = np.frombuffer(raw, dtype=DTYPE).copy()

    for ci in group:
        for qubits, U in local_ops:
            if len(qubits) == 1:
                apply_1q(data[ci], qubits[0], U)
            else:
                apply_2q(data[ci], qubits[0], qubits[1], U)

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


def _run_local_step(sc, n_chunks, n_partitions, local_ops,
                    local_dir, src_buf, dst_buf):
    ops_ser = _serialise_ops(local_ops)
    tasks = []
    for p in range(n_partitions):
        s, e = _partition_range(p, n_chunks, n_partitions)
        tasks.append((p, s, e, ops_ser, local_dir, src_buf, dst_buf))
    sc.parallelize(tasks, numSlices=n_partitions) \
      .map(_process_local_partition).collect()


def _run_nonlocal_step(sc, n_chunks, n_partitions, local_ops,
                       nonlocal_ops, k, chunk_size, local_dir,
                       src_buf, dst_buf):
    groups = _compute_nonlocal_groups(n_chunks, k, nonlocal_ops)
    local_ops_ser = _serialise_ops(local_ops)
    nonlocal_ops_ser = _serialise_ops(nonlocal_ops)

    affected_chunks = set()
    for group in groups:
        affected_chunks.update(group)

    def process_group_task(task):
        group, chunks_by_owner = task
        all_chunks = []
        for owner, indices in chunks_by_owner.items():
            result = _read_chunks_for_exchange((indices, local_dir, src_buf))
            all_chunks.extend(result)

        processed = _process_nonlocal_group((
            group, all_chunks, local_ops_ser, nonlocal_ops_ser, k, chunk_size,
        ))

        by_owner = {}
        for ci, raw in processed:
            owner = _chunk_owner(ci, n_chunks, n_partitions)
            by_owner.setdefault(owner, []).append((ci, raw))
        for owner, chunks_data in by_owner.items():
            _write_chunks_after_exchange((chunks_data, local_dir, dst_buf))

        return len(group)

    group_tasks = []
    for group in groups:
        chunks_by_owner = {}
        for ci in group:
            owner = _chunk_owner(ci, n_chunks, n_partitions)
            chunks_by_owner.setdefault(owner, []).append(ci)
        group_tasks.append((group, chunks_by_owner))

    if group_tasks:
        sc.parallelize(group_tasks,
                       numSlices=max(1, min(len(group_tasks), n_partitions * 4))) \
          .map(process_group_task).collect()

    unaffected_by_owner = {}
    for ci in range(n_chunks):
        if ci not in affected_chunks:
            owner = _chunk_owner(ci, n_chunks, n_partitions)
            unaffected_by_owner.setdefault(owner, []).append(ci)

    if unaffected_by_owner:
        if local_ops:
            ops_ser = _serialise_ops(local_ops)
            tasks = [(indices, ops_ser, local_dir, src_buf, dst_buf)
                     for indices in unaffected_by_owner.values()]
            sc.parallelize(tasks, numSlices=len(tasks)) \
              .map(_process_local_chunks).collect()
        else:
            tasks = [(indices, local_dir, src_buf, dst_buf)
                     for indices in unaffected_by_owner.values()]
            sc.parallelize(tasks, numSlices=len(tasks)) \
              .map(_copy_src_to_dst).collect()


def run(
    circuit_dict: dict,
    sc: "SparkContext",
    local_dir: str = "/mnt/nvme/wenbo_data",
    wal_dir: str | Path | None = None,
    chunk_size: int = 1 << 20,
    n_partitions: int | None = None,
    use_wal: bool = True,
    use_fusion: bool = False,
) -> dict:
    cd = validate_circuit_dict(circuit_dict)
    n = cd["number_of_qubits"]
    N = 1 << n
    if chunk_size > N:
        chunk_size = N
    if N % chunk_size != 0:
        raise ValueError("2^n must be divisible by chunk_size")

    k = int(math.log2(chunk_size))
    n_chunks = N // chunk_size
    if n_partitions is None:
        n_partitions = sc.defaultParallelism

    levels = levelize(cd)
    if use_fusion:
        steps = batch_levels(levels, k)
    else:
        steps = []
        for lv in levels:
            if not lv:
                continue
            lo, nlo = _classify_ops(lv, k)
            steps.append({"local_ops": lo, "nonlocal_ops": nlo})

    wal_path = Path(wal_dir or local_dir)
    wal_path.mkdir(parents=True, exist_ok=True)
    wal = WAL(wal_path / "wal.json", circuit_dict=cd) if use_wal else None

    start_step = wal.done_steps if wal else 0
    current_buf = wal.committed_buf if wal else "a"

    state_gb = N * 8 / (1024**3)
    log.info(f"{n}q ({state_gb:.1f} GB), {n_chunks} chunks, "
             f"{n_partitions} executors, {len(steps)} steps "
             f"(resume from step {start_step})")

    if start_step == 0:
        init_tasks = []
        for p in range(n_partitions):
            s, e = _partition_range(p, n_chunks, n_partitions)
            init_tasks.append((p, s, e, chunk_size, local_dir, "a"))
        sc.parallelize(init_tasks, numSlices=n_partitions) \
          .map(_init_partition).collect()
        log.info("wrote |0...0> across all executors")

    crash_after_step = _crash_after_step()

    for step_idx in range(start_step, len(steps)):
        src_buf = current_buf
        dst_buf = "b" if src_buf == "a" else "a"
        step = steps[step_idx]
        local_ops = step["local_ops"]
        nonlocal_ops = step["nonlocal_ops"]

        log.info(f"step {step_idx+1}/{len(steps)}: "
                 f"{len(local_ops)} local, {len(nonlocal_ops)} non-local")

        dst_dir = Path(local_dir) / f"state_{dst_buf}" / "chunks"
        if dst_dir.exists():
            shutil.rmtree(dst_dir)
        dst_dir.mkdir(parents=True, exist_ok=True)

        if not nonlocal_ops:
            _run_local_step(sc, n_chunks, n_partitions, local_ops,
                           local_dir, src_buf, dst_buf)
        else:
            _run_nonlocal_step(sc, n_chunks, n_partitions, local_ops,
                              nonlocal_ops, k, chunk_size, local_dir,
                              src_buf, dst_buf)

        current_buf = dst_buf
        if wal:
            wal.commit_step(step_idx, current_buf)

        if crash_after_step is not None and step_idx + 1 >= crash_after_step:
            log.warning(f"crash injection: dying after step {step_idx + 1}")
            raise SystemExit(1)

    if wal:
        wal.close()

    return {
        "committed_buf": current_buf,
        "n_chunks": n_chunks,
        "local_dir": local_dir,
        "n_partitions": n_partitions,
    }


def collect_state_distributed(sc, result):
    """Collect full state to driver. Only for testing — don't use at 38q."""
    n_chunks = result["n_chunks"]
    n_partitions = result["n_partitions"]
    local_dir = result["local_dir"]
    buf = result["committed_buf"]

    def read_partition(args):
        start, end, ld, b = args
        from wenbo_engine.storage.block_store import read_chunk, chunk_filename
        src_dir = Path(ld) / f"state_{b}" / "chunks"
        return [(ci, read_chunk(src_dir / chunk_filename(ci)).tobytes())
                for ci in range(start, end)]

    tasks = []
    for p in range(n_partitions):
        s, e = _partition_range(p, n_chunks, n_partitions)
        tasks.append((s, e, local_dir, buf))

    all_chunks = sc.parallelize(tasks, numSlices=n_partitions) \
                   .flatMap(read_partition).collect()
    all_chunks.sort(key=lambda x: x[0])
    arrays = [np.frombuffer(raw, dtype=DTYPE) for _, raw in all_chunks]
    return np.concatenate(arrays).astype(np.complex128)
