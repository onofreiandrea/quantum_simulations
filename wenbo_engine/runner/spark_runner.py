"""Spark runner: orchestration only — no amplitude shuffles (double-buffer).

Spark parallelises local-gate processing over chunk indices.
Non-local gates are handled by the driver (sequential group processing).
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

from wenbo_engine.circuit.io import validate_circuit_dict, levelize
from wenbo_engine.kernel import gates as gmod
from wenbo_engine.kernel.cpu_nonlocal import (
    apply_1q_pair, apply_2q_pair_qa_local,
    apply_2q_pair_qb_local, apply_2q_quad,
)
from wenbo_engine.storage.block_store import (
    read_chunk, write_chunk_atomic, DTYPE, chunk_filename, init_zero_state,
)
from wenbo_engine.storage.manifest import Manifest, write_manifest_atomic
from wenbo_engine.wal.wal import WAL
from wenbo_engine.runner.single_node import (
    _buf_dir, _other, _wipe_buf, _apply_nonlocal,
)

if TYPE_CHECKING:
    from pyspark import SparkContext


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


def _serialisable_ops(ops):
    return [(qs, U.tobytes(), U.shape) for qs, U in ops]


def _process_chunk(args):
    ci, cname, src_str, dst_str, ops_ser = args
    from wenbo_engine.kernel.cpu_batched import apply_1q, apply_2q
    from wenbo_engine.storage.block_store import read_chunk, write_chunk_atomic
    import numpy as np

    data = read_chunk(f"{src_str}/chunks/{cname}")
    for qubits, ubytes, ushape in ops_ser:
        U = np.frombuffer(ubytes, dtype=np.complex128).reshape(ushape)
        if len(qubits) == 1:
            apply_1q(data, qubits[0], U)
        else:
            apply_2q(data, qubits[0], qubits[1], U)
    Path(f"{dst_str}/chunks").mkdir(parents=True, exist_ok=True)
    write_chunk_atomic(f"{dst_str}/chunks/{cname}", data)
    return ci


def run(
    circuit_dict: dict,
    work_dir: str | Path,
    sc: "SparkContext",
    chunk_size: int = 1 << 20,
    use_wal: bool = True,
) -> Path:
    cd = validate_circuit_dict(circuit_dict)
    n = cd["number_of_qubits"]
    N = 1 << n
    if chunk_size > N:
        chunk_size = N
    if N % chunk_size != 0:
        raise ValueError("2^n must be divisible by chunk_size")

    k = int(math.log2(chunk_size))
    work = Path(work_dir)
    levels = levelize(cd)

    steps = []
    for lv in levels:
        if not lv:
            continue
        lo, nlo = _classify_ops(lv, k)
        steps.append({"local_ops": lo, "nonlocal_ops": nlo})

    wal = WAL(work / "wal.json", circuit_dict=cd) if use_wal else None

    start_step = wal.done_steps if wal else 0
    current_buf = wal.committed_buf if wal else "a"

    a_dir = _buf_dir(work, "a")
    if start_step == 0 and not (a_dir / "manifest.json").exists():
        init_zero_state(str(a_dir), n, chunk_size)

    n_chunks = N // chunk_size
    man = Manifest(
        n_qubits=n, chunk_size=chunk_size, n_chunks=n_chunks,
        chunks=[chunk_filename(i) for i in range(n_chunks)],
    )

    for step_idx in range(start_step, len(steps)):
        src_buf = current_buf
        dst_buf = _other(src_buf)
        src_dir = _buf_dir(work, src_buf)
        dst_dir = _buf_dir(work, dst_buf)

        _wipe_buf(dst_dir)

        step = steps[step_idx]
        local_ops = step["local_ops"]
        nonlocal_ops = step["nonlocal_ops"]

        # Process non-local groups (driver, sequential)
        affected = set()
        if nonlocal_ops:
            affected = _process_nonlocal_groups_driver(
                src_dir, dst_dir, man, local_ops, nonlocal_ops, k,
            )

        # Spark-parallel local-only chunks
        local_chunks = [ci for ci in range(man.n_chunks) if ci not in affected]
        if local_chunks:
            ops_ser = _serialisable_ops(local_ops)
            tasks = [
                (ci, man.chunks[ci], str(src_dir), str(dst_dir), ops_ser)
                for ci in local_chunks
            ]
            rdd = sc.parallelize(tasks, numSlices=max(1, len(tasks)))
            rdd.map(_process_chunk).collect()

        write_manifest_atomic(dst_dir, man)
        current_buf = dst_buf
        if wal:
            wal.commit_step(step_idx, current_buf)

    if wal:
        wal.close()
    return _buf_dir(work, current_buf)


def _process_nonlocal_groups_driver(src_dir, dst_dir, man, local_ops, nonlocal_ops, k):
    """Process non-local groups on the driver.  Returns affected chunk set."""
    from wenbo_engine.kernel.cpu_batched import apply_1q, apply_2q

    nl_bits = set()
    for qs, U in nonlocal_ops:
        for q in qs:
            if q >= k:
                nl_bits.add(q - k)
    nl_bits_sorted = sorted(nl_bits)
    mask = sum(1 << b for b in nl_bits_sorted)

    chunks_src = src_dir / "chunks"
    chunks_dst = dst_dir / "chunks"
    chunks_dst.mkdir(parents=True, exist_ok=True)

    processed_bases = set()
    affected = set()

    for c in range(man.n_chunks):
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
        affected.update(group)

        data = {ci: read_chunk(chunks_src / man.chunks[ci]) for ci in group}
        for ci in group:
            for qs, U in local_ops:
                if len(qs) == 1:
                    apply_1q(data[ci], qs[0], U)
                else:
                    apply_2q(data[ci], qs[0], qs[1], U)
        for qs, U in nonlocal_ops:
            _apply_nonlocal(data, qs, U, k)
        for ci in group:
            write_chunk_atomic(chunks_dst / man.chunks[ci], data[ci])

    return affected
