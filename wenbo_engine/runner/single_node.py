"""Single-node out-of-core runner (double-buffer).

Architecture:
  - Two directories (state_a, state_b) alternate as source and destination.
  - The source buffer is NEVER modified — this is the recovery guarantee.
  - All chunk writes are atomic (tmp + fsync + os.replace).
  - On crash: wipe dst, redo the current step from the intact source.

Crash injection via WE_CRASH_AFTER_CHUNK env var (testing only).
"""
from __future__ import annotations

import logging
import math
import os
import shutil
from pathlib import Path

import numpy as np

from wenbo_engine.circuit.io import validate_circuit_dict, levelize
from wenbo_engine.circuit.fusion import batch_levels
from wenbo_engine.kernel import gates as gmod
from wenbo_engine.kernel.cpu_scalar import apply_1q, apply_2q
from wenbo_engine.kernel.cpu_nonlocal import (
    apply_1q_pair, apply_2q_pair_qa_local,
    apply_2q_pair_qb_local, apply_2q_quad,
)
from wenbo_engine.storage.block_store import (
    read_chunk, write_chunk_atomic, chunk_filename, init_zero_state, DTYPE,
)
from wenbo_engine.storage.manifest import Manifest, write_manifest_atomic, read_manifest
from wenbo_engine.wal.wal import WAL
from wenbo_engine.wal.fencing import FencingLock

log = logging.getLogger(__name__)


# ── helpers ──────────────────────────────────────────────────────────

def _buf_dir(work: Path, buf: str) -> Path:
    return work / f"state_{buf}"


def _other(buf: str) -> str:
    return "b" if buf == "a" else "a"


def _classify_ops(gates: list[dict], k: int):
    local, nonlocal_ = [], []
    for g in gates:
        U = gmod.gate_matrix(g["gate"], g["params"])
        qs = g["qubits"]
        if all(q < k for q in qs):
            local.append((qs, U))
        else:
            nonlocal_.append((qs, U))
    return local, nonlocal_


def _crash_after() -> int | None:
    val = os.environ.get("WE_CRASH_AFTER_CHUNK")
    return int(val) if val is not None else None


def _wipe_buf(buf_dir: Path) -> None:
    """Remove chunks and manifest from a buffer directory."""
    chunks = buf_dir / "chunks"
    if chunks.exists():
        shutil.rmtree(chunks)
    manifest = buf_dir / "manifest.json"
    if manifest.exists():
        manifest.unlink()


# ── public API ───────────────────────────────────────────────────────

def run(
    circuit_dict: dict,
    work_dir: str | Path,
    chunk_size: int = 1 << 20,
    kernel: str = "scalar",
    use_wal: bool = True,
    use_fencing: bool = False,
    use_fusion: bool = False,
    use_staging: bool = False,
    staging_method: str = "heuristic",
) -> Path:
    """Run full circuit out-of-core.  Returns path to final state buffer."""
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
    crash_after = _crash_after()

    if kernel == "batched":
        from wenbo_engine.kernel.cpu_batched import apply_1q as a1, apply_2q as a2
    else:
        a1, a2 = apply_1q, apply_2q

    # Build steps
    log_to_phys = None
    if use_staging:
        from wenbo_engine.circuit.staging import atlas_stages
        steps, log_to_phys = atlas_stages(cd, k, method=staging_method)
    elif use_fusion:
        steps = batch_levels(levels, k)
    else:
        steps = []
        for lv in levels:
            if not lv:
                continue
            lo, nlo = _classify_ops(lv, k)
            steps.append({"local_ops": lo, "nonlocal_ops": nlo})

    fence = FencingLock(work) if use_fencing else None
    if fence:
        fence.acquire()
    try:
        result = _run_inner(cd, n, chunk_size, k, work, steps, crash_after,
                            a1, a2, use_wal)
        # Store qubit mapping if staging produced a non-identity permutation
        if log_to_phys and log_to_phys != list(range(n)):
            import json
            mp = work / "qubit_mapping.json"
            with open(mp, "w") as f:
                json.dump(log_to_phys, f)
        return result
    finally:
        if fence:
            fence.release()


def _run_inner(cd, n, chunk_size, k, work, steps, crash_after, a1, a2,
               use_wal):
    wal = WAL(work / "wal.json", circuit_dict=cd) if use_wal else None

    start_step = wal.done_steps if wal else 0
    current_buf = wal.committed_buf if wal else "a"

    # Init |0…0⟩ in state_a if this is a fresh run
    a_dir = _buf_dir(work, "a")
    if start_step == 0 and not (a_dir / "manifest.json").exists():
        init_zero_state(str(a_dir), n, chunk_size)

    n_chunks = (1 << n) // chunk_size
    man = Manifest(
        n_qubits=n, chunk_size=chunk_size, n_chunks=n_chunks,
        chunks=[chunk_filename(i) for i in range(n_chunks)],
    )

    for step_idx in range(start_step, len(steps)):
        src_buf = current_buf
        dst_buf = _other(src_buf)
        src_dir = _buf_dir(work, src_buf)
        dst_dir = _buf_dir(work, dst_buf)

        # Wipe dst (clears any partial data from a prior crash)
        _wipe_buf(dst_dir)

        step = steps[step_idx]
        _apply_step(src_dir, dst_dir, man,
                     step["local_ops"], step["nonlocal_ops"],
                     k, a1, a2, crash_after)

        write_manifest_atomic(dst_dir, man)
        current_buf = dst_buf
        if wal:
            wal.commit_step(step_idx, current_buf)

    if wal:
        wal.close()
    return _buf_dir(work, current_buf)


# ── step execution ───────────────────────────────────────────────────

def _apply_step(src_dir, dst_dir, man, local_ops, nonlocal_ops,
                k, a1, a2, crash_after):
    """Process one step: non-local groups first, then local-only chunks."""
    (dst_dir / "chunks").mkdir(parents=True, exist_ok=True)

    # Process non-local groups (need multiple chunks loaded at once)
    affected = set()
    if nonlocal_ops:
        affected = _process_nonlocal_groups(
            src_dir, dst_dir, man, local_ops, nonlocal_ops, k, a1, a2,
        )

    # Process remaining local-only chunks
    chunks_written = 0
    for ci in range(man.n_chunks):
        if ci in affected:
            continue
        _process_local_chunk(src_dir, dst_dir, man, ci, local_ops, a1, a2)
        chunks_written += 1
        if crash_after is not None and chunks_written >= crash_after:
            os._exit(1)


def _process_local_chunk(src_dir, dst_dir, man, ci, local_ops, a1, a2):
    """Read one chunk from src, apply local gates, write to dst."""
    data = read_chunk(src_dir / "chunks" / man.chunks[ci])
    for qs, U in local_ops:
        if len(qs) == 1:
            a1(data, qs[0], U)
        else:
            a2(data, qs[0], qs[1], U)
    write_chunk_atomic(dst_dir / "chunks" / man.chunks[ci], data)


def _process_nonlocal_groups(src_dir, dst_dir, man, local_ops, nonlocal_ops,
                              k, a1, a2):
    """Process all non-local chunk groups.  Returns set of affected chunk indices."""
    nl_bits = set()
    for qs, U in nonlocal_ops:
        for q in qs:
            if q >= k:
                nl_bits.add(q - k)
    nl_bits_sorted = sorted(nl_bits)
    mask = sum(1 << b for b in nl_bits_sorted)

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

        # Load group from src
        data = {ci: read_chunk(src_dir / "chunks" / man.chunks[ci])
                for ci in group}

        # Apply local gates
        for ci in group:
            for qs, U in local_ops:
                if len(qs) == 1:
                    a1(data[ci], qs[0], U)
                else:
                    a2(data[ci], qs[0], qs[1], U)

        # Apply non-local gates
        for qs, U in nonlocal_ops:
            _apply_nonlocal(data, qs, U, k)

        # Write group to dst
        for ci in group:
            write_chunk_atomic(dst_dir / "chunks" / man.chunks[ci], data[ci])

    return affected


def _apply_nonlocal(data: dict, qs: list[int], U: np.ndarray, k: int):
    """Apply one non-local gate within a loaded chunk group."""
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
        qa_local, qb_local = qa < k, qb < k
        if qa_local:
            pbit = qb - k
            done = set()
            for ci in data:
                c0 = ci & ~(1 << pbit)
                if c0 in done:
                    continue
                done.add(c0)
                c1 = c0 | (1 << pbit)
                apply_2q_pair_qa_local(data[c0], data[c1], qa, U)
        elif qb_local:
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


# ── utility ──────────────────────────────────────────────────────────

def collect_state(buf_path: str | Path, apply_permutation: bool = False,
                   work_dir: str | Path | None = None) -> np.ndarray:
    """Read all chunks of a buffer back into a single numpy array.

    If apply_permutation=True and a qubit_mapping.json exists in work_dir,
    the state is permuted back to the standard logical qubit order.
    """
    m = read_manifest(buf_path)
    p = Path(buf_path) / "chunks"
    parts = [read_chunk(p / c) for c in m.chunks]
    state = np.concatenate(parts).astype(np.complex128)

    if apply_permutation and work_dir is not None:
        import json
        mp = Path(work_dir) / "qubit_mapping.json"
        if mp.exists():
            with open(mp) as f:
                log_to_phys = json.load(f)
            from wenbo_engine.circuit.staging import permute_state
            state = permute_state(state, log_to_phys)
    return state
