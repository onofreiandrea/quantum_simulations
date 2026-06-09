#!/usr/bin/env python3
"""MPI-nonlocal communication benchmark suite.

The 38-40 qubit benchmark circuits were *reordered* (see
``wenbo_engine.circuit.reorder``) so that, after static qubit reordering,
zero gates landed on rank bits — every gate executed locally or
rank-nonlocally and no MPI traffic was generated.  That is the right
thing to do for production, but it means inter-node communication was
never actually exercised.

This module produces workloads that **deliberately force** the three
gate classes the MPI runner distinguishes, and measures the resulting
communication.  Reordering is therefore *off* by default here — the
whole point is to keep MPI-nonlocal gates MPI-nonlocal.

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
``rank_nonlocal_heavy(n, depth, chunk_bits, seed)``
    Gates touch chunk-index bits on the same rank (no MPI).
``mpi_nonlocal_heavy(n, depth, chunk_bits, num_ranks, seed)``
    Gates touch rank bits, forcing inter-node Sendrecv.
``mixed_staged(n, depth, chunk_bits, num_ranks, seed)``
    Phased: a block of local, then rank-nonlocal, then MPI-nonlocal gates.

CLI
---
    mpirun -np 4 python -m wenbo_engine.bench.communication_workloads \
        --kind mpi_nonlocal_heavy --n 24 --depth 20

Writes a profile artifact (JSON) with the required metrics:
mpi_nonlocal_gate_count, rank_nonlocal_gate_count, partner_rank_pairs,
mpi_bytes_sent, mpi_sendrecv_time, mpi_wait_time, bytes_read,
bytes_written, kernel_time, stage_time.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import logging
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock

import numpy as np

log = logging.getLogger(__name__)

# Reserved rank bits assumed by generators that are not told ``num_ranks``
# (``rank_nonlocal_heavy``).  Keeping the top ``DEFAULT_RANK_BITS`` qubits
# out of the rank-nonlocal pool guarantees the generated gates stay
# rank-local for any run with ``num_ranks <= 2**DEFAULT_RANK_BITS``.
DEFAULT_RANK_BITS = 2


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


def _pick(rng: np.random.RandomState, lo: int, hi: int, exclude: int | None = None) -> int:
    """Random qubit in [lo, hi); raises if the range is empty after exclusion."""
    if hi <= lo:
        raise ValueError(f"empty qubit range [{lo}, {hi})")
    while True:
        q = int(rng.randint(lo, hi))
        if q != exclude:
            return q


# ── generators ─────────────────────────────────────────────────────────

def communication_light(n: int, depth: int, seed: int = 42) -> dict:
    """Mostly low-bit gates → low MPI traffic.

    ~85% of gates act on the lowest ``max(1, n//4)`` qubits (which map to
    chunk-local positions under any reasonable partition); the remaining
    gates are local 2-qubit entanglers between adjacent low qubits.  No
    gate is placed on a high qubit, so MPI traffic stays minimal.
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
    return {"number_of_qubits": n, "gates": gates}


def rank_nonlocal_heavy(n: int, depth: int, chunk_bits: int,
                        seed: int = 42) -> dict:
    """Gates touch chunk-index bits on the *same* rank (no MPI).

    Each gate is a 2-qubit entangler between a local qubit (``< chunk_bits``)
    and a chunk-index qubit drawn from ``[chunk_bits, n - DEFAULT_RANK_BITS)``.
    Reserving the top ``DEFAULT_RANK_BITS`` qubits keeps these rank-local for
    any run with ``num_ranks <= 2**DEFAULT_RANK_BITS``.
    """
    k = chunk_bits
    hi = n - DEFAULT_RANK_BITS  # exclusive upper bound of rank-local chunk bits
    if k < 1:
        raise ValueError("chunk_bits must be >= 1 (need a local partner qubit)")
    if hi <= k:
        raise ValueError(
            f"no rank-local chunk bits available: need n > chunk_bits + "
            f"{DEFAULT_RANK_BITS} (got n={n}, chunk_bits={k})")
    rng = np.random.RandomState(seed)
    gates: list[dict] = []
    for _ in range(depth):
        local_q = _pick(rng, 0, k)
        chunk_q = _pick(rng, k, hi)
        gates.append(_two(rng, local_q, chunk_q))
    return {"number_of_qubits": n, "gates": gates}


def mpi_nonlocal_heavy(n: int, depth: int, chunk_bits: int,
                       num_ranks: int, seed: int = 42) -> dict:
    """Gates deliberately touch rank bits, forcing MPI Sendrecv.

    The top ``p = log2(num_ranks)`` qubits are rank bits.  Each gate is a
    2-qubit entangler between a rank-bit qubit (``>= n - p``) and a local
    qubit (``< chunk_bits``), guaranteeing an inter-rank exchange.
    """
    _check_pow2(num_ranks)
    k = chunk_bits
    p = int(round(math.log2(num_ranks)))
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
    for _ in range(depth):
        rank_q = _pick(rng, n - p, n)
        local_q = _pick(rng, 0, k)
        gates.append(_two(rng, rank_q, local_q))
    return {"number_of_qubits": n, "gates": gates}


def mixed_staged(n: int, depth: int, chunk_bits: int,
                 num_ranks: int, seed: int = 42) -> dict:
    """Phased workload: local block, then rank-nonlocal, then MPI-nonlocal.

    ``depth`` gates are split roughly into thirds.  The phase boundaries
    are deterministic for a fixed seed.  Useful for observing how a stage
    profiler attributes time as the communication intensity ramps up.
    """
    _check_pow2(num_ranks)
    k = chunk_bits
    p = int(round(math.log2(num_ranks)))
    third = max(1, depth // 3)
    n_local = third
    n_rank = third
    n_mpi = depth - n_local - n_rank

    light = communication_light(n, n_local, seed)
    rank = rank_nonlocal_heavy(n, n_rank, chunk_bits, seed + 1)
    mpi = mpi_nonlocal_heavy(n, max(n_mpi, 0), chunk_bits, num_ranks, seed + 2)

    return {
        "number_of_qubits": n,
        "gates": light["gates"] + rank["gates"] + mpi["gates"],
    }


GENERATORS = {
    "communication_light": communication_light,
    "rank_nonlocal_heavy": rank_nonlocal_heavy,
    "mpi_nonlocal_heavy": mpi_nonlocal_heavy,
    "mixed_staged": mixed_staged,
}


def build_circuit(kind: str, n: int, depth: int, chunk_bits: int,
                  num_ranks: int, seed: int = 42) -> dict:
    """Dispatch to a generator, supplying only the args it accepts."""
    if kind == "communication_light":
        return communication_light(n, depth, seed)
    if kind == "rank_nonlocal_heavy":
        return rank_nonlocal_heavy(n, depth, chunk_bits, seed)
    if kind == "mpi_nonlocal_heavy":
        return mpi_nonlocal_heavy(n, depth, chunk_bits, num_ranks, seed)
    if kind == "mixed_staged":
        return mixed_staged(n, depth, chunk_bits, num_ranks, seed)
    raise ValueError(f"unknown workload kind: {kind!r} (choices: {list(GENERATORS)})")


# ── static gate classification ─────────────────────────────────────────

def _check_pow2(num_ranks: int) -> None:
    if num_ranks < 1 or (num_ranks & (num_ranks - 1)) != 0:
        raise ValueError(f"num_ranks must be a power of 2, got {num_ranks}")


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

    Each MPI qubit maps to a rank bit ``(q - k) - n_local_bits``.  A gate
    touching rank bits ``B`` makes every rank exchange with every other
    rank in the 2^|B| group reachable by flipping bits in ``B`` (the
    runner does pairwise Sendrecv within that group).
    """
    rank_bits = sorted({(q - k) - n_local_bits
                        for q in qubits if (q - k) >= n_local_bits})
    if not rank_bits:
        return set()
    num_ranks = 1 << p
    pairs: set[frozenset[int]] = set()
    for r in range(num_ranks):
        # Enumerate the group reachable by flipping any subset of rank_bits.
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


def classify_circuit(circuit_dict: dict, chunk_bits: int,
                     num_ranks: int) -> dict:
    """Statically count gate classes and the communication topology.

    Returns ``mpi_nonlocal_gate_count``, ``rank_nonlocal_gate_count``,
    ``local_gate_count``, ``partner_rank_pairs`` (number of distinct
    unordered rank pairs that exchange data), and the derived layout
    parameters.
    """
    _check_pow2(num_ranks)
    n = circuit_dict["number_of_qubits"]
    k = chunk_bits
    p = int(round(math.log2(num_ranks)))
    n_local_bits = n - k - p
    if n_local_bits < 0:
        raise ValueError(
            f"invalid layout: n - chunk_bits - log2(num_ranks) = "
            f"{n_local_bits} < 0 (n={n}, chunk_bits={k}, num_ranks={num_ranks})")

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


# ── runtime metrics ────────────────────────────────────────────────────

@dataclass
class Metrics:
    """Per-rank runtime counters, accumulated during a workload run.

    All increment paths take ``lock`` because the MPI runner reads/writes
    chunks and applies kernels from background pipeline threads.
    """
    mpi_bytes_sent: int = 0
    mpi_sendrecv_time: float = 0.0
    mpi_wait_time: float = 0.0
    sendrecv_count: int = 0
    bytes_read: int = 0
    bytes_written: int = 0
    kernel_time: float = 0.0
    stage_time: float = 0.0
    observed_partner_pairs: set = field(default_factory=set)
    lock: Lock = field(default_factory=Lock, repr=False, compare=False)

    def to_dict(self) -> dict:
        # Built by hand (not dataclasses.asdict) because the Lock field is
        # not deep-copyable / picklable, and the result is sent over MPI.
        return {
            "mpi_bytes_sent": self.mpi_bytes_sent,
            "mpi_sendrecv_time": self.mpi_sendrecv_time,
            "mpi_wait_time": self.mpi_wait_time,
            "sendrecv_count": self.sendrecv_count,
            "bytes_read": self.bytes_read,
            "bytes_written": self.bytes_written,
            "kernel_time": self.kernel_time,
            "stage_time": self.stage_time,
            "observed_partner_pairs": sorted(
                sorted(pr) for pr in self.observed_partner_pairs),
        }


class ProfilingComm:
    """Transparent proxy around an MPI communicator that times Sendrecv.

    Everything except ``Sendrecv`` / ``Barrier`` is forwarded unchanged to
    the wrapped communicator via ``__getattr__``, so the runner sees a
    normal comm.  ``Sendrecv`` is timed and its bytes counted;
    ``Barrier`` time is attributed to ``mpi_wait_time`` (the runner uses
    blocking Sendrecv, so synchronization cost shows up at the barriers).
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
        return r

    def Barrier(self):
        t0 = time.perf_counter()
        r = self._comm.Barrier()
        with self._m.lock:
            self._m.mpi_wait_time += time.perf_counter() - t0
        return r


@contextlib.contextmanager
def _instrument_runner(metrics: Metrics):
    """Monkeypatch the MPI runner's chunk I/O and kernel calls to count.

    The kernels themselves stay pure — we wrap the *names the runner
    resolves* (module globals), so no kernel, I/O, or MPI code is edited.
    Originals are restored on exit.
    """
    from wenbo_engine.mpi import mpi_runner as mr

    io_names = ("read_chunk", "write_chunk_atomic")
    kernel_names = ("apply_1q", "apply_2q", "apply_1q_pair",
                    "apply_2q_pair_qa_local", "apply_2q_pair_qb_local",
                    "apply_2q_quad")
    saved = {name: getattr(mr, name) for name in io_names + kernel_names}

    def wrap_read(orig):
        def w(path):
            data = orig(path)
            with metrics.lock:
                metrics.bytes_read += int(np.asarray(data).nbytes)
            return data
        return w

    def wrap_write(orig):
        def w(path, data):
            with metrics.lock:
                metrics.bytes_written += int(np.asarray(data).nbytes)
            return orig(path, data)
        return w

    def wrap_kernel(orig):
        def w(*a, **k):
            t0 = time.perf_counter()
            r = orig(*a, **k)
            with metrics.lock:
                metrics.kernel_time += time.perf_counter() - t0
            return r
        return w

    try:
        mr.read_chunk = wrap_read(saved["read_chunk"])
        mr.write_chunk_atomic = wrap_write(saved["write_chunk_atomic"])
        for name in kernel_names:
            setattr(mr, name, wrap_kernel(saved[name]))
        yield
    finally:
        for name, fn in saved.items():
            setattr(mr, name, fn)


# ── run + aggregate ────────────────────────────────────────────────────

def run_workload(kind: str, n: int, depth: int, chunk_bits: int,
                 work_dir: str | Path, comm=None, seed: int = 42,
                 use_wal: bool = True, verify: bool = False) -> dict:
    """Run a communication workload under instrumentation.

    Builds the circuit (no reordering — see module docstring), runs it on
    the MPI runner with a ``ProfilingComm`` and patched I/O/kernels, then
    gathers per-rank metrics onto rank 0.  Returns the assembled result
    dict on rank 0 and ``None`` on other ranks.
    """
    from mpi4py import MPI
    from wenbo_engine.mpi.mpi_runner import run, collect_state

    if comm is None:
        comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    num_ranks = comm.Get_size()

    chunk_size = 1 << chunk_bits
    cd = build_circuit(kind, n, depth, chunk_bits, num_ranks, seed)
    static = classify_circuit(cd, chunk_bits, num_ranks)

    metrics = Metrics()
    pcomm = ProfilingComm(comm, metrics)

    comm.Barrier()
    t0 = time.perf_counter()
    with _instrument_runner(metrics):
        run(cd, work_dir, chunk_size=chunk_size, use_wal=use_wal, comm=pcomm)
    metrics.stage_time = time.perf_counter() - t0
    comm.Barrier()

    correct = None
    if verify:
        from wenbo_engine.kernel.ref_dense import simulate
        got = collect_state(work_dir, comm)
        if rank == 0:
            ref = simulate(cd)
            correct = bool(np.allclose(got, ref, atol=1e-5))

    # Gather per-rank metric dicts onto rank 0.
    all_metrics = comm.gather(metrics.to_dict(), root=0)
    if rank != 0:
        return None

    aggregate = _aggregate(all_metrics)
    result = {
        "kind": kind,
        "n": n,
        "depth": depth,
        "seed": seed,
        "chunk_bits": chunk_bits,
        "num_ranks": num_ranks,
        "use_wal": use_wal,
        "mpi_nonlocal_gate_count": static["mpi_nonlocal_gate_count"],
        "rank_nonlocal_gate_count": static["rank_nonlocal_gate_count"],
        "local_gate_count": static["local_gate_count"],
        "partner_rank_pairs": static["partner_rank_pairs"],
        "n_local_bits": static["n_local_bits"],
        "aggregate": aggregate,
        "per_rank": all_metrics,
    }
    if correct is not None:
        result["correct"] = correct
    return result


def _aggregate(all_metrics: list[dict]) -> dict:
    """Combine per-rank metrics: sum counters/bytes, max wall-clock times."""
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
        "bytes_read": s("bytes_read"),
        "bytes_written": s("bytes_written"),
        "kernel_time": mx("kernel_time"),
        "stage_time": mx("stage_time"),
        "observed_partner_pairs": len(pair_set),
    }


# ── CLI ────────────────────────────────────────────────────────────────

def _default_chunk_bits(n: int, num_ranks: int) -> int:
    """Pick chunk_bits that leaves rank bits + a couple of rank-nonlocal bits.

    Targets ``n_local_bits = 2`` so that the top ``p`` qubits are rank
    bits and the next two are rank-nonlocal — enough to exercise all three
    gate classes.  Clamped so the layout stays valid for small ``n``.
    """
    p = int(round(math.log2(max(num_ranks, 1))))
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
    ap.add_argument("--work-dir", type=str,
                    default="/tmp/wenbo_comm_bench")
    ap.add_argument("--output", type=str, default=None,
                    help="JSON artifact path (default: <work-dir>/<kind>_profile.json)")
    ap.add_argument("--no-wal", action="store_true", help="disable WAL")
    ap.add_argument("--verify", action="store_true",
                    help="collect full state and compare to ref_dense (small n only)")
    args = ap.parse_args(argv)

    chunk_bits = args.chunk_bits
    if chunk_bits is None:
        chunk_bits = _default_chunk_bits(args.n, num_ranks)

    if rank == 0:
        log.info("=" * 60)
        log.info("  communication workload: %s", args.kind)
        log.info("  n=%d depth=%d chunk_bits=%d ranks=%d seed=%d",
                 args.n, args.depth, chunk_bits, num_ranks, args.seed)
        log.info("=" * 60)

    result = run_workload(
        kind=args.kind, n=args.n, depth=args.depth, chunk_bits=chunk_bits,
        work_dir=args.work_dir, comm=comm, seed=args.seed,
        use_wal=not args.no_wal, verify=args.verify,
    )

    if rank == 0:
        agg = result["aggregate"]
        log.info("  mpi_nonlocal_gate_count : %d", result["mpi_nonlocal_gate_count"])
        log.info("  rank_nonlocal_gate_count: %d", result["rank_nonlocal_gate_count"])
        log.info("  partner_rank_pairs      : %d", result["partner_rank_pairs"])
        log.info("  mpi_bytes_sent          : %d", agg["mpi_bytes_sent"])
        log.info("  mpi_sendrecv_time       : %.4fs", agg["mpi_sendrecv_time"])
        log.info("  mpi_wait_time           : %.4fs", agg["mpi_wait_time"])
        log.info("  bytes_read              : %d", agg["bytes_read"])
        log.info("  bytes_written           : %d", agg["bytes_written"])
        log.info("  kernel_time             : %.4fs", agg["kernel_time"])
        log.info("  stage_time              : %.4fs", agg["stage_time"])
        if "correct" in result:
            log.info("  correct (vs ref_dense)  : %s", result["correct"])

        output = args.output or str(
            Path(args.work_dir) / f"{args.kind}_profile.json")
        Path(output).parent.mkdir(parents=True, exist_ok=True)
        Path(output).write_text(json.dumps(result, indent=2))
        log.info("  artifact written to %s", output)


if __name__ == "__main__":
    main()
