"""Run a reproducible, fully-profiled experiment.

CLI:
    python -m wenbo_engine.experiments.run_experiment --config configs/smoke_local.yaml

Under MPI:
    mpirun -np 4 python -m wenbo_engine.experiments.run_experiment \
        --config configs/smoke_mpi.yaml

Produces a self-contained ``experiments/<run_id>/`` directory (see the
package docstring for the artifact list).  The simulator's recovery logic is
not modified — the single-node path uses an instrumented copy of the runner
that reuses the real WAL, and the MPI path calls the existing MPI runner
verbatim.
"""
from __future__ import annotations

import argparse
import inspect
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from wenbo_engine.circuit.io import validate_circuit_dict, levelize
from wenbo_engine.storage.block_store import DTYPE, read_chunk
from wenbo_engine.storage.manifest import read_manifest
from wenbo_engine.experiments.config import ExperimentConfig, load_config
from wenbo_engine.experiments import summary as summary_mod
from wenbo_engine.profiling import (
    StageProfiler, IOProfiler, MPIProfiler, CalibrationRunner,
)

_ITEMSIZE = np.dtype(DTYPE).itemsize


# ── circuit construction ───────────────────────────────────────────────────

def build_circuit(cfg: ExperimentConfig) -> dict:
    c = cfg.circuit
    if c.source == "json":
        with open(c.path) as f:
            return validate_circuit_dict(json.load(f))
    if c.source == "qasm":
        from qiskit import QuantumCircuit
        from wenbo_engine.circuit.import_qiskit import qiskit_to_dict
        qc = QuantumCircuit.from_qasm_file(c.path)
        return validate_circuit_dict(qiskit_to_dict(qc))
    # builtin fixture / generator
    from wenbo_engine.tests.fixtures import circuits as fx
    if not hasattr(fx, c.name):
        raise ValueError(f"unknown builtin circuit {c.name!r}")
    fn = getattr(fx, c.name)
    sig = inspect.signature(fn)
    kwargs = {}
    params = dict(c.params or {})
    # default seed if the generator accepts one and the config didn't set it
    params.setdefault("seed", cfg.seed)
    for name, p in sig.parameters.items():
        if name in ("n", "n_qubits"):
            kwargs[name] = c.n_qubits
        elif name in params:
            kwargs[name] = params[name]
    return validate_circuit_dict(fn(**kwargs))


# ── plan compilation (matches the runners' classification) ──────────────────

def compile_plan(circuit_dict: dict, chunk_size: int,
                 n_ranks: int = 1) -> dict:
    import math
    n = circuit_dict["number_of_qubits"]
    k = int(math.log2(chunk_size))
    p = int(math.log2(n_ranks)) if n_ranks > 1 else 0
    n_local_bits = n - k - p
    levels = levelize(circuit_dict)

    steps = []
    for li, lv in enumerate(levels):
        if not lv:
            continue
        local = rank_nl = mpi_nl = 0
        for g in lv:
            qs = g["qubits"]
            if all(q < k for q in qs):
                local += 1
            elif any((q - k) >= n_local_bits for q in qs if q >= k):
                mpi_nl += 1
            else:
                rank_nl += 1
        steps.append({
            "step": len(steps),
            "level": li,
            "local_ops": local,
            "rank_nonlocal_ops": rank_nl,
            "mpi_nonlocal_ops": mpi_nl,
        })
    return {
        "n_qubits": n,
        "n_gates": len(circuit_dict["gates"]),
        "chunk_size": chunk_size,
        "chunk_bits": k,
        "n_chunks": (1 << n) // chunk_size,
        "n_ranks": n_ranks,
        "n_steps": len(steps),
        "steps": steps,
        "totals": {
            "local_ops": sum(s["local_ops"] for s in steps),
            "rank_nonlocal_ops": sum(s["rank_nonlocal_ops"] for s in steps),
            "mpi_nonlocal_ops": sum(s["mpi_nonlocal_ops"] for s in steps),
        },
    }


# ── misc artifact helpers ──────────────────────────────────────────────────

def _git_commit() -> str:
    repo = Path(__file__).resolve().parent.parent.parent

    def _git(*args):
        return subprocess.run(["git", "-C", str(repo), *args],
                              capture_output=True, text=True, timeout=10)
    try:
        head = _git("rev-parse", "HEAD").stdout.strip() or "unknown"
        branch = _git("rev-parse", "--abbrev-ref", "HEAD").stdout.strip()
        dirty = bool(_git("status", "--porcelain").stdout.strip())
    except Exception:
        head, branch, dirty = "unknown", "unknown", False
    return f"commit: {head}\nbranch: {branch}\ndirty: {dirty}\n"


def _state_norm(final_dir: Path) -> float:
    """L2 norm of the state vector, streamed chunk-by-chunk."""
    man = read_manifest(final_dir)
    acc = 0.0
    for cname in man.chunks:
        data = read_chunk(final_dir / "chunks" / cname).astype(np.complex128)
        acc += float(np.sum(np.abs(data) ** 2))
    return float(np.sqrt(acc))


def _make_run_id(cfg: ExperimentConfig, circuit: dict) -> str:
    if cfg.run_id:
        return cfg.run_id
    stamp = time.strftime("%Y%m%d-%H%M%S")
    return f"{stamp}_{cfg.runner}_{cfg.circuit.name}_{circuit['number_of_qubits']}q"


# ── main orchestration ─────────────────────────────────────────────────────

def run_experiment(cfg: ExperimentConfig, *, run_id: str | None = None,
                   comm=None) -> Path:
    """Execute one experiment, returning the run directory."""
    cfg.validate()
    if run_id:
        cfg.run_id = run_id

    # MPI context (single-process by default)
    rank, n_ranks = 0, 1
    if cfg.runner == "mpi":
        from mpi4py import MPI
        comm = comm or MPI.COMM_WORLD
        rank, n_ranks = comm.Get_rank(), comm.Get_size()

    circuit = build_circuit(cfg)
    n = circuit["number_of_qubits"]
    chunk_size = cfg.resolved_chunk_size(n)

    run_id = _make_run_id(cfg, circuit)
    if comm is not None:
        run_id = comm.bcast(run_id, root=0)
    run_dir = Path(cfg.output_dir) / run_id

    if rank == 0:
        run_dir.mkdir(parents=True, exist_ok=True)
    if comm is not None:
        comm.Barrier()

    work_dir = Path(cfg.work_dir) if cfg.work_dir else (run_dir / "work")

    # ── static artifacts (rank 0) ──────────────────────────────────────
    if rank == 0:
        (run_dir / "git_commit.txt").write_text(_git_commit())
        cfg_out = cfg.to_dict()
        cfg_out["run_id"] = run_id
        cfg_out["planner_mode"] = getattr(cfg, "planner", None) or "current"
        cfg_out["storage_layout"] = getattr(cfg, "storage_layout", "chunks")
        cfg_out["execution_mode"] = getattr(cfg, "execution_mode", "step")
        (run_dir / "config.json").write_text(json.dumps(cfg_out, indent=2))
        (run_dir / "circuit.json").write_text(json.dumps(circuit, indent=2, default=str))
        if cfg.circuit.source == "qasm" and cfg.circuit.path:
            try:
                (run_dir / "circuit.qasm").write_text(Path(cfg.circuit.path).read_text())
            except OSError:
                pass
        plan = compile_plan(circuit, chunk_size, n_ranks)
        (run_dir / "plan.json").write_text(json.dumps(plan, indent=2))

        # Optimizer-v2 ablation report (deterministic plan metrics for all
        # modes).  Always written so the data-movement comparison is
        # available alongside the run; selecting a mode is done via the
        # --planner CLI flag, which is recorded in config.json.
        planner_mode = getattr(cfg, "planner", None)
        try:
            import math as _math
            from wenbo_engine.planner import HardwareConfig, ablation_report
            hw = HardwareConfig(
                n_qubits=n, chunk_bits=int(_math.log2(chunk_size)),
                num_ranks=n_ranks, recovery=cfg.resolved_recovery())
            report = ablation_report(circuit, hw, verify_norm=False)
            report["selected_mode"] = planner_mode or "current"
            (run_dir / "ablation_report.json").write_text(
                json.dumps(report, indent=2, default=str))
        except Exception as e:  # pragma: no cover - defensive
            (run_dir / "ablation_report.json").write_text(
                json.dumps({"error": repr(e)}, indent=2))

    # ── calibration (all ranks; MPI parts are collective) ──────────────
    if cfg.calibrate:
        calib_dir = work_dir / f"calib_rank{rank}"
        runner = CalibrationRunner(calib_dir, chunk_size=min(chunk_size, 1 << 20),
                                   n_chunks=cfg.calib_chunks, comm=comm)
        cost_model = runner.run(include_mpi=(cfg.runner == "mpi"))
        if rank == 0:
            (run_dir / "cost_model.json").write_text(json.dumps(cost_model, indent=2))
    elif rank == 0:
        (run_dir / "cost_model.json").write_text(json.dumps({"calibrated": False}, indent=2))

    if comm is not None:
        comm.Barrier()

    # ── simulate ───────────────────────────────────────────────────────
    wall0 = time.perf_counter()
    if cfg.runner == "single_node":
        result = _run_single_node(cfg, circuit, work_dir, chunk_size, run_dir)
        wall = time.perf_counter() - wall0
        norm = _state_norm(result.final_dir)
        recovery = {
            "wal_enabled": cfg.use_wal,
            "resumed": result.resumed,
            "start_step": result.start_step,
            "n_steps": result.n_steps,
            "crash_env": _crash_env(),
            "events": ([{"type": "wal_resume", "from_step": result.start_step}]
                       if result.resumed else []),
        }
    else:
        norm, wall_mpi, mpi_recovery = _run_mpi(
            cfg, circuit, work_dir, chunk_size, run_dir, comm, n_ranks, rank)
        wall = wall_mpi
        recovery = {
            "wal_enabled": cfg.use_wal,
            "recovery_mode": cfg.resolved_recovery(),
            "crash_env": _crash_env(),
            "note": "MPI per-stage timings are not instrumented; see final_summary.",
            "events": [],
        }
        recovery.update(mpi_recovery)

    # ── final artifacts (rank 0) ───────────────────────────────────────
    if rank == 0:
        (run_dir / "final_norm.txt").write_text(f"{norm:.12f}\n")
        (run_dir / "recovery_events.json").write_text(json.dumps(recovery, indent=2))
        summary_mod.write_summary(run_dir, extra={
            "run_id": run_id,
            "runner": cfg.runner,
            "n_qubits": n,
            "n_gates": len(circuit["gates"]),
            "chunk_size": chunk_size,
            "n_ranks": n_ranks,
            "recovery_mode": cfg.resolved_recovery(),
            "storage_layout": getattr(cfg, "storage_layout", "chunks"),
            "execution_mode": getattr(cfg, "execution_mode", "step"),
            "wall_sec": round(wall, 6),
            "final_norm": norm,
        })
    if comm is not None:
        comm.Barrier()
    return run_dir


def _crash_env() -> dict:
    return {k: v for k, v in os.environ.items()
            if k.startswith("WE_CRASH") or k.startswith("WE_FAULT")}


def _as_bool(val) -> bool:
    if isinstance(val, bool):
        return val
    return str(val).strip().lower() in ("1", "true", "yes", "on")


def _apply_durable_overrides(cfg: ExperimentConfig, args) -> None:
    """Merge --durable.* CLI flags into cfg.durable (CLI wins over YAML)."""
    d = dict(cfg.durable or {})
    if args.durable_enabled is not None:
        d["enabled"] = _as_bool(args.durable_enabled)
    if args.durable_backend is not None:
        d["backend"] = args.durable_backend
    if args.durable_root is not None:
        d["root"] = args.durable_root
    if args.durable_interval is not None:
        d["interval_generations"] = args.durable_interval
    cfg.durable = d


def _run_single_node(cfg, circuit, work_dir, chunk_size, run_dir):
    from wenbo_engine.experiments import instrumented_runner as ir
    io = IOProfiler()
    mpi = MPIProfiler()
    stage = StageProfiler(io_profiler=io, mpi_profiler=mpi)

    result = ir.run(circuit, work_dir, chunk_size=chunk_size,
                    kernel=cfg.kernel, use_wal=cfg.use_wal,
                    use_fusion=cfg.use_fusion, profiler=stage,
                    checksum=cfg.checksum)

    stage.to_csv(run_dir / "stage_profile.csv")
    io.to_csv(run_dir / "io_profile.csv")
    mpi.to_csv(run_dir / "mpi_profile.csv")
    return result


def _run_mpi(cfg, circuit, work_dir, chunk_size, run_dir, comm, n_ranks, rank):
    """Run the (untouched) MPI runner and emit op-count stage artifacts.

    The MPI runner's hot path is not instrumented (that would require editing
    its recovery-adjacent step loop), so per-stage *timings* are left at 0 and
    flagged in recovery_events / final_summary.  Op counts, byte estimates,
    the global final norm, and the real MPI calibration ARE recorded.

    Returns ``(norm, wall, recovery_extra)`` where ``recovery_extra`` is merged
    into the run's recovery_events.json (real scanner events in generation mode).
    """
    from wenbo_engine.mpi import mpi_runner

    mode = cfg.resolved_recovery()
    run_id = run_dir.name

    # Durable R4: if enabled and the local work_dir has no committed generation
    # (e.g. lost NVMe), restore the newest valid durable generation BEFORE the
    # run so the generation runner resumes from it.  This touches durable
    # storage only here, not on the hot path.
    durable_recovery = _durable_restore_if_needed(
        cfg, work_dir, run_id, comm, mode)

    t0 = time.perf_counter()
    mpi_runner.run(circuit, work_dir, chunk_size=chunk_size,
                   comm=comm, buffer_depth=cfg.buffer_depth,
                   recovery=mode,
                   storage_layout=getattr(cfg, "storage_layout", "chunks"),
                   execution_mode=getattr(cfg, "execution_mode", "step"),
                   compute_unit_min_gates=getattr(cfg, "compute_unit_min_gates", 4))
    comm.Barrier()

    # Durable R4: promote committed generations to durable storage AFTER the
    # run (between/at end of steps — never during gate execution).
    durable_promotion = _durable_promote_after_run(
        cfg, work_dir, run_id, comm, mode)
    wall = comm.allreduce(time.perf_counter() - t0, op=__import__("mpi4py").MPI.MAX)
    norm = mpi_runner.compute_norm(work_dir, comm=comm)

    # Generation mode: capture the real recovery-scanner decisions as events,
    # proving the global commit record (not wal.json) is the source of truth.
    recovery_extra: dict = {"events": []}
    if mode == "generation" and rank == 0:
        from wenbo_engine.recovery import RecoveryScanner
        sc = RecoveryScanner(work_dir).scan(quarantine=False)
        recovery_extra = {
            "recovery_mode": "generation",
            "source_of_truth": "global_commit_record",
            "committed_generation": sc.generation,
            "commits_dir": str(Path(work_dir) / "commits"),
            "wal_json_present": (Path(work_dir) / f"rank_{rank}" / "wal.json").exists(),
            "events": [e.to_dict() for e in sc.events],
        }
    if rank == 0 and (durable_recovery or durable_promotion):
        recovery_extra.setdefault("events", [])
        recovery_extra["durable"] = {
            "restore": durable_recovery,
            "promotion": durable_promotion,
        }

    if rank == 0:
        plan = compile_plan(circuit, chunk_size, n_ranks)
        chunk_bytes = chunk_size * _ITEMSIZE
        n_chunks_per_rank = plan["n_chunks"] // n_ranks
        stage = StageProfiler()
        for s in plan["steps"]:
            with stage.stage(s["step"], local_ops=s["local_ops"],
                             rank_nonlocal_ops=s["rank_nonlocal_ops"],
                             mpi_nonlocal_ops=s["mpi_nonlocal_ops"]) as h:
                # byte estimates: every local chunk read once + written once
                h.add_bytes_read(n_chunks_per_rank * chunk_bytes)
                h.add_bytes_written(n_chunks_per_rank * chunk_bytes)
                h.add_mpi_bytes_sent(
                    s["mpi_nonlocal_ops"] * n_chunks_per_rank * chunk_bytes)
        stage.to_csv(run_dir / "stage_profile.csv")
        IOProfiler().to_csv(run_dir / "io_profile.csv")
        MPIProfiler(rank=rank).to_csv(run_dir / "mpi_profile.csv")
    return norm, wall, recovery_extra


# ── durable checkpoint (R4) wiring ──────────────────────────────────────────

def _durable_enabled(cfg) -> bool:
    return bool((cfg.durable or {}).get("enabled"))


def _build_durable(cfg, work_dir, run_id, comm):
    """Build (DurableConfig, backend, coordinator) for the MPI run."""
    from wenbo_engine.durable import DurableConfig
    from wenbo_engine.recovery.generation_manager import MPICoordinator
    dconf = DurableConfig.from_dict(cfg.durable)
    backend = dconf.build_backend()
    coord = MPICoordinator(comm)
    return dconf, backend, coord


def _durable_restore_if_needed(cfg, work_dir, run_id, comm, mode) -> dict | None:
    """Restore the newest durable generation if no local committed gen exists.

    Returns a small status dict (rank 0) or None when durable is off / nothing
    needed.  Only runs in generation recovery mode.
    """
    if mode != "generation" or not _durable_enabled(cfg):
        return None
    from wenbo_engine.recovery.recovery_scanner import RecoveryScanner
    from wenbo_engine.durable import DurableRestoreManager

    # Does the local work_dir already hold a committed generation?
    local_gen = RecoveryScanner(work_dir).scan(quarantine=False).generation
    needed = comm.bcast(local_gen is None, root=0)
    if not needed:
        return ({"restored": False, "reason": "local committed gen present"}
                if comm.Get_rank() == 0 else None)

    dconf, backend, coord = _build_durable(cfg, work_dir, run_id, comm)
    rm = DurableRestoreManager(work_dir, run_id, backend, coord)
    result = rm.restore_latest(check_checksums=True)
    if comm.Get_rank() == 0:
        return {"restored": result.restored, "generation": result.generation}
    return None


def _durable_promote_after_run(cfg, work_dir, run_id, comm, mode) -> dict | None:
    """Promote committed generations to durable storage after the run.

    Promotes generation 0 and every generation that is a multiple of the
    configured interval AND still present locally (pruning keeps the newest
    few), plus the final committed generation.  Returns a status dict on rank
    0.  Never touches durable storage during gate execution — this is called
    once the runner has returned.
    """
    if mode != "generation" or not _durable_enabled(cfg):
        return None
    from wenbo_engine.recovery.recovery_scanner import RecoveryScanner
    from wenbo_engine.durable import DurableConfig, DurableCheckpointManager

    dconf, backend, coord = _build_durable(cfg, work_dir, run_id, comm)
    cm = DurableCheckpointManager(work_dir, run_id, backend, coord)
    cm.upload_run_metadata()

    final_gen = RecoveryScanner(work_dir).scan(quarantine=False).generation
    final_gen = comm.bcast(final_gen, root=0)
    promoted: list[int] = []
    if final_gen is not None:
        interval = max(1, int(dconf.interval_generations))
        # Candidate generations: interval multiples (and 0) up to final, plus
        # the final one.  Only those whose local dir still exists promote.
        from wenbo_engine.recovery.generation_manager import gen_dir
        candidates = sorted({0, *range(0, final_gen + 1, interval), final_gen})
        for g in candidates:
            # rank 0's local existence is authoritative for the collective
            # (all ranks prune identically, so this never diverges).
            if not comm.bcast(gen_dir(work_dir, 0, g).exists(), root=0):
                continue
            try:
                rec = cm.promote(g)
            except Exception as e:  # pragma: no cover - defensive
                if comm.Get_rank() == 0:
                    print(f"durable promotion of gen {g} skipped: {e}")
                continue
            if rec is not None:
                promoted.append(g)
    if comm.Get_rank() == 0:
        return {"promoted_generations": promoted,
                "interval_generations": dconf.interval_generations,
                "backend": dconf.backend, "root": dconf.root}
    return None


# ── CLI ─────────────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Run a reproducible experiment")
    ap.add_argument("--config", required=True, help="path to YAML/JSON config")
    ap.add_argument("--run-id", default=None, help="override run id")
    ap.add_argument("--output-dir", default=None, help="override output_dir")
    ap.add_argument("--no-calibrate", action="store_true",
                    help="skip machine calibration")
    ap.add_argument("--recovery", choices=["none", "wal", "generation"],
                    default=None,
                    help="crash-recovery mode (overrides config; "
                         "generation requires runner=mpi)")
    # Durable checkpoint (R4) overrides.  These are dotted to mirror the YAML
    # ``durable:`` block; they only take effect with --recovery generation.
    ap.add_argument("--durable.enabled", dest="durable_enabled",
                    default=None,
                    help="enable durable checkpointing (true/false)")
    ap.add_argument("--durable.backend", dest="durable_backend",
                    choices=["local_path", "s3"], default=None,
                    help="durable backend (default local_path)")
    ap.add_argument("--durable.root", dest="durable_root", default=None,
                    help="durable storage root (filesystem path / mount)")
    ap.add_argument("--durable.interval-generations",
                    dest="durable_interval", type=int, default=None,
                    help="promote every N committed generations (default 5)")
    ap.add_argument("--planner", default=None,
                    help="Optimizer-v2 ablation mode recorded in config.json "
                         "and used to select the ablation_report's mode "
                         "(current | current_static_reorder | stage_v2 | "
                         "stage_v2_fusion | stage_v2_placement_fusion)")
    ap.add_argument("--storage-layout", dest="storage_layout",
                    choices=["chunks", "extents"], default="chunks",
                    help="On-disk layout for committed generations: chunks "
                         "(one file per chunk, default) or extents (pack many "
                         "chunks into few extent files). Generation recovery.")
    ap.add_argument("--execution-mode", dest="execution_mode",
                    choices=["step", "compute_unit"], default="step",
                    help="step (default) or compute_unit (RAM-overlay fusion of "
                         "consecutive local-only steps). Generation recovery.")
    ap.add_argument("--compute-unit-min-gates", dest="compute_unit_min_gates",
                    type=int, default=4,
                    help="min local gates to form a compute unit (default 4).")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.no_calibrate:
        cfg.calibrate = False
    if args.recovery is not None:
        cfg.recovery = args.recovery
    _apply_durable_overrides(cfg, args)
    if args.planner is not None:
        cfg.planner = args.planner
    cfg.storage_layout = args.storage_layout
    cfg.execution_mode = args.execution_mode
    cfg.compute_unit_min_gates = args.compute_unit_min_gates
    cfg.validate()

    run_dir = run_experiment(cfg, run_id=args.run_id)

    # Only rank 0 should announce (others may not have written artifacts).
    is_rank0 = True
    if cfg.runner == "mpi":
        from mpi4py import MPI
        is_rank0 = MPI.COMM_WORLD.Get_rank() == 0
    if is_rank0:
        print(f"experiment complete: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
