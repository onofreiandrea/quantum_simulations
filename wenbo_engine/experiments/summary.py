"""Aggregate a run's artifacts into final_summary.json.

Reads the per-stage CSV (and, when present, cost_model.json / plan.json) from
a run directory and produces a compact summary: totals per timing bucket,
bytes moved, achieved throughput, and a rough comparison against the
calibrated cost model.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

_SEC_COLS = ("read_sec", "write_sec", "kernel_sec", "mpi_sec",
             "commit_sec", "checksum_sec")
_INT_COLS = ("local_ops", "rank_nonlocal_ops", "mpi_nonlocal_ops",
             "bytes_read", "bytes_written", "mpi_bytes_sent")


def _read_stage_csv(path: Path) -> list[dict]:
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            row = {}
            for c in _SEC_COLS:
                row[c] = float(r.get(c, 0) or 0)
            for c in _INT_COLS:
                row[c] = int(float(r.get(c, 0) or 0))
            row["step_or_stage_id"] = r.get("step_or_stage_id")
            row["recovery_mode"] = r.get("recovery_mode", "normal")
            rows.append(row)
    return rows


def summarize(run_dir: str | Path) -> dict:
    run_dir = Path(run_dir)
    stage_csv = run_dir / "stage_profile.csv"
    rows = _read_stage_csv(stage_csv) if stage_csv.exists() else []

    totals = {c: 0.0 for c in _SEC_COLS}
    counts = {c: 0 for c in _INT_COLS}
    for r in rows:
        for c in _SEC_COLS:
            totals[c] += r[c]
        for c in _INT_COLS:
            counts[c] += r[c]

    work_sec = sum(totals.values())
    read_bytes = counts["bytes_read"]
    write_bytes = counts["bytes_written"]

    summary: dict = {
        "n_stages": len(rows),
        "n_recovery_stages": sum(1 for r in rows if r["recovery_mode"] != "normal"),
        "seconds": {c: round(totals[c], 6) for c in _SEC_COLS},
        "seconds_total_work": round(work_sec, 6),
        "counts": dict(counts),
        "gate_ops": (counts["local_ops"] + counts["rank_nonlocal_ops"]
                     + counts["mpi_nonlocal_ops"]),
        "throughput": {
            "read_MBps": round(read_bytes / 1e6 / totals["read_sec"], 3)
            if totals["read_sec"] > 0 else 0.0,
            "write_MBps": round(write_bytes / 1e6 / totals["write_sec"], 3)
            if totals["write_sec"] > 0 else 0.0,
            "mpi_MBps": round(counts["mpi_bytes_sent"] / 1e6 / totals["mpi_sec"], 3)
            if totals["mpi_sec"] > 0 else 0.0,
        },
    }

    # Optional: compare achieved vs calibrated bandwidth.
    cm_path = run_dir / "cost_model.json"
    if cm_path.exists():
        try:
            cm = json.loads(cm_path.read_text())
            nvme = cm.get("nvme", {})
            summary["cost_model_comparison"] = {
                "calib_read_MBps": nvme.get("read_bandwidth_MBps"),
                "calib_write_MBps": nvme.get("write_bandwidth_MBps"),
                "achieved_read_MBps": summary["throughput"]["read_MBps"],
                "achieved_write_MBps": summary["throughput"]["write_MBps"],
            }
        except (json.JSONDecodeError, OSError):
            pass

    return summary


def write_summary(run_dir: str | Path, extra: dict | None = None) -> dict:
    """Compute the summary, merge `extra`, write final_summary.json."""
    run_dir = Path(run_dir)
    summary = summarize(run_dir)
    if extra:
        summary.update(extra)
    out = run_dir / "final_summary.json"
    out.write_text(json.dumps(summary, indent=2))
    return summary
