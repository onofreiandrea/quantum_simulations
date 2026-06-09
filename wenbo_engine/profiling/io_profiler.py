"""Per-event disk I/O profiler → io_profile.csv.

Records one row per read/write operation: which stage it belonged to, the
direction, the number of bytes, and the wall-clock seconds spent in the
call.  Accumulation is lock-guarded so the pipelined runners (which read /
compute / write on separate threads) can share a single profiler safely.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock

IO_COLUMNS = [
    "stage_id",
    "direction",   # "read" | "write"
    "bytes",
    "seconds",
    "mb_per_s",
]


@dataclass
class IOProfiler:
    events: list[tuple] = field(default_factory=list)
    _lock: Lock = field(default_factory=Lock, repr=False)

    def record(self, stage_id, direction: str, nbytes: int, seconds: float) -> None:
        with self._lock:
            self.events.append((stage_id, direction, int(nbytes), float(seconds)))

    # ── derived ────────────────────────────────────────────────────────
    def totals(self) -> dict:
        read_bytes = write_bytes = 0
        read_sec = write_sec = 0.0
        for _sid, direction, nbytes, seconds in self.events:
            if direction == "read":
                read_bytes += nbytes
                read_sec += seconds
            else:
                write_bytes += nbytes
                write_sec += seconds
        return {
            "n_events": len(self.events),
            "read_bytes": read_bytes,
            "write_bytes": write_bytes,
            "read_sec": read_sec,
            "write_sec": write_sec,
            "read_mb_per_s": (read_bytes / 1e6 / read_sec) if read_sec > 0 else 0.0,
            "write_mb_per_s": (write_bytes / 1e6 / write_sec) if write_sec > 0 else 0.0,
        }

    def to_csv(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(IO_COLUMNS)
            for stage_id, direction, nbytes, seconds in self.events:
                mbps = (nbytes / 1e6 / seconds) if seconds > 0 else 0.0
                w.writerow([stage_id, direction, nbytes,
                            f"{seconds:.6f}", f"{mbps:.3f}"])
        return path
