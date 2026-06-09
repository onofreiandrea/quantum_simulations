"""Per-event MPI profiler → mpi_profile.csv.

Records one row per MPI exchange (point-to-point Sendrecv) or collective
(Barrier / Allreduce / etc.).  Works whether or not mpi4py is installed —
if MPI is unavailable the profiler simply never receives events and writes
a header-only CSV.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from threading import Lock

MPI_COLUMNS = [
    "stage_id",
    "kind",        # "sendrecv" | "collective"
    "op",          # e.g. "Sendrecv", "Barrier", "Allreduce"
    "peer",        # partner rank for sendrecv, -1 for collectives
    "bytes_sent",
    "seconds",
    "mb_per_s",
]


@dataclass
class MPIProfiler:
    rank: int = 0
    events: list[tuple] = field(default_factory=list)
    _lock: Lock = field(default_factory=Lock, repr=False)

    def record(self, stage_id, kind: str, op: str,
               bytes_sent: int = 0, seconds: float = 0.0,
               peer: int = -1) -> None:
        with self._lock:
            self.events.append(
                (stage_id, kind, op, int(peer), int(bytes_sent), float(seconds))
            )

    def record_sendrecv(self, stage_id, peer: int,
                        bytes_sent: int, seconds: float) -> None:
        self.record(stage_id, "sendrecv", "Sendrecv",
                    bytes_sent=bytes_sent, seconds=seconds, peer=peer)

    def record_collective(self, stage_id, op: str, seconds: float) -> None:
        self.record(stage_id, "collective", op, seconds=seconds)

    # ── derived ────────────────────────────────────────────────────────
    def totals(self) -> dict:
        total_bytes = 0
        total_sec = 0.0
        for _sid, _kind, _op, _peer, nbytes, seconds in self.events:
            total_bytes += nbytes
            total_sec += seconds
        return {
            "n_events": len(self.events),
            "mpi_bytes_sent": total_bytes,
            "mpi_sec": total_sec,
            "mpi_mb_per_s": (total_bytes / 1e6 / total_sec) if total_sec > 0 else 0.0,
        }

    def to_csv(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(MPI_COLUMNS)
            for stage_id, kind, op, peer, nbytes, seconds in self.events:
                mbps = (nbytes / 1e6 / seconds) if seconds > 0 else 0.0
                w.writerow([stage_id, kind, op, peer, nbytes,
                            f"{seconds:.6f}", f"{mbps:.3f}"])
        return path
