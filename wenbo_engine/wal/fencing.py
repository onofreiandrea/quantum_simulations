"""Fencing lock — prevents concurrent runs on the same work directory.

Uses an atomic lock file with PID + hostname + timestamp.
On startup, checks if the lock holder is still alive.
Works on shared filesystems (BeeGFS, NFS) — no POSIX locks, just atomic rename.
"""
from __future__ import annotations

import json
import os
import platform
import time
from pathlib import Path


class FencingLock:
    """Acquire / release a fencing lock on a work directory."""

    def __init__(self, work_dir: str | Path):
        self.lock_path = Path(work_dir) / "run.lock"
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)

    def acquire(self, force: bool = False) -> None:
        """Acquire the lock. Raises RuntimeError if held by another live process."""
        if self.lock_path.exists() and not force:
            holder = self._read_lock()
            if holder and self._is_alive(holder):
                raise RuntimeError(
                    f"Work directory locked by PID {holder['pid']} on "
                    f"{holder['host']} since {holder.get('time', '?')}. "
                    f"If stale, delete {self.lock_path} or use force=True."
                )
        self._write_lock()

    def release(self) -> None:
        """Release the lock."""
        if self.lock_path.exists():
            self.lock_path.unlink(missing_ok=True)

    def _write_lock(self) -> None:
        info = {
            "pid": os.getpid(),
            "host": platform.node(),
            "time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "ts": time.time(),
        }
        tmp = self.lock_path.with_suffix(".tmp")
        with open(tmp, "w") as f:
            f.write(json.dumps(info))
            f.flush()
            os.fsync(f.fileno())
        os.replace(str(tmp), str(self.lock_path))

    def _read_lock(self) -> dict | None:
        try:
            with open(self.lock_path) as f:
                return json.loads(f.read())
        except (json.JSONDecodeError, OSError):
            return None

    def _is_alive(self, holder: dict) -> bool:
        """Check if the lock holder is still running."""
        # Different host → can't check PID, assume alive
        if holder.get("host") != platform.node():
            # Stale check: if lock is older than 24 hours, assume dead
            age = time.time() - holder.get("ts", 0)
            return age < 86400
        # Same host → check if PID exists
        try:
            os.kill(holder["pid"], 0)
            return True
        except (OSError, ProcessLookupError):
            return False

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *args):
        self.release()
