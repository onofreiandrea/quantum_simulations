"""Fault injector — the decision layer for deterministic crash testing.

Reads a config dict (typically loaded from JSON) and, when invoked at a named
:class:`~wenbo_engine.faults.fault_points.FaultPoint`, decides whether to fire
a crash for the current ``(rank, stage_id, fault_point)``.

Config schema::

    {
      "fault_injection": {
        "enabled": true,
        "rank": 2,
        "stage_id": 4,
        "fault_point": "AFTER_MANIFEST_RENAME",
        "mode": "os_exit"
      }
    }

The injector fires ONLY when **all** of these hold:
  * ``enabled`` is true, AND
  * the current rank matches ``rank``, AND
  * the current stage_id matches ``stage_id``, AND
  * the current fault_point matches ``fault_point``.

When disabled (or unmatched) the :meth:`maybe_fire` hot path is a couple of
attribute reads and integer compares — negligible overhead, and zero when the
injector is the shared :data:`NULL_INJECTOR`.

``rank`` / ``stage_id`` may be ``null``/omitted to mean "any" (wildcard); a
specific value matches only that rank/stage.  ``fault_point`` is required when
enabled.

Optionally an :class:`~wenbo_engine.recovery.recovery_events.RecoveryEventLog`
can be attached so a fired fault is recorded as a structured event (proof #4 —
``recovery_events.json`` records the fault).
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from wenbo_engine.faults.crash_controller import CrashController, OS_EXIT
from wenbo_engine.faults.fault_points import FaultPoint, parse_fault_point

log = logging.getLogger(__name__)

CONFIG_KEY = "fault_injection"


class FaultInjector:
    """Decides + executes faults from a config dict."""

    def __init__(self, config: dict | None = None, *,
                 controller: CrashController | None = None,
                 events=None):
        fi = (config or {}).get(CONFIG_KEY, {}) if config else {}
        self.enabled: bool = bool(fi.get("enabled", False))
        # rank / stage_id of None ⇒ wildcard ("any").
        self.rank = fi.get("rank", None)
        self.rank = None if self.rank is None else int(self.rank)
        self.stage_id = fi.get("stage_id", None)
        self.stage_id = None if self.stage_id is None else int(self.stage_id)
        self.mode: str = str(fi.get("mode", OS_EXIT))
        self._fault_point: FaultPoint | None = None
        if self.enabled:
            fp = fi.get("fault_point")
            if fp is None:
                raise ValueError(
                    "fault_injection.enabled is true but no fault_point given")
            self._fault_point = parse_fault_point(fp)
        self._controller = controller or CrashController(self.mode)
        self.events = events
        # Set true once a fault has fired (so we never double-fire and so tests
        # can assert a fault occurred even when the crash is a non-terminating
        # stub).
        self.fired = False

    # ── factories ──────────────────────────────────────────────────────

    @classmethod
    def disabled(cls) -> "FaultInjector":
        """A no-op injector (never fires)."""
        return cls(None)

    @classmethod
    def from_json_file(cls, path: str | Path, **kw) -> "FaultInjector":
        with open(path) as f:
            return cls(json.load(f), **kw)

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None, **kw) -> "FaultInjector":
        """Build from ``WE_FAULT_*`` environment variables.

        ``WE_FAULT_POINT`` enables injection; ``WE_FAULT_RANK`` /
        ``WE_FAULT_STAGE`` / ``WE_FAULT_MODE`` refine it.  Used to thread a
        fault config into a subprocess / mpirun run via the environment.
        """
        import os as _os
        env = _os.environ if env is None else env
        point = env.get("WE_FAULT_POINT")
        if not point:
            return cls(None, **kw)      # disabled injector (honours kw, e.g. events)
        rank = env.get("WE_FAULT_RANK")
        stage = env.get("WE_FAULT_STAGE")
        cfg = {CONFIG_KEY: {
            "enabled": True,
            "fault_point": point,
            "rank": int(rank) if rank not in (None, "", "any") else None,
            "stage_id": int(stage) if stage not in (None, "", "any") else None,
            "mode": env.get("WE_FAULT_MODE", OS_EXIT),
        }}
        return cls(cfg, **kw)

    # ── decision + execution ───────────────────────────────────────────

    def will_fire(self, fault_point, rank: int, stage_id: int) -> bool:
        """Pure predicate: would a fault fire here?  No side effects."""
        if not self.enabled or self.fired:
            return False
        if self._fault_point is None or fault_point != self._fault_point:
            return False
        if self.rank is not None and rank != self.rank:
            return False
        if self.stage_id is not None and stage_id != self.stage_id:
            return False
        return True

    def maybe_fire(self, fault_point, rank: int, stage_id: int) -> None:
        """Hot-path hook: fire iff the config matches this exact point.

        Disabled / unmatched ⇒ returns immediately (negligible overhead).
        On a match it records the fault (if an event log is attached) and then
        executes the crash via the :class:`CrashController` — which, in
        ``os_exit`` mode, does not return.
        """
        if not self.enabled or self.fired:
            return
        if not self.will_fire(fault_point, rank, stage_id):
            return
        self.fired = True
        self._record(fault_point, rank, stage_id)
        self._controller.crash(fault_point, rank, stage_id)

    # ── event recording (proof #4) ─────────────────────────────────────

    def _record(self, fault_point, rank: int, stage_id: int) -> None:
        if self.events is None:
            return
        # Lazy import to keep this module free of a hard recovery dependency.
        from wenbo_engine.recovery.recovery_events import EventType
        self.events.emit(
            EventType.FAULT_INJECTED,
            f"injected fault {fault_point} (mode={self.mode})",
            generation=None, rank=rank,
            fault_point=str(fault_point), stage_id=stage_id, mode=self.mode,
        )

    def describe(self) -> dict[str, Any]:
        """JSON-able summary of the configured fault (for artifacts/logging)."""
        return {
            "enabled": self.enabled,
            "rank": self.rank,
            "stage_id": self.stage_id,
            "fault_point": str(self._fault_point) if self._fault_point else None,
            "mode": self.mode,
            "fired": self.fired,
        }


# A shared, always-off injector — use as the default so callers never branch on
# ``None``.  ``maybe_fire`` on it is a single ``self.enabled`` check.
NULL_INJECTOR = FaultInjector.disabled()
