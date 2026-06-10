"""Deterministic fault injection for generation-recovery crash testing.

Makes crash testing *reproducible and reusable*: instead of racing a real
process death, a config selects an exact ``(rank, stage_id, fault_point)`` at
which the commit protocol will crash, so the recovery proofs (recover previous
gen before commit, recover new gen after commit, no partial generation ever
accepted) can be asserted deterministically.

Layers
------
* :mod:`~wenbo_engine.faults.fault_points`   — the named protocol positions.
* :mod:`~wenbo_engine.faults.fault_injector` — reads config, decides to fire.
* :mod:`~wenbo_engine.faults.crash_controller` — executes the crash.

The injector is threaded through the real commit protocol
(:mod:`wenbo_engine.recovery.generation_manager`) and the MPI runner; when no
fault is configured every hook is a single ``enabled`` check (negligible
overhead) and all existing behaviour is preserved.
"""
from __future__ import annotations

from wenbo_engine.faults.fault_points import (
    FaultPoint, ALL_FAULT_POINTS, GLOBAL_COMMIT_BOUNDARY, parse_fault_point,
)
from wenbo_engine.faults.crash_controller import (
    CrashController, InjectedFault, CRASH_MODES, OS_EXIT, EXCEPTION,
)
from wenbo_engine.faults.fault_injector import (
    FaultInjector, NULL_INJECTOR, CONFIG_KEY,
)

__all__ = [
    "FaultPoint", "ALL_FAULT_POINTS", "GLOBAL_COMMIT_BOUNDARY",
    "parse_fault_point",
    "CrashController", "InjectedFault", "CRASH_MODES", "OS_EXIT", "EXCEPTION",
    "FaultInjector", "NULL_INJECTOR", "CONFIG_KEY",
]
