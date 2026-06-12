"""Process-local runtime timing accumulators (calibrated-cost-model telemetry).

A tiny shared bag of wall-clock timers any hot-path layer can add to without
threading a profiler object through every call.  Mirrors the existing
module-level telemetry pattern (``_RAM`` / ``_MPI_TELE`` / ``_WIN_EXEC`` in
``mpi_runner``): the runner resets it at the start of a run and snapshots it at
the end into ``timing_metrics.json``.

This is **measurement only** — adding to a timer never changes execution, and
when nothing instruments a timer its value is simply absent from the snapshot
(the reporter renders missing timers as ``null`` with a reason, never silently
omitted).

Timers are keyed by the final-summary field name, e.g.::

    from wenbo_engine.profiling import runtime_timers as rt
    with rt.timed("commit_time"):
        gm.commit_step(...)

Counts (e.g. number of commits, gates applied) live alongside times under the
same dict so the calibration step can divide one by the other.
"""
from __future__ import annotations

import contextlib
from time import perf_counter

# field name -> accumulated seconds (or an integer count for *_count keys)
_TIMERS: dict[str, float] = {}


def reset() -> None:
    """Clear all timers/counters (called once at the start of a run)."""
    _TIMERS.clear()


def add(key: str, seconds: float) -> None:
    _TIMERS[key] = _TIMERS.get(key, 0.0) + float(seconds)


def add_count(key: str, n: int = 1) -> None:
    _TIMERS[key] = _TIMERS.get(key, 0) + int(n)


@contextlib.contextmanager
def timed(key: str):
    """Accumulate the wall-clock duration of the block into ``key``."""
    t0 = perf_counter()
    try:
        yield
    finally:
        add(key, perf_counter() - t0)


def get(key: str, default=0.0):
    return _TIMERS.get(key, default)


def snapshot() -> dict:
    """A plain-dict copy of the current timers (safe to serialise / reduce)."""
    return dict(_TIMERS)
