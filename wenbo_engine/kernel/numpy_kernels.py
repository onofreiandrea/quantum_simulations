"""Pure-numpy CPU kernels (the trusted baseline backend).

These are the vectorised numpy reference implementations.  ``cpu_batched`` /
``cpu_nonlocal`` use these whenever the active backend is ``numpy`` (see
:mod:`wenbo_engine.kernel.backend`).  Exposed here as an explicit namespace so
tests and callers can reference the baseline directly.
"""
from __future__ import annotations

from wenbo_engine.kernel.cpu_scalar import apply_1q, apply_2q, check_local

__all__ = ["apply_1q", "apply_2q", "check_local"]
