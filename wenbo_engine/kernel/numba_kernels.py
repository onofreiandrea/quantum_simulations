"""Numba-JIT CPU kernel backend (optional, faster path).

The JIT primitives themselves live next to their numpy fallbacks in
``cpu_batched`` / ``cpu_nonlocal`` (so a single dispatcher picks the path per
the active backend).  This module is the explicit *availability + warm-up*
namespace for the numba backend: it never imports numba at module load and is
safe to import when numba is absent.
"""
from __future__ import annotations

from wenbo_engine.kernel import backend


def available() -> bool:
    """True iff numba can be imported (no compilation triggered)."""
    return backend.numba_available()


def warmup() -> dict:
    """Select + compile the numba backend, returning the backend state.

    Equivalent to ``backend.set_backend('numba')``: compiles the JIT kernels
    (timing the compile) and, if numba is missing or compilation fails, falls
    back to numpy with a recorded reason.  Safe to call when numba is absent.
    """
    return backend.set_backend("numba")


__all__ = ["available", "warmup"]
