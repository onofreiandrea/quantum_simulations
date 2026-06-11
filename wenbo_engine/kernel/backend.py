"""Selectable numerical backend for the CPU kernels.

The local/nonlocal gate kernels (``cpu_batched`` / ``cpu_nonlocal``) ship both a
numba-JIT path and a pure-numpy path.  Historically the numba path was used
implicitly whenever numba happened to be importable.  This module makes the
choice **explicit, measurable, and safe**:

  * ``numpy`` — the trusted baseline; never uses numba.
  * ``numba`` — use the JIT path, but only if numba imports AND compiles;
    otherwise fall back to numpy and record why.
  * ``auto``  — numba when available (preserving the previous behaviour),
    numpy otherwise (always safe).

Nothing here changes circuit semantics or precision: both paths operate
in-place on the same ``complex64`` chunk arrays.  The selected backend is a
per-process choice (one run per process under MPI), set by the runner at the
start of a run via :func:`set_backend`.
"""
from __future__ import annotations

import time

BACKENDS = ("numpy", "numba", "auto")

_STATE = {
    "requested": "auto",
    "used": "numpy",          # actually-active backend (numpy until selected)
    "available": None,        # is numba importable? (lazy)
    "compile_time": None,     # numba warm-up/compile seconds (if used)
    "fallback_reason": None,  # why numba was not used, when requested
}

_AVAILABLE_CACHE: bool | None = None


def numba_available() -> bool:
    """Whether numba can be imported (cached).  Does not compile anything."""
    global _AVAILABLE_CACHE
    if _AVAILABLE_CACHE is None:
        try:
            import numba  # noqa: F401
            _AVAILABLE_CACHE = True
        except Exception:
            _AVAILABLE_CACHE = False
    return _AVAILABLE_CACHE


def use_numba() -> bool:
    """True iff the active backend is numba (read by the kernel dispatchers)."""
    return _STATE["used"] == "numba"


def backend_info() -> dict:
    """A copy of the current backend state (for final_summary reporting)."""
    return dict(_STATE)


def _warmup() -> tuple[bool, str | None]:
    """Compile the numba kernels by exercising them on tiny arrays.

    Assumes ``_STATE['used'] == 'numba'`` so the dispatchers take the numba
    path.  Returns ``(ok, error)``; on failure the caller reverts to numpy.
    """
    try:
        import numpy as np
        from wenbo_engine.storage.block_store import DTYPE
        from wenbo_engine.kernel import cpu_batched, cpu_nonlocal
        H = (np.array([[1, 1], [1, -1]], dtype=DTYPE) / np.sqrt(2))
        U2 = np.eye(4, dtype=DTYPE)
        c = np.zeros(4, dtype=DTYPE); c[0] = 1.0
        cpu_batched.apply_1q(c, 0, H)
        cpu_batched.apply_2q(c, 0, 1, U2)
        a = np.zeros(2, dtype=DTYPE); b = np.zeros(2, dtype=DTYPE)
        cpu_nonlocal.apply_1q_pair(a, b, H)
        return True, None
    except Exception as e:  # pragma: no cover - defensive (numba toolchain)
        return False, repr(e)


def set_backend(mode: str | None) -> dict:
    """Select the numerical backend for this process; return the state dict.

    Safe by construction: an unavailable or non-compiling numba request falls
    back to numpy with a recorded ``fallback_reason``.  Idempotent enough to
    call once per run.
    """
    mode = (mode or "auto").lower()
    if mode not in BACKENDS:
        raise ValueError(f"kernel backend must be one of {BACKENDS}, got {mode!r}")
    avail = numba_available()
    requested = mode
    fallback_reason = None

    if mode == "numpy":
        used = "numpy"
    elif mode == "numba":
        used = "numba" if avail else "numpy"
        if not avail:
            fallback_reason = "numba not installed"
    else:  # auto
        used = "numba" if avail else "numpy"
        if not avail:
            fallback_reason = "numba unavailable; using numpy"

    _STATE.update(requested=requested, used=used, available=avail,
                  compile_time=None, fallback_reason=fallback_reason)

    if used == "numba":
        t0 = time.perf_counter()
        ok, err = _warmup()
        if ok:
            _STATE["compile_time"] = round(time.perf_counter() - t0, 6)
        else:
            _STATE.update(used="numpy",
                          fallback_reason=f"numba compile failed: {err}")
    return backend_info()
