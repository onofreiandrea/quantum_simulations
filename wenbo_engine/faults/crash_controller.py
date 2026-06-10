"""Crash controller — executes an injected fault.

Separated from the *decision* (:mod:`wenbo_engine.faults.fault_injector`) so
the crash mechanism is independently testable.  The decision layer asks "do we
fire here?"; this layer answers "how do we crash".

Modes
-----
``os_exit``
    Hard crash: ``os._exit(code)`` — the most faithful model of a node/process
    dying mid-commit.  It skips ``atexit`` handlers, buffer flushes, and stack
    unwinding, exactly like a kill -9 / power loss, so any partially written
    (un-fsynced-rename) state is what recovery must cope with.
``exception``
    Soft crash: raise :class:`InjectedFault`.  Useful in-process where
    ``os._exit`` would kill the test runner; the surrounding harness catches it
    and then runs recovery.  Semantically equivalent for the recovery proofs
    because the fault hooks fire at the SAME protocol position either way.

The exit function is indirected through ``CrashController._exit`` so a test can
monkeypatch it and observe the crash without terminating pytest.
"""
from __future__ import annotations

import logging
import os
from typing import Callable

log = logging.getLogger(__name__)

OS_EXIT = "os_exit"
EXCEPTION = "exception"
CRASH_MODES = (OS_EXIT, EXCEPTION)

DEFAULT_EXIT_CODE = 137  # 128 + SIGKILL(9): "killed", mirrors a hard crash


class InjectedFault(RuntimeError):
    """Raised by the ``exception`` crash mode (a deliberate, labelled fault)."""

    def __init__(self, fault_point, rank, stage_id, message: str = ""):
        self.fault_point = fault_point
        self.rank = rank
        self.stage_id = stage_id
        super().__init__(
            message
            or f"injected fault {fault_point} at rank={rank} stage_id={stage_id}")


class CrashController:
    """Executes the configured crash ``mode`` for a fired fault point."""

    def __init__(self, mode: str = OS_EXIT, *,
                 exit_code: int = DEFAULT_EXIT_CODE,
                 exit_fn: Callable[[int], None] | None = None):
        if mode not in CRASH_MODES:
            raise ValueError(
                f"unknown crash mode {mode!r} (valid: {', '.join(CRASH_MODES)})")
        self.mode = mode
        self.exit_code = exit_code
        # Indirection so tests can observe os_exit without dying.  Defaults to
        # the real hard-crash primitive.
        self._exit: Callable[[int], None] = exit_fn or os._exit

    def crash(self, fault_point, rank, stage_id) -> None:
        """Execute the crash.  ``os_exit`` does not return; ``exception`` raises."""
        log.warning(
            "[fault] firing %s mode=%s rank=%s stage_id=%s",
            fault_point, self.mode, rank, stage_id)
        if self.mode == OS_EXIT:
            # Hard crash: no flushing, no unwinding — like a killed node.
            self._exit(self.exit_code)
            # If a test monkeypatched _exit to a non-terminating stub, fall
            # through to raising so the caller still observes the crash and
            # never proceeds past the fault point.
            raise InjectedFault(fault_point, rank, stage_id,
                                f"os_exit stub returned (code={self.exit_code})")
        raise InjectedFault(fault_point, rank, stage_id)
