"""Unit tests for the deterministic fault-injection package.

Covers the decision layer (FaultInjector: fires ONLY on an exact
rank+stage+point match, no-op otherwise) and the crash layer
(CrashController: os_exit vs exception modes, observable via a stubbed exit).
"""
import json

import pytest

from wenbo_engine.faults import (
    FaultPoint, ALL_FAULT_POINTS, GLOBAL_COMMIT_BOUNDARY, parse_fault_point,
    FaultInjector, CrashController, InjectedFault, OS_EXIT, EXCEPTION,
    NULL_INJECTOR, CONFIG_KEY,
)


# ── fault points enum ───────────────────────────────────────────────────

def test_all_required_fault_points_defined():
    required = {
        "BEFORE_STAGE", "AFTER_READ", "AFTER_PARTIAL_WRITE", "AFTER_ALL_WRITES",
        "AFTER_RENAME", "AFTER_MANIFEST_WRITE", "AFTER_MANIFEST_RENAME",
        "AFTER_ALLGATHER_PREPARED", "BEFORE_GLOBAL_COMMIT",
        "AFTER_GLOBAL_COMMIT", "DURING_DURABLE_UPLOAD", "BEFORE_DURABLE_COMMIT",
    }
    assert {fp.value for fp in FaultPoint} == required
    assert {fp.value for fp in ALL_FAULT_POINTS} == required


def test_fault_point_str_is_bare_name():
    assert str(FaultPoint.AFTER_GLOBAL_COMMIT) == "AFTER_GLOBAL_COMMIT"
    assert f"{FaultPoint.AFTER_READ}" == "AFTER_READ"


def test_parse_fault_point_roundtrip_and_error():
    assert parse_fault_point("AFTER_RENAME") is FaultPoint.AFTER_RENAME
    assert parse_fault_point(FaultPoint.BEFORE_STAGE) is FaultPoint.BEFORE_STAGE
    with pytest.raises(ValueError):
        parse_fault_point("NOPE")


def test_global_commit_boundary():
    assert GLOBAL_COMMIT_BOUNDARY is FaultPoint.AFTER_GLOBAL_COMMIT


# ── config + decision (will_fire) ───────────────────────────────────────

def _cfg(**kw):
    base = {"enabled": True, "fault_point": "AFTER_MANIFEST_RENAME",
            "rank": 2, "stage_id": 4, "mode": "os_exit"}
    base.update(kw)
    return {CONFIG_KEY: base}


def test_disabled_injector_never_fires():
    inj = FaultInjector.disabled()
    assert not inj.enabled
    for fp in FaultPoint:
        assert not inj.will_fire(fp, rank=0, stage_id=0)


def test_null_injector_is_noop():
    assert not NULL_INJECTOR.enabled
    NULL_INJECTOR.maybe_fire(FaultPoint.AFTER_GLOBAL_COMMIT, 0, 0)  # no raise


def test_fires_only_on_exact_match():
    inj = FaultInjector(_cfg())
    # exact match
    assert inj.will_fire(FaultPoint.AFTER_MANIFEST_RENAME, 2, 4)
    # wrong point
    assert not inj.will_fire(FaultPoint.AFTER_GLOBAL_COMMIT, 2, 4)
    # wrong rank
    assert not inj.will_fire(FaultPoint.AFTER_MANIFEST_RENAME, 1, 4)
    # wrong stage
    assert not inj.will_fire(FaultPoint.AFTER_MANIFEST_RENAME, 2, 5)


def test_wildcard_rank_and_stage():
    inj = FaultInjector(_cfg(rank=None, stage_id=None))
    assert inj.will_fire(FaultPoint.AFTER_MANIFEST_RENAME, 0, 0)
    assert inj.will_fire(FaultPoint.AFTER_MANIFEST_RENAME, 9, 9)
    assert not inj.will_fire(FaultPoint.AFTER_READ, 0, 0)


def test_enabled_without_point_raises():
    with pytest.raises(ValueError):
        FaultInjector({CONFIG_KEY: {"enabled": True}})


def test_empty_config_is_disabled():
    assert not FaultInjector(None).enabled
    assert not FaultInjector({}).enabled


# ── crash controller ────────────────────────────────────────────────────

def test_os_exit_mode_calls_exit_fn():
    seen = {}
    cc = CrashController(OS_EXIT, exit_code=137,
                         exit_fn=lambda c: seen.setdefault("code", c))
    # stub exit_fn returns → controller raises so caller never proceeds
    with pytest.raises(InjectedFault):
        cc.crash(FaultPoint.AFTER_GLOBAL_COMMIT, 2, 4)
    assert seen["code"] == 137


def test_exception_mode_raises_injected_fault():
    cc = CrashController(EXCEPTION)
    with pytest.raises(InjectedFault) as ei:
        cc.crash(FaultPoint.BEFORE_GLOBAL_COMMIT, 1, 3)
    assert ei.value.fault_point == FaultPoint.BEFORE_GLOBAL_COMMIT
    assert ei.value.rank == 1
    assert ei.value.stage_id == 3


def test_unknown_mode_rejected():
    with pytest.raises(ValueError):
        CrashController("meltdown")


# ── maybe_fire executes the crash + records the event ───────────────────

def test_maybe_fire_executes_and_records(tmp_path):
    from wenbo_engine.recovery.recovery_events import RecoveryEventLog, EventType
    events = RecoveryEventLog()
    fired = {}
    cc = CrashController(OS_EXIT, exit_fn=lambda c: fired.setdefault("c", c))
    inj = FaultInjector(_cfg(), controller=cc, events=events)
    with pytest.raises(InjectedFault):
        inj.maybe_fire(FaultPoint.AFTER_MANIFEST_RENAME, 2, 4)
    assert "c" in fired                 # the exit fn was called (crash executed)
    assert inj.fired
    # proof #4 substrate: a structured FAULT_INJECTED event was recorded.
    assert events.has(EventType.FAULT_INJECTED)
    ev = events.of_type(EventType.FAULT_INJECTED)[0]
    assert ev.details["fault_point"] == "AFTER_MANIFEST_RENAME"
    assert ev.details["stage_id"] == 4


def test_maybe_fire_is_noop_when_unmatched():
    fired = {}
    cc = CrashController(OS_EXIT, exit_fn=lambda c: fired.setdefault("c", c))
    inj = FaultInjector(_cfg(), controller=cc)
    inj.maybe_fire(FaultPoint.AFTER_READ, 2, 4)   # wrong point
    inj.maybe_fire(FaultPoint.AFTER_MANIFEST_RENAME, 0, 4)  # wrong rank
    assert not inj.fired
    assert "c" not in fired


# ── from_env plumbing ───────────────────────────────────────────────────

def test_from_env_disabled_when_unset():
    assert not FaultInjector.from_env({}).enabled


def test_from_env_builds_config():
    inj = FaultInjector.from_env({
        "WE_FAULT_POINT": "AFTER_GLOBAL_COMMIT",
        "WE_FAULT_RANK": "1", "WE_FAULT_STAGE": "2", "WE_FAULT_MODE": "exception",
    })
    assert inj.enabled
    assert inj.will_fire(FaultPoint.AFTER_GLOBAL_COMMIT, 1, 2)
    assert inj.mode == "exception"


def test_from_json_file(tmp_path):
    p = tmp_path / "fi.json"
    p.write_text(json.dumps(_cfg(mode="exception")))
    inj = FaultInjector.from_json_file(p)
    assert inj.enabled and inj.mode == "exception"
    assert inj.will_fire(FaultPoint.AFTER_MANIFEST_RENAME, 2, 4)
