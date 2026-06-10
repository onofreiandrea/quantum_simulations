"""Deterministic plan (de)serialization to / from JSON.

A plan serializes to a plain dict with sorted keys and a fixed field
order, so ``serialize -> deserialize -> serialize`` is byte-identical and
the same input always yields the same plan bytes.  Matrices inside fused
ops are stored as nested ``[real, imag]`` lists (see
:mod:`.stage_builder`), keeping the JSON free of numpy objects.
"""
from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from wenbo_engine.planner.optimizer_v2 import Plan, Stage

_SCHEMA_VERSION = 1


def _op_to_dict(op) -> dict:
    return {
        "qubits": list(op.qubits),
        "gate": op.gate,
        "params": _normalize_params(op.params),
        "klass": op.klass,
    }


def _normalize_params(params: dict) -> dict:
    """Return params with deterministic key order and JSON-safe values."""
    out: dict = {}
    for key in sorted(params):
        out[key] = params[key]
    return out


def _stage_to_dict(stage) -> dict:
    return {
        "index": stage.index,
        "kind": stage.kind,
        "ops": [_op_to_dict(op) for op in stage.ops],
        "local_ops": stage.local_ops,
        "rank_nonlocal_ops": stage.rank_nonlocal_ops,
        "mpi_nonlocal_ops": stage.mpi_nonlocal_ops,
        "bytes_read": stage.bytes_read,
        "bytes_written": stage.bytes_written,
        "mpi_bytes_sent": stage.mpi_bytes_sent,
        "sendrecv_count": stage.sendrecv_count,
        "commits": stage.commits,
        "full_state_pass": stage.full_state_pass,
        "cost": _normalize_params(stage.cost),
    }


def serialize_plan(plan) -> dict:
    """Return a deterministic, JSON-serializable dict for ``plan``."""
    return {
        "schema_version": _SCHEMA_VERSION,
        "mode": plan.mode,
        "hardware": _normalize_params(plan.hardware.to_dict()),
        "perm": (None if plan.perm is None
                 else {str(kk): vv for kk, vv in sorted(plan.perm.items())}),
        "log_to_phys": (None if plan.log_to_phys is None
                        else list(plan.log_to_phys)),
        "stages": [_stage_to_dict(s) for s in plan.stages],
        "metrics": _normalize_params(plan.metrics),
    }


def plan_to_json(plan, *, indent: int | None = 2) -> str:
    """Serialize a plan to a deterministic JSON string."""
    return json.dumps(serialize_plan(plan), sort_keys=True, indent=indent)


def deserialize_plan(data: dict):
    """Rebuild a :class:`~.optimizer_v2.Plan` from a serialized dict."""
    from wenbo_engine.planner.optimizer_v2 import Plan, Stage, HardwareConfig
    from wenbo_engine.planner.stage_builder import PlannedOp

    hw = HardwareConfig(**data["hardware"])
    perm_raw = data.get("perm")
    perm = (None if perm_raw is None
            else {int(kk): int(vv) for kk, vv in perm_raw.items()})
    ltp_raw = data.get("log_to_phys")
    log_to_phys = None if ltp_raw is None else [int(v) for v in ltp_raw]

    stages: list[Stage] = []
    for sd in data["stages"]:
        ops = [
            PlannedOp(
                qubits=list(od["qubits"]),
                gate=od["gate"],
                params=dict(od.get("params", {})),
                klass=od["klass"],
            )
            for od in sd["ops"]
        ]
        stages.append(Stage(
            index=sd["index"],
            kind=sd["kind"],
            ops=ops,
            local_ops=sd["local_ops"],
            rank_nonlocal_ops=sd["rank_nonlocal_ops"],
            mpi_nonlocal_ops=sd["mpi_nonlocal_ops"],
            bytes_read=sd["bytes_read"],
            bytes_written=sd["bytes_written"],
            mpi_bytes_sent=sd["mpi_bytes_sent"],
            sendrecv_count=sd["sendrecv_count"],
            commits=sd["commits"],
            full_state_pass=sd["full_state_pass"],
            cost=dict(sd.get("cost", {})),
        ))

    return Plan(
        mode=data["mode"],
        hardware=hw,
        perm=perm,
        stages=stages,
        metrics=dict(data.get("metrics", {})),
        log_to_phys=log_to_phys,
    )
