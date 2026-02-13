"""Circuit dict validation, parsing, and levelization.

Endianness convention: LITTLE-ENDIAN.
  qubit 0 = bit 0 (LSB) of the state-vector index.
  |q_{n-1} ... q_1 q_0>  has index  q_0 + 2*q_1 + ... + 2^{n-1}*q_{n-1}.
"""
from __future__ import annotations

import re
from typing import Any

ENDIANNESS = "little"

GATES_1Q_NO_PARAMS = frozenset({"H", "X", "Y", "Z", "S", "T"})
GATES_1Q_PARAM_SPEC: dict[str, dict[str, type | str]] = {
    "RY": {"theta": float},
    "R":  {"k": int},
    "G":  {"p": int},
}
GATES_2Q_NO_PARAMS = frozenset({"CNOT", "SWAP", "CZ", "CY"})
GATES_2Q_PARAM_SPEC: dict[str, dict[str, type | str]] = {
    "CR": {"k": int},
    "CU": {"U": "array", "exponent": int},
}

ALL_1Q = GATES_1Q_NO_PARAMS | set(GATES_1Q_PARAM_SPEC)
ALL_2Q = GATES_2Q_NO_PARAMS | set(GATES_2Q_PARAM_SPEC)
ALL_GATES = ALL_1Q | ALL_2Q


# ── name-encoded parsing ────────────────────────────────────────────
def _parse_name_encoded(raw: str) -> tuple[str, dict]:
    """CR3 → ('CR', {'k':3}),  R3 → ('R', {'k':3}),  H → ('H', {})."""
    m = re.match(r"^CR(\d+)$", raw)
    if m:
        return "CR", {"k": int(m.group(1))}
    if raw not in ("RY",):
        m = re.match(r"^R(\d+)$", raw)
        if m:
            return "R", {"k": int(m.group(1))}
    return raw, {}


# ── validation ──────────────────────────────────────────────────────
def validate_circuit_dict(d: dict[str, Any]) -> dict:
    """Validate and normalise a circuit dict.  Raises ValueError on bad input."""
    if not isinstance(d, dict):
        raise ValueError("circuit must be a dict")
    missing = {"number_of_qubits", "gates"} - set(d)
    if missing:
        raise ValueError(f"missing required keys: {missing}")
    extra = set(d) - {"number_of_qubits", "gates"}
    if extra:
        raise ValueError(f"unknown top-level keys: {extra}")

    n = d["number_of_qubits"]
    if not isinstance(n, int) or n < 1:
        raise ValueError(f"number_of_qubits must be positive int, got {n!r}")

    if not isinstance(d["gates"], list):
        raise ValueError("gates must be a list")

    return {
        "number_of_qubits": n,
        "gates": [_validate_gate(g, n, i) for i, g in enumerate(d["gates"])],
    }


def _validate_gate(g: dict, nq: int, idx: int) -> dict:
    tag = f"gate[{idx}]"
    if not isinstance(g, dict):
        raise ValueError(f"{tag}: must be a dict")
    if not {"qubits", "gate"} <= set(g):
        raise ValueError(f"{tag}: missing 'qubits' or 'gate'")
    if set(g) - {"qubits", "gate", "params"}:
        raise ValueError(f"{tag}: unknown keys {set(g) - {'qubits', 'gate', 'params'}}")

    qubits = g["qubits"]
    if not isinstance(qubits, list) or not all(isinstance(q, int) for q in qubits):
        raise ValueError(f"{tag}: qubits must be list[int]")
    for q in qubits:
        if q < 0 or q >= nq:
            raise ValueError(f"{tag}: qubit {q} out of range [0, {nq})")

    base, name_params = _parse_name_encoded(g["gate"])
    if base not in ALL_GATES:
        raise ValueError(f"{tag}: unsupported gate '{g['gate']}'")

    expected_arity = 1 if base in ALL_1Q else 2
    if len(qubits) != expected_arity:
        raise ValueError(f"{tag}: {base} needs {expected_arity} qubit(s), got {len(qubits)}")

    merged = {**name_params, **(g.get("params") or {})}

    spec = GATES_1Q_PARAM_SPEC.get(base) or GATES_2Q_PARAM_SPEC.get(base) or {}
    for key, expected in spec.items():
        if key not in merged:
            raise ValueError(f"{tag}: {base} requires param '{key}'")
        if expected != "array" and not isinstance(merged[key], (expected, int)):
            raise ValueError(f"{tag}: param '{key}' bad type")

    return {"qubits": list(qubits), "gate": base, "params": merged}


# ── levelization ────────────────────────────────────────────────────
def levelize(circuit_dict: dict) -> list[list[dict]]:
    """Group gates into dependency-free levels (same qubits ⇒ different levels)."""
    levels: list[list[dict]] = []
    qubit_free: dict[int, int] = {}
    for g in circuit_dict["gates"]:
        t = max((qubit_free.get(q, 0) for q in g["qubits"]), default=0)
        while len(levels) <= t:
            levels.append([])
        levels[t].append(g)
        for q in g["qubits"]:
            qubit_free[q] = t + 1
    return levels
