"""
Translate Gate objects into SQL statements.
"""
from __future__ import annotations

from .gates import Gate


def sql_apply_one_qubit_gate(gate_name: str, q: int, v: int) -> str:
    v1 = v + 1
    return f"""
    INSERT INTO state(version, idx, real, imag)
    SELECT
        {v1} AS version,
        ((S.idx & ~(1 << {q})) | (U.row << {q})) AS idx,
        SUM(U.real * S.real - U.imag * S.imag) AS real,
        SUM(U.real * S.imag + U.imag * S.real) AS imag
    FROM state AS S
    JOIN gate_matrix AS U
      ON U.gate_name = '{gate_name}'
     AND U.arity = 1
     AND U.col = ((S.idx >> {q}) & 1)
    WHERE S.version = {v}
    GROUP BY ((S.idx & ~(1 << {q})) | (U.row << {q}));
    """


def sql_apply_two_qubit_gate(gate_name: str, q0: int, q1: int, v: int) -> str:
    v1 = v + 1
    return f"""
    INSERT INTO state(version, idx, real, imag)
    SELECT
        {v1} AS version,
        (
            (S.idx & ~((1 << {q0}) | (1 << {q1})))
            | (((U.row >> 0) & 1) << {q0})
            | (((U.row >> 1) & 1) << {q1})
        ) AS idx,
        SUM(U.real * S.real - U.imag * S.imag) AS real,
        SUM(U.real * S.imag + U.imag * S.real) AS imag
    FROM state AS S
    JOIN gate_matrix AS U
      ON U.gate_name = '{gate_name}'
     AND U.arity = 2
     AND U.col = (
         ((S.idx >> {q0}) & 1)
         | (((S.idx >> {q1}) & 1) << 1)
     )
    WHERE S.version = {v}
    GROUP BY (
        (S.idx & ~((1 << {q0}) | (1 << {q1})))
        | (((U.row >> 0) & 1) << {q0})
        | (((U.row >> 1) & 1) << {q1})
    );
    """


def translate_gate(gate: Gate, version: int) -> str:
    if gate.two_qubit_gate:
        q0, q1 = gate.qubits
        return sql_apply_two_qubit_gate(gate.gate_name, q0, q1, version)
    (q,) = gate.qubits
    return sql_apply_one_qubit_gate(gate.gate_name, q, version)

