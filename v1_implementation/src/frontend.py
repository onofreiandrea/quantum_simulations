"""
Circuit front-end that converts dict circuits into Gate objects.
"""
from __future__ import annotations

from .gates import (
    Gate,
    HadamardGate,
    XGate,
    YGate,
    ZGate,
    SGate,
    TGate,
    RYGate,
    RGate,
    GGate,
    CNOTGate,
    SWAPGate,
    CZGate,
    CYGate,
    CRGate,
    CUGate,
)


GATE_DISPATCH = {
    "H": lambda qubits, params: HadamardGate(qubits[0]),
    "X": lambda qubits, params: XGate(qubits[0]),
    "Y": lambda qubits, params: YGate(qubits[0]),
    "Z": lambda qubits, params: ZGate(qubits[0]),
    "S": lambda qubits, params: SGate(qubits[0]),
    "T": lambda qubits, params: TGate(qubits[0]),
    "RY": lambda qubits, params: RYGate(qubits[0], params["theta"]),
    "R": lambda qubits, params: RGate(qubits[0], params["k"]),
    "G": lambda qubits, params: GGate(qubits[0], params["p"]),
    "CNOT": lambda qubits, params: CNOTGate(qubits[0], qubits[1]),
    "SWAP": lambda qubits, params: SWAPGate(qubits[0], qubits[1]),
    "CZ": lambda qubits, params: CZGate(qubits[0], qubits[1]),
    "CY": lambda qubits, params: CYGate(qubits[0], qubits[1]),
    "CR": lambda qubits, params: CRGate(qubits[0], qubits[1], params["k"]),
    "CU": lambda qubits, params: CUGate(
        qubits[0], qubits[1], params["U"], params["exponent"], params.get("name")
    ),
}


def circuit_dict_to_gates(circuit_dict):
    n_qubits = circuit_dict["number_of_qubits"]
    gates: list[Gate] = []
    for gate_info in circuit_dict["gates"]:
        name = gate_info["gate"]
        qubits = gate_info["qubits"]
        params = gate_info.get("params", {})
        if name not in GATE_DISPATCH:
            raise ValueError(f"Unsupported gate {name}")
        gates.append(GATE_DISPATCH[name](qubits, params))
    return n_qubits, gates

