"""Convert a Qiskit QuantumCircuit to our internal circuit dict. """
from __future__ import annotations

SUPPORTED_BASIS = ["h", "x", "y", "z", "s", "t", "ry", "cx", "cz", "swap", "cy"]

_QISKIT_MAP = {
    "h": "H", "x": "X", "y": "Y", "z": "Z",
    "s": "S", "t": "T",
    "ry": "RY",
    "cx": "CNOT", "cnot": "CNOT",
    "swap": "SWAP",
    "cz": "CZ", "cy": "CY",
}

_SKIP = frozenset({"barrier", "measure", "reset", "delay", "id"})


def qiskit_to_dict(qc) -> dict:
    """Convert a Qiskit QuantumCircuit (already transpiled) to circuit dict."""
    n = qc.num_qubits
    gates = []
    for inst in qc.data:
        op = inst.operation
        name = op.name.lower()
        if name in _SKIP:
            continue
        qubits = [qc.find_bit(q).index for q in inst.qubits]
        if name not in _QISKIT_MAP:
            raise ValueError(
                f"Unsupported gate '{name}'. Transpile to basis {SUPPORTED_BASIS} first."
            )
        entry: dict = {"qubits": qubits, "gate": _QISKIT_MAP[name], "params": {}}
        if name == "ry":
            entry["params"]["theta"] = float(op.params[0])
        gates.append(entry)
    return {"number_of_qubits": n, "gates": gates}
