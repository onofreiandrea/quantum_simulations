# Circuit Dict Contract

## Schema

```python
{
  "number_of_qubits": int,   # ≥ 1
  "gates": [
    {
      "qubits": [int, ...],  # 1 or 2 elements, 0-indexed
      "gate":   str,          # gate name (see below)
      "params": { ... }       # optional, gate-specific
    },
    ...
  ]
}
```

## Endianness

**LITTLE-ENDIAN**: qubit 0 = bit 0 (LSB) of the state-vector index.

## Supported gates

| Name | Arity | Params | Notes |
|------|-------|--------|-------|
| H    | 1     | —      | Hadamard |
| X    | 1     | —      | Pauli-X |
| Y    | 1     | —      | Pauli-Y |
| Z    | 1     | —      | Pauli-Z |
| S    | 1     | —      | Phase (π/2) |
| T    | 1     | —      | Phase (π/4) |
| RY   | 1     | theta: float | Y-rotation |
| R    | 1     | k: int | Phase 2π/2^k. Name-encoded: "R3" → R with k=3 |
| G    | 1     | p: int | Custom rotation |
| CNOT | 2     | —      | qubits[0]=control, qubits[1]=target |
| SWAP | 2     | —      | |
| CZ   | 2     | —      | |
| CY   | 2     | —      | |
| CR   | 2     | k: int | Controlled-R. Name-encoded: "CR3" → CR with k=3 |
| CU   | 2     | U: 2×2 array, exponent: int | Controlled-U^exp |

## Name encoding

`CR3` is equivalent to `{"gate": "CR", "params": {"k": 3}}`.
`R3` is equivalent to `{"gate": "R", "params": {"k": 3}}`.
Explicit params override name-encoded values.
