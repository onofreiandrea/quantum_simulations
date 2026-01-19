# Test Coverage

What we test and why.

## What's Covered

1. All gate types (15 different gates)
2. Sparse states (few non-zero amplitudes)
3. Dense states (all amplitudes non-zero)
4. Complex circuit combinations
5. Non-stabilizer circuits (T, RY, R, G, CU gates)

## Test Suite: `test_all_gates_and_states.py`

### All Gate Types

Tests every gate type individually to make sure they preserve normalization.

**Single-qubit gates (6):**
- H (Hadamard)
- X, Y, Z (Pauli gates)
- S (Phase gate)
- T (T gate - non-stabilizer)

**Two-qubit gates (5):**
- CNOT (Controlled-NOT)
- CZ, CY (Controlled-Z/Y)
- SWAP
- CR (Controlled-Rotation)

**Parameterized gates (4):**
- RY (Rotation-Y)
- R (Rotation)
- G (G gate)
- CU (Controlled-Unitary)

**Total: 15 gate types**

### Sparse States

Tests circuits that produce sparse states (few non-zero amplitudes):

- GHZ(3, 4, 5): Exactly 2 amplitudes (|00...0⟩ and |11...1⟩)
- W-state(3, 4): Exactly n amplitudes
- Bell state: Exactly 2 amplitudes (|00⟩ and |11⟩)

Note: GHZ states naturally have only 2 amplitudes - that's how they work, not a bug.

### Intermediate Sparsity

Tests states with medium sparsity (more than 2, less than 2^n):

- 4 amplitudes: 2 Hadamards on 4-qubit system
- 8 amplitudes: 3 Hadamards on 4-qubit system
- 16 amplitudes: 4 Hadamards on 5-qubit system
- 32 amplitudes: 5 Hadamards on 7-qubit system
- 64 amplitudes: 6 Hadamards on 8-qubit system
- Partial superpositions: Various sparsity levels
- Cluster states: Many amplitudes (medium sparsity)
- Ring states: Many amplitudes (medium sparsity)

### Dense States

Tests circuits that produce dense states (all amplitudes non-zero):

- Hadamard Wall(3, 4, 5): All 2^n amplitudes non-zero
- QFT(3, 4, 5): All 2^n amplitudes non-zero

### Complex Circuits

Tests complex combinations:

- Mixed gate types: H, T, CNOT, RY, CZ, CR, S in one circuit
- All gates together: Every gate type in one circuit
- GHZ+QFT: Sparse → Dense transition
- W+QFT: Sparse → Dense transition
- QPE: Quantum Phase Estimation (uses CU gates)

### Non-Stabilizer Circuits

Tests non-Clifford gates:

- T gate circuits: T on |1⟩, T on superposition
- RY gate circuits: Multiple RY gates with CNOT
- CR gate circuits: Controlled rotations (used in QFT)
- CU gate circuits: Controlled unitaries (used in QPE)

## Test Statistics

**Gate coverage:**
- Single-qubit: 6 gates
- Two-qubit: 5 gates
- Parameterized: 4 gates
- Total: 15 gate types

**State coverage:**
- Sparse: GHZ, W-state, Bell (2, n, 2 amplitudes)
- Intermediate: 4, 8, 16, 32, 64 amplitudes
- Dense: Hadamard Wall, QFT (2^n amplitudes)
- Mixed: GHZ+QFT, W+QFT, QPE (variable)

## What This Means

All gate types work correctly. We test:
- Sparse states (2 amplitudes)
- Intermediate sparsity (4, 8, 16, 32, 64 amplitudes)
- Dense states (2^n amplitudes)
- Full sparsity spectrum (2 → 2^n)
- Complex circuit combinations
- Non-stabilizer circuits

## Summary

We test:
- All gate types (15 different gates)
- Sparse states (2, n, 2 amplitudes)
- Intermediate sparsity (4, 8, 16, 32, 64 amplitudes)
- Dense states (2^n amplitudes)
- Full sparsity spectrum (2 → 2^n)
- Complex combinations
- Non-stabilizer circuits

This covers the full range from sparse to dense states and all gate types.
