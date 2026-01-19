# Why GHZ State Has Only 2 Non-Zero Amplitudes

## GHZ State Definition

The **Greenberger-Horne-Zeilinger (GHZ) state** for `n` qubits is:

```
|GHZ_n⟩ = (|00...0⟩ + |11...1⟩) / √2
```

This means:
- **Only TWO basis states** have non-zero amplitude
- `|00...0⟩` (all qubits in |0⟩) → amplitude = `1/√2`
- `|11...1⟩` (all qubits in |1⟩) → amplitude = `1/√2`
- **All other 2^n - 2 states** → amplitude = `0`

## Sparsity Examples

| Qubits | Total States | Non-Zero | Zero | Sparsity |
|--------|-------------|----------|------|----------|
| 2 | 4 | 2 | 2 | 50% |
| 3 | 8 | 2 | 6 | 75% |
| 4 | 16 | 2 | 14 | 87.5% |
| 5 | 32 | 2 | 30 | 93.75% |
| 10 | 1,024 | 2 | 1,022 | 99.8% |
| 20 | 1,048,576 | 2 | 1,048,574 | 99.9998% |

**As n increases, GHZ becomes exponentially sparser!**

## How GHZ Circuit Works

### Circuit Structure
```
H[0] → CNOT[0,1] → CNOT[1,2] → ... → CNOT[n-2, n-1]
```

### Step-by-Step Evolution

**Initial State:**
```
|00...0⟩  (amplitude = 1)
```

**After H[0] (Hadamard on qubit 0):**
```
(|00...0⟩ + |10...0⟩) / √2
```
- Qubit 0 is now in superposition: `(|0⟩ + |1⟩) / √2`
- All other qubits remain |0⟩

**After CNOT[0,1] (Copy qubit 0 to qubit 1):**
```
(|00...0⟩ + |11...0⟩) / √2
```
- If qubit 0 is |0⟩ → qubit 1 stays |0⟩
- If qubit 0 is |1⟩ → qubit 1 flips to |1⟩
- Result: Qubits 0 and 1 are now correlated

**After CNOT[1,2] (Copy qubit 1 to qubit 2):**
```
(|00...0⟩ + |11...1⟩) / √2
```
- Qubit 2 copies qubit 1's state
- All three qubits are now correlated

**After all CNOTs:**
```
(|00...0⟩ + |11...1⟩) / √2  = |GHZ_n⟩
```

## Why Only 2 States?

The key insight is that **CNOT gates COPY the first qubit to all others**.

Since the first qubit is in superposition `(|0⟩ + |1⟩) / √2`, and each CNOT copies this state, we get:

- **Path 1**: First qubit = |0⟩ → All qubits = |0⟩ → `|00...0⟩`
- **Path 2**: First qubit = |1⟩ → All qubits = |1⟩ → `|11...1⟩`

**No other combinations are possible** because:
- If qubit 0 is |0⟩, all CNOTs keep other qubits at |0⟩
- If qubit 0 is |1⟩, all CNOTs flip other qubits to |1⟩
- Mixed states like |01...0⟩ or |10...1⟩ **cannot occur**

## Implications for Distribution

### Why GHZ Can't Benefit from Distribution

```
GHZ(20 qubits):
  Total states: 2^20 = 1,048,576
  Non-zero: 2
  Partitions needed: 16
  
Result: Most partitions are EMPTY!
```

**Problem:**
- With 16 partitions, we'd have:
  - 1 partition with 2 rows (both amplitudes)
  - 15 partitions with 0 rows (empty)
- **No parallelism possible** - only 1 partition has work!

### Comparison with Dense States

**QFT (Quantum Fourier Transform):**
```
QFT(10 qubits):
  Total states: 2^10 = 1,024
  Non-zero: ~1,024 (dense)
  Partitions: 16
  
Result: ~64 rows per partition - GOOD for distribution!
```

**Random Circuit:**
```
Random(10 qubits):
  Total states: 2^10 = 1,024
  Non-zero: ~1,024 (dense)
  Partitions: 16
  
Result: ~64 rows per partition - GOOD for distribution!
```

## Why This Matters

1. **Sparse states** (GHZ, W-state) → **Can't distribute effectively**
   - Only a few non-zero amplitudes
   - Most partitions empty
   - Sequential execution is fine

2. **Dense states** (QFT, random circuits) → **CAN distribute effectively**
   - Many non-zero amplitudes
   - Work spread across partitions
   - Parallel execution beneficial

3. **This is why HiSVSIM-style circuit partitioning helps:**
   - Partition the **circuit** (not just state)
   - Run independent sub-circuits in parallel
   - Even sparse states can benefit from circuit-level parallelism

## Summary

**GHZ has only 2 rows because:**
1. It's defined as `(|00...0⟩ + |11...1⟩) / √2`
2. CNOT gates create perfect correlation (all |0⟩ or all |1⟩)
3. No mixed states are possible
4. This makes it **maximally sparse** but **maximally entangled**

This is a fundamental property of the GHZ state, not a limitation of our implementation!
