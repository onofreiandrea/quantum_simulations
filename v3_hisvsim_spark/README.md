# Distributed Quantum Circuit Simulator

A Spark-based quantum simulator with parallel gate execution.

## Quick Start

```python
from pathlib import Path
from driver import SparkHiSVSIMDriver
from v2_common.config import SimulatorConfig
from v2_common.circuits import generate_ghz_circuit

config = SimulatorConfig(
    base_path=Path("./data"),
    spark_master="local[*]",
)
config.ensure_paths()

with SparkHiSVSIMDriver(config) as driver:
    result = driver.run_circuit(generate_ghz_circuit(100), resume=False)
    state = driver.get_state_vector(result)
```

## Requirements

- Python 3.10+
- Java 11+ (for Spark)
- `pip install -r requirements.txt`

No Docker needed. PySpark runs locally with your system Java.

---

## Scalability Results

Tested maximum qubits for different circuit types:

| Circuit | Max Qubits | Non-Zero Amplitudes | Sparsity |
|---------|------------|---------------------|----------|
| GHZ | **1000** | 2 | O(1) |
| W State | **200** | 2.7M | O(n) |
| Hadamard Wall | **12** | 4,096 | O(2^n) |

**The number of non-zero amplitudes determines scalability, not the qubit count.**

- **Sparse circuits** (GHZ, W) scale to hundreds or thousands of qubits
- **Dense circuits** (Hadamard wall) are limited to ~12 qubits locally due to exponential memory growth

With a cluster, dense circuits can reach ~30-35 qubits before hitting memory limits.

---

## How It Works

### State Representation

Quantum states are stored as Spark DataFrames with sparse representation:

```
idx  | real      | imag
-----|-----------|----------
0    | 0.707107  | 0.0
31   | 0.707107  | 0.0
```

Only non-zero amplitudes are stored. A 1000-qubit GHZ state needs just 2 rows.

### Circuit Partitioning

Gates are organized into topological levels based on qubit dependencies:

```
Circuit: H(0), H(2), CNOT(0,1), CNOT(1,2)

Level 0: H(0), H(2)     ← independent, run in parallel
Level 1: CNOT(0,1)      ← depends on H(0)
Level 2: CNOT(1,2)      ← depends on both
```

### Parallel Gate Execution

Independent single-qubit gates within a level are fused into a single tensor product operation:

```
Hadamard wall (8 qubits):
  Before: 8 separate H gate applications
  After:  1 fused operation (H ⊗ H ⊗ H ⊗ H ⊗ H ⊗ H ⊗ H ⊗ H)

Result: parallel_groups = [8]  ← all 8 gates in 1 transformation
```

### Gate Application

For each gate:
1. Extract target qubit bit from state index (bitwise ops)
2. Join with gate matrix (broadcast to all workers)
3. Compute new amplitudes
4. Group by index and sum (handles interference)
5. Filter near-zero values to maintain sparsity

### Fault Tolerance

- **Write-ahead log**: Track gate progress (PENDING → COMMITTED)
- **Checkpoints**: Periodic state snapshots to Parquet
- **Recovery**: Resume from last checkpoint after crash

---

## Project Structure

```
src/
├── driver.py                    # Main orchestrator
├── parallel_gate_applicator.py  # Fuses independent gates
├── hisvsim/
│   └── partition_adapter.py     # Circuit partitioning (DAG, levels)
└── v2_common/
    ├── config.py                # Configuration
    ├── circuits.py              # Circuit generators (GHZ, W, QFT, etc.)
    ├── gate_applicator.py       # Applies gates via Spark
    └── state_manager.py         # State I/O (Parquet)

scripts/
├── test_max_qubits.py          # Scalability testing
├── sparsity_analysis.py        # Sparsity impact analysis
└── test_parallel_execution.py  # Parallel gate verification
```

---

## Supported Gates

**Single-qubit:** H, X, Y, Z, S, T, RY, R, G

**Two-qubit:** CNOT, CZ, CY, SWAP, CR, CU

---

## Why Sparsity Matters

```
GHZ-1000:     2 amplitudes    → 32 bytes    → runs in seconds
W-200:        2.7M amplitudes → 43 MB       → runs in minutes  
H-Wall-30:    1B amplitudes   → 16 GB       → needs a cluster
H-Wall-50:    1 quadrillion   → 18 PB       → impossible
```

Distribution helps with speed but doesn't remove the 2^n memory requirement for dense states. Sparse circuits sidestep this entirely.

---

## More Details

See [TECHNICAL.md](TECHNICAL.md) for:
- Algorithm pseudocode (DAG building, topological sorting)
- Spark execution flow
- Gate application internals
- Tensor product fusion
- Fault tolerance mechanisms
- Performance optimizations

---

## References

- HiSVSIM: "Efficient Hierarchical State Vector Simulation of Quantum Circuits via Acyclic Graph Partitioning", IEEE CLUSTER 2022
