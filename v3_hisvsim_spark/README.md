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

## Architecture Diagram

```
+---------------------------------------------------------------------+
|                      QUANTUM CIRCUIT INPUT                          |
|                 {gates: [...], number_of_qubits: n}                 |
+---------------------------------------------------------------------+
                                  |
                                  v
+---------------------------------------------------------------------+
|                       CIRCUIT PARTITIONER                           |
|                                                                     |
|   +-------------+    +-------------+    +-------------+             |
|   | Build DAG   |--->| Find Levels |--->| Group Gates |             |
|   |(dependencies)|   |(topological)|    |(independent)|             |
|   +-------------+    +-------------+    +-------------+             |
|                                                                     |
|   Example: H(0), H(2), CNOT(0,1), CNOT(1,2)                        |
|   Output:  Level 0: [H(0), H(2)]                                   |
|            Level 1: [CNOT(0,1)]                                    |
|            Level 2: [CNOT(1,2)]                                    |
+---------------------------------------------------------------------+
                                  |
                                  v
+---------------------------------------------------------------------+
|                      SPARK DRIVER (Main Loop)                       |
|                                                                     |
|   FOR each level:                                                   |
|     1. Write WAL entry (PENDING)                                    |
|     2. Fuse independent single-qubit gates -> tensor product        |
|     3. Apply gates via Spark DataFrame transformations              |
|     4. Save new state version (Parquet)                             |
|     5. Checkpoint if needed                                         |
|     6. Mark WAL entry (COMMITTED)                                   |
+---------------------------------------------------------------------+
                                  |
                                  v
+---------------------------------------------------------------------+
|                    PARALLEL GATE APPLICATOR                         |
|                                                                     |
|   Single-Qubit Gate Fusion:                                         |
|   H(q0), H(q1), H(q2) -> H x H x H -> Single 8x8 transformation    |
|   Benefits: 1 Spark job instead of 3                                |
|                                                                     |
|   Gate Application (Spark DataFrame):                               |
|   +-------------+      +---------+      +-------------+             |
|   | State DF    |  x   | Gate    |  =   | New State   |             |
|   | idx | amp   |      | Matrix  |      | idx | amp   |             |
|   | 0   | 1.0   |      | [H]     |      | 0   | 0.707 |             |
|   +-------------+      +---------+      | 1   | 0.707 |             |
|                                         +-------------+             |
+---------------------------------------------------------------------+
                                  |
                                  v
+---------------------------------------------------------------------+
|                        STATE MANAGER                                |
|                                                                     |
|   +-------------+    +-------------+    +-------------+             |
|   | Spark DF    |--->|  Parquet    |--->|  Recovery   |             |
|   | (in memory) |    | (on disk)   |    | (on crash)  |             |
|   +-------------+    +-------------+    +-------------+             |
|                                                                     |
|   Storage: data/state/run_id=xxx/state_version=N/                  |
+---------------------------------------------------------------------+
                                  |
                                  v
+---------------------------------------------------------------------+
|                       SIMULATION RESULT                             |
|                                                                     |
|   {                                                                 |
|     final_state: {0: 0.707+0j, 31: 0.707+0j},                      |
|     gates_applied: 100,                                             |
|     parallel_groups: [5, 1, 1, ...],                               |
|     time_seconds: 45.2                                              |
|   }                                                                 |
+---------------------------------------------------------------------+
```

---

## Scalability Results

Tested maximum qubits for different circuit types:

| Circuit | Max Qubits | Non-Zero Amplitudes | Sparsity |
|---------|------------|---------------------|----------|
| GHZ | **1000** | 2 | O(1) |
| W State | **200** | 2.7M | O(n) |
| Hadamard Wall | **12** | 4,096 | O(2^n) |

**The number of non-zero amplitudes determines scalability, not the qubit count.**

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

### Fault Tolerance

- **Write-ahead log**: Track gate progress (PENDING -> COMMITTED)
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
```

---

## Supported Gates

**Single-qubit:** H, X, Y, Z, S, T, RY, R, G

**Two-qubit:** CNOT, CZ, CY, SWAP, CR, CU

---

## More Details

See [TECHNICAL.md](TECHNICAL.md) for deep technical details.

---

## References

- HiSVSIM: IEEE CLUSTER 2022
