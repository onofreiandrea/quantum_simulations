# v3: HiSVSIM + Spark Integration

This is v3 of the quantum simulator, integrating **HiSVSIM's circuit partitioning** with **Apache Spark** for distributed execution.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Circuit Input (QASM or Circuit Dict)                        │
└─────────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────────┐
│  HiSVSIM Circuit Partitioning                                │
│  - Build DAG from circuit                                    │
│  - Partition into acyclic sub-circuits                      │
│  - Output: List of independent partitions                   │
└─────────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────────┐
│  Spark Parallel Execution                                    │
│                                                              │
│  Partition 1 → Worker 1 → State 1                          │
│  Partition 2 → Worker 2 → State 2                          │
│  Partition 3 → Worker 3 → State 3                          │
│  ...                                                         │
└─────────────────────────────────────────────────────────────┘
                    ↓
┌─────────────────────────────────────────────────────────────┐
│  State Merging                                               │
│  - Combine states from partitions                            │
│  - Handle qubit overlaps                                     │
│  - Final state vector                                        │
└─────────────────────────────────────────────────────────────┘
```

## Key Features

- **Circuit-Level Parallelism**: Partition circuit into independent sub-circuits
- **Acyclic Partitioning**: HiSVSIM's proven partitioning strategy
- **Spark Distribution**: Leverage Spark's distributed computing
- **State Merging**: Properly combine results from parallel partitions

## Components

### 1. HiSVSIM Integration (`src/hisvsim/`)
- Circuit graph building (from HiSVSIM repo)
- Acyclic partitioning algorithms
- Partition validation

### 2. Spark Execution (`src/spark_executor/`)
- Parallel sub-circuit simulation
- State vector management
- Result aggregation

### 3. State Merging (`src/state_merger/`)
- Combine partition results
- Handle qubit overlaps
- Final state construction

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Build HiSVSIM C++ wrapper (if needed)
cd hisvsim_repo
# Follow HiSVSIM build instructions

# Run tests
pytest tests/ -v
```

## Usage

```python
from src.driver import SparkHiSVSIMDriver
from src.config import SimulatorConfig

config = SimulatorConfig(
    run_id="test",
    base_path="./data",
    spark_master="spark://master:7077",
    num_partitions=4
)

driver = SparkHiSVSIMDriver(config)
result = driver.run_circuit(circuit_dict)
```

## References

- HiSVSIM: https://github.com/pnnl/hisvsim.git
- Paper: "Efficient Hierarchical State Vector Simulation of Quantum Circuits via Acyclic Graph Partitioning", IEEE CLUSTER 2022
- arXiv: https://arxiv.org/pdf/2205.06973.pdf
