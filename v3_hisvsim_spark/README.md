# v3: HiSVSIM + Spark Quantum Simulator

A distributed quantum circuit simulator that splits circuits into parallelizable parts and runs them on Spark.

## How It Works

1. **Input**: You give it a quantum circuit (list of gates)
2. **Partitioning**: HiSVSIM figures out which gates can run in parallel by building a dependency graph
3. **Execution**: Spark runs the parallel gates across multiple workers
4. **Merge**: Results get combined back into the final quantum state

## What It Can Do

- Simulates up to 30 qubits (1.07 billion amplitudes) for non-stabilizer circuits
- Handles both sparse states (few non-zero amplitudes) and dense states (all amplitudes)
- Supports fault tolerance with checkpoints and recovery
- Works in parallel or sequential mode

## Quick Start

```python
from driver import SparkHiSVSIMDriver
from v2_common import config, circuits

# Setup
cfg = config.SimulatorConfig(
    base_path=Path("./data"),
    spark_master="local[2]",
    spark_shuffle_partitions=4,
    batch_size=10,
)
cfg.ensure_paths()

# Run a circuit
circuit = circuits.generate_ghz_circuit(3)
with SparkHiSVSIMDriver(cfg, enable_parallel=True) as driver:
    result = driver.run_circuit(circuit, enable_parallel=True, resume=False)
    state = driver.get_state_vector(result)
```

## Tested Gates

- **RY gates**: 30 qubits max - RY(π/4) on each qubit
- **H+T gates**: 25 qubits max - Hadamard then T on each qubit
- **H+T+CR gates**: Testing 25-30 qubits
- **G gates**: Testing 25-30 qubits
- **R gates**: Testing 25-30 qubits
- **CU gates**: Testing 25-30 qubits

Plus all standard gates: H, X, Y, Z, S, T, CNOT, CZ, CY, SWAP, CR

## Setup

```bash
pip install -r requirements.txt
```

For Spark, you need Java installed. Or use Docker:

```bash
cd ../v2_spark
docker-compose run --rm -v "$(pwd)/../v3_hisvsim_spark:/v3" quantum-simulator bash
```

## Files

- `src/driver.py` - Main driver that orchestrates everything
- `src/hisvsim/` - Circuit partitioning using HiSVSIM
- `src/v2_common/` - Shared code (gates, state management, etc.)
- `tests/` - Test suite

## References

- Paper: "Efficient Hierarchical State Vector Simulation of Quantum Circuits via Acyclic Graph Partitioning", IEEE CLUSTER 2022
