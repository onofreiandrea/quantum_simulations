# Spark Quantum Simulator (v2)

A distributed quantum circuit simulator using Apache Spark for parallel state vector computation.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           SparkQuantumDriver                            │
├─────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌──────────────┐  ┌────────────┐  ┌───────────────┐  │
│  │   Frontend  │  │ GateApplicator│  │ StateMgr   │  │ CheckpointMgr │  │
│  │  (circuit   │  │  (Spark DF    │  │ (Parquet)  │  │   (DuckDB)    │  │
│  │   parser)   │  │  transforms)  │  │            │  │               │  │
│  └─────────────┘  └──────────────┘  └────────────┘  └───────────────┘  │
└─────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────┐
│                          Spark Cluster                                   │
├──────────────────┬──────────────────┬──────────────────────────────────┤
│   Worker 1       │   Worker 2       │   Worker 3                       │
│  ┌──────────┐    │  ┌──────────┐    │  ┌──────────┐                    │
│  │Partition │    │  │Partition │    │  │Partition │                    │
│  │ 0, 4, 8  │    │  │ 1, 5, 9  │    │  │ 2, 6, 10 │                    │
│  └──────────┘    │  └──────────┘    │  └──────────┘                    │
└──────────────────┴──────────────────┴──────────────────────────────────┘
```

## Key Features

- **Distributed Computation**: State vector partitioned across Spark workers
- **Sparse Representation**: Only non-zero amplitudes stored (efficient for GHZ, etc.)
- **Fault Tolerance**: WAL + checkpointing for crash recovery
- **64-bit Indices**: Supports up to 63 qubits (for sparse states)
- **Non-Stabilizer Circuits**: Full support for T, CR, RY, G gates

## Quick Start

### Local Mode (Single Machine)

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
python -m pytest tests/ -v

# Run demo
python scripts/run_spark_demo.py

# Check scalability
python scripts/scalability_test.py
```

### Docker - Local Mode (Multi-threaded, Not Distributed)

```bash
# Run tests in local[*] mode (multi-threaded, not distributed)
docker-compose run --rm quantum-simulator python3 -m pytest tests/ -v

# Run distribution verification (shows partitioning but not true distribution)
docker-compose run --rm quantum-simulator python3 scripts/verify_distribution.py
```

### Docker - Real Cluster Mode (Multiple Workers, True Distribution)

```bash
# Start Spark cluster with 3 workers
./scripts/start_cluster.sh
# OR manually:
docker-compose -f docker-compose-cluster.yml up -d

# View Spark UIs:
# - Master: http://localhost:8080
# - Worker 1: http://localhost:8081
# - Worker 2: http://localhost:8082
# - Worker 3: http://localhost:8083

# Run distribution verification on real cluster
docker-compose -f docker-compose-cluster.yml run --rm quantum-simulator \
  python3 scripts/verify_real_distribution.py

# Run a simulation and monitor task distribution
docker-compose -f docker-compose-cluster.yml run --rm quantum-simulator \
  python3 scripts/run_spark_demo.py

# Stop cluster
docker-compose -f docker-compose-cluster.yml down
```

**To verify real distribution:**
1. Start the cluster (above)
2. Run `verify_real_distribution.py` - it will show:
   - Number of executors/workers
   - Partition distribution across workers
   - Task distribution in Spark UI
3. Open http://localhost:8080 → Click your app → "Stages" tab
4. Click on a stage → See tasks distributed across workers

## Scalability Limits

| Circuit Type | Max Qubits | Reason |
|-------------|-----------|--------|
| Sparse (GHZ) | **63** | 64-bit integer limit for indices |
| Dense (QFT) | **~20-22** | Memory: 2^n × 16 bytes |

### Memory Requirements (Dense States)

| Qubits | Amplitudes | Memory |
|--------|------------|--------|
| 10 | 1,024 | 16 KB |
| 20 | 1,048,576 | 16 MB |
| 30 | 1,073,741,824 | 16 GB |
| 40 | 1,099,511,627,776 | 16 TB |
| 50 | 1.13 × 10^15 | 16 PB |

## Distribution Verification

### Local Mode (Multi-threaded)
```
TEST 3: Gate Application Parallelism
  After H[13]: 16384 amplitudes, 16 partitions
  ✓ State is distributed across multiple partitions
  ⚠️  Running in LOCAL mode (multi-threaded, not distributed)
```

### Cluster Mode (True Distribution)
```
CLUSTER INFORMATION
  Active Executors: 3
    Executor 1: Host=spark-worker-1, Cores=2
    Executor 2: Host=spark-worker-2, Cores=2
    Executor 3: Host=spark-worker-3, Cores=2
  ✓ Running on multi-worker cluster

Partition Distribution:
  Total partitions: 16
  Non-empty partitions: 16
  Tasks distributed across 3 workers
```

## Test Suite

```
228 tests covering:
- Gate definitions and unitarity
- Circuit simulation correctness
- V1 SQLite parity
- Spark distribution
- Non-stabilizer circuits (T, CR, RY, G gates)
- Checkpointing/recovery
- Edge cases and fuzzing
```

## How It Works

### Gate Application (DataFrame Transformation)

```python
# 1-qubit gate: H on qubit q
state_df
  .withColumn("qubit_bit", (idx >> q) & 1)
  .join(gate_matrix, col("qubit_bit") == col("col"))
  .withColumn("new_idx", (idx ^ (qubit_bit << q)) | (row << q))
  .withColumn("new_real", g_real * real - g_imag * imag)
  .withColumn("new_imag", g_real * imag + g_imag * real)
  .groupBy("new_idx").agg(sum("new_real"), sum("new_imag"))
  .repartition(num_partitions, "idx")  # Distribute for parallelism
```

### Recovery Flow

```
1. Load latest checkpoint (if exists)
2. Check WAL for PENDING entries
3. Mark incomplete batches as FAILED
4. Resume from last committed gate
```

## File Structure

```
v2_spark/
├── src/
│   ├── config.py           # Configuration
│   ├── driver.py           # Main orchestrator
│   ├── gates.py            # Gate definitions
│   ├── gate_applicator.py  # Spark DataFrame transforms
│   ├── state_manager.py    # Parquet I/O
│   ├── checkpoint_manager.py
│   ├── recovery_manager.py
│   └── metadata_store.py   # DuckDB for WAL
├── tests/
│   ├── test_unit.py        # Pure Python tests
│   ├── test_spark_simulation.py
│   ├── test_adversarial.py # V1 parity tests
│   ├── test_non_stabilizer.py  # T, CR, RY gates
│   └── ...
├── scripts/
│   ├── verify_distribution.py      # Local mode check
│   ├── verify_real_distribution.py # Cluster mode check
│   ├── scalability_test.py
│   └── run_spark_demo.py
├── docker-compose.yml              # Local mode
├── docker-compose-cluster.yml     # Multi-worker cluster
└── Dockerfile
```

## Limitations

1. **Sparse circuits only** for > 30 qubits (memory constraint)
2. **63 qubit maximum** due to 64-bit index addressing
3. **Local mode** uses threads, not true distribution
4. For 50+ qubit dense simulation, need:
   - Petabytes of memory, OR
   - Tensor network methods, OR
   - Stabilizer formalism (Clifford-only circuits)
