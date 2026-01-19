#!/usr/bin/env python3
"""
Verify that Spark is ACTUALLY distributing work properly.

This script:
1. Checks if Spark cluster is available
2. Verifies partitioning of state DataFrames
3. Tests that dense states actually use multiple partitions
4. Shows task distribution across workers
5. Measures actual parallelism benefit
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path
import tempfile
import shutil

sys.path.insert(0, str(Path(__file__).parent.parent))

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import numpy as np

from src.config import SimulatorConfig
from src.state_manager import StateManager
from src.gate_applicator import GateApplicator
from src.gates import HadamardGate, XGate, CNOTGate
from src.driver import SparkQuantumDriver
from src.circuits import generate_qft_circuit


def get_spark_session(master_url: str = None):
    """Create SparkSession, connecting to cluster if available."""
    master = master_url or os.environ.get("SPARK_MASTER_URL", "local[*]")
    
    print(f"Connecting to Spark: {master}")
    
    builder = (
        SparkSession.builder
        .appName("QuantumSimulator-DistributionTest")
        .master(master)
        .config("spark.sql.shuffle.partitions", "16")
        .config("spark.default.parallelism", "16")
        .config("spark.driver.memory", "2g")
        .config("spark.executor.memory", "2g")
    )
    
    return builder.getOrCreate()


def test_cluster_connectivity(spark: SparkSession):
    """Test basic cluster connectivity."""
    print("\n" + "="*70)
    print("TEST 1: Cluster Connectivity")
    print("="*70)
    
    # Get cluster info
    sc = spark.sparkContext
    
    print(f"Spark Version: {spark.version}")
    print(f"Master: {sc.master}")
    print(f"App Name: {sc.appName}")
    print(f"Default Parallelism: {sc.defaultParallelism}")
    
    # Count executors (workers)
    # This is a trick to count active executors
    rdd = sc.parallelize(range(100), 16)
    executor_ids = rdd.mapPartitions(
        lambda _: [os.environ.get("SPARK_EXECUTOR_ID", "driver")]
    ).distinct().collect()
    
    print(f"Active Executors: {len(executor_ids)}")
    print(f"Executor IDs: {executor_ids}")
    
    if sc.master.startswith("local"):
        print("\n⚠️  WARNING: Running in LOCAL mode - no actual distribution!")
        print("   For real distribution, use: spark://spark-master:7077")
        return False
    else:
        print("\n✓ Connected to Spark cluster")
        return True


def test_dataframe_partitioning(spark: SparkSession):
    """Test that DataFrames are properly partitioned."""
    print("\n" + "="*70)
    print("TEST 2: DataFrame Partitioning")
    print("="*70)
    
    # Create a test DataFrame with many rows
    n_rows = 10000
    data = [(i, float(i), 0.0) for i in range(n_rows)]
    df = spark.createDataFrame(data, ["idx", "real", "imag"])
    
    print(f"Created DataFrame with {n_rows} rows")
    print(f"Number of partitions: {df.rdd.getNumPartitions()}")
    
    # Check partition distribution
    partition_sizes = df.rdd.mapPartitionsWithIndex(
        lambda idx, it: [(idx, sum(1 for _ in it))]
    ).collect()
    
    print("\nPartition distribution:")
    for pid, count in sorted(partition_sizes):
        print(f"  Partition {pid}: {count} rows")
    
    # Repartition by idx for better distribution
    df_repartitioned = df.repartition(16, "idx")
    print(f"\nAfter repartition by idx: {df_repartitioned.rdd.getNumPartitions()} partitions")
    
    new_sizes = df_repartitioned.rdd.mapPartitionsWithIndex(
        lambda idx, it: [(idx, sum(1 for _ in it))]
    ).collect()
    
    print("New partition distribution:")
    for pid, count in sorted(new_sizes)[:5]:
        print(f"  Partition {pid}: {count} rows")
    print(f"  ... ({len(new_sizes)} total partitions)")
    
    return True


def test_gate_parallelism(spark: SparkSession, temp_dir: Path):
    """Test that gate application actually uses parallelism."""
    print("\n" + "="*70)
    print("TEST 3: Gate Application Parallelism")
    print("="*70)
    
    config = SimulatorConfig(
        base_path=temp_dir,
        batch_size=100,
        spark_master=spark.sparkContext.master,
        spark_shuffle_partitions=16,
    )
    
    state_manager = StateManager(spark, config)
    applicator = GateApplicator(spark)
    
    # Create a dense state (H on multiple qubits)
    n_qubits = 14  # 2^14 = 16384 amplitudes
    gates = [HadamardGate(i) for i in range(n_qubits)]
    applicator.register_gates(gates)
    
    # Start with |0⟩
    state = state_manager.initialize_state(n_qubits)
    print(f"\nApplying H to all {n_qubits} qubits (creating 2^{n_qubits} = {2**n_qubits} amplitudes)")
    
    start_time = time.time()
    for i, gate in enumerate(gates):
        state = applicator.apply_gate(state, gate)
        n_amps = state.count()
        n_parts = state.rdd.getNumPartitions()
        if i < 5 or i == n_qubits - 1:
            print(f"  After H[{i}]: {n_amps} amplitudes, {n_parts} partitions")
    
    elapsed = time.time() - start_time
    print(f"\nTotal time: {elapsed:.2f}s")
    
    # Check final state partitioning
    final_partitions = state.rdd.mapPartitionsWithIndex(
        lambda idx, it: [(idx, sum(1 for _ in it))]
    ).collect()
    
    non_empty = [(p, c) for p, c in final_partitions if c > 0]
    print(f"\nFinal state: {len(non_empty)} non-empty partitions")
    
    if len(non_empty) > 1:
        print("✓ State is distributed across multiple partitions")
    else:
        print("⚠️  State is in single partition - no parallelism!")
    
    applicator.cleanup()
    return len(non_empty) > 1


def test_parallel_speedup(spark: SparkSession, temp_dir: Path):
    """Measure actual speedup from parallelism."""
    print("\n" + "="*70)
    print("TEST 4: Parallel Speedup Measurement")
    print("="*70)
    
    n_qubits = 12  # 4096 amplitudes
    
    # Create a Hadamard wall circuit
    circuit = {
        "number_of_qubits": n_qubits,
        "gates": [{"qubits": [i], "gate": "H"} for i in range(n_qubits)]
    }
    
    # Test with different partition counts
    for n_partitions in [1, 4, 8, 16]:
        config = SimulatorConfig(
            base_path=temp_dir / f"p{n_partitions}",
            batch_size=n_qubits,
            spark_master=spark.sparkContext.master,
            spark_shuffle_partitions=n_partitions,
        )
        config.ensure_paths()
        
        start = time.time()
        with SparkQuantumDriver(config) as driver:
            # Override shuffle partitions
            driver.spark.conf.set("spark.sql.shuffle.partitions", str(n_partitions))
            result = driver.run_circuit(circuit, resume=False)
            _ = result.final_state_df.count()  # Force evaluation
        elapsed = time.time() - start
        
        print(f"  {n_partitions:2} partitions: {elapsed:.2f}s")
        
        shutil.rmtree(config.base_path, ignore_errors=True)


def test_qft_distribution(spark: SparkSession, temp_dir: Path):
    """Test QFT with proper distribution."""
    print("\n" + "="*70)
    print("TEST 5: QFT Circuit with Distribution Monitoring")
    print("="*70)
    
    n_qubits = 10
    circuit = generate_qft_circuit(n_qubits)
    
    config = SimulatorConfig(
        base_path=temp_dir / "qft_test",
        batch_size=20,
        spark_master=spark.sparkContext.master,
        spark_shuffle_partitions=16,
    )
    config.ensure_paths()
    
    print(f"Running QFT on {n_qubits} qubits ({len(circuit['gates'])} gates)")
    
    start = time.time()
    with SparkQuantumDriver(config) as driver:
        result = driver.run_circuit(circuit, resume=False)
        
        # Get distribution info
        final_df = result.final_state_df
        n_rows = final_df.count()
        n_parts = final_df.rdd.getNumPartitions()
        
    elapsed = time.time() - start
    
    print(f"\nResults:")
    print(f"  Amplitudes: {n_rows}")
    print(f"  Partitions: {n_parts}")
    print(f"  Time: {elapsed:.2f}s")
    
    # Verify correctness
    if n_rows == 2**n_qubits:
        print("  ✓ Correct number of amplitudes (dense state)")
    else:
        print(f"  ⚠️  Expected {2**n_qubits} amplitudes, got {n_rows}")
    
    shutil.rmtree(config.base_path, ignore_errors=True)


def main():
    print("="*70)
    print("SPARK DISTRIBUTION VERIFICATION")
    print("="*70)
    
    # Get Spark session
    master_url = os.environ.get("SPARK_MASTER_URL")
    spark = get_spark_session(master_url)
    
    try:
        # Create temp directory
        temp_dir = Path(tempfile.mkdtemp())
        
        # Run tests
        is_cluster = test_cluster_connectivity(spark)
        test_dataframe_partitioning(spark)
        test_gate_parallelism(spark, temp_dir)
        test_parallel_speedup(spark, temp_dir)
        test_qft_distribution(spark, temp_dir)
        
        # Summary
        print("\n" + "="*70)
        print("SUMMARY")
        print("="*70)
        
        if is_cluster:
            print("\n✓ Running on Spark cluster - work is distributed across workers")
        else:
            print("\n⚠️  Running in LOCAL mode:")
            print("   - Uses multiple threads on ONE machine")
            print("   - NOT truly distributed across multiple machines")
            print("   - For real distribution, deploy to Spark cluster")
        
        print("\nKey insights:")
        print("  - Sparse states (GHZ): Cannot benefit from parallelism (only 2 rows)")
        print("  - Dense states (QFT): CAN benefit from parallelism across partitions")
        print("  - For maximum parallelism, increase spark.sql.shuffle.partitions")
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        spark.stop()


if __name__ == "__main__":
    main()
