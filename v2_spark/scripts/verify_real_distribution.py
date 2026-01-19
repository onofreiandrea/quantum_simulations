#!/usr/bin/env python3
"""
Verify ACTUAL distribution across multiple Spark workers.

This script:
1. Connects to a real Spark cluster (not local[*])
2. Runs a quantum circuit simulation
3. Monitors task distribution across workers
4. Shows partition distribution
5. Verifies work is actually split across machines
"""
from __future__ import annotations

import os
import sys
import time
import json
from pathlib import Path
import tempfile
import shutil

sys.path.insert(0, str(Path(__file__).parent.parent))

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
import numpy as np

from src.config import SimulatorConfig
from src.driver import SparkQuantumDriver
from src.circuits import generate_qft_circuit, generate_ghz_circuit


def get_cluster_info(spark: SparkSession):
    """Get information about the Spark cluster."""
    sc = spark.sparkContext
    
    print("="*70)
    print("CLUSTER INFORMATION")
    print("="*70)
    print(f"Spark Version: {spark.version}")
    print(f"Master URL: {sc.master}")
    print(f"App Name: {sc.appName}")
    print(f"Default Parallelism: {sc.defaultParallelism}")
    
    # Get executor information
    status_tracker = sc.statusTracker()
    executor_infos = status_tracker.getExecutorInfos()
    
    print(f"\nActive Executors: {len(executor_infos)}")
    for i, info in enumerate(executor_infos):
        print(f"  Executor {i+1}:")
        print(f"    Host: {info.executorHost}")
        print(f"    Cores: {info.totalCores}")
        print(f"    Max Tasks: {info.maxTasks}")
    
    return len(executor_infos) > 1


def monitor_task_distribution(spark: SparkSession, df, operation_name: str):
    """Monitor how tasks are distributed across executors."""
    print(f"\n{'='*70}")
    print(f"MONITORING: {operation_name}")
    print("="*70)
    
    sc = spark.sparkContext
    
    # Force evaluation and collect partition info
    print("Collecting partition distribution...")
    
    # Get partition distribution
    partition_info = df.rdd.mapPartitionsWithIndex(
        lambda idx, it: [(idx, sum(1 for _ in it))]
    ).collect()
    
    print(f"\nPartition Distribution:")
    print(f"  Total partitions: {len(partition_info)}")
    print(f"  Non-empty partitions: {sum(1 for _, count in partition_info if count > 0)}")
    
    # Group by partition size
    size_groups = {}
    for pid, count in partition_info:
        if count > 0:
            size_groups[count] = size_groups.get(count, 0) + 1
    
    print(f"\nPartition sizes:")
    for size, count in sorted(size_groups.items()):
        print(f"  {count} partitions with {size} rows")
    
    # Force evaluation to trigger task execution
    print("\nForcing evaluation (this will show task distribution)...")
    start_time = time.time()
    row_count = df.count()
    elapsed = time.time() - start_time
    
    print(f"\nResults:")
    print(f"  Total rows: {row_count}")
    print(f"  Evaluation time: {elapsed:.2f}s")
    
    # Get task distribution from Spark UI (if accessible)
    print(f"\n📊 Check Spark UI for task distribution:")
    print(f"   Master UI: http://localhost:8080")
    print(f"   Application UI: http://localhost:4040")
    print(f"   Look for 'Stages' tab to see task distribution across executors")


def test_distributed_simulation(config: SimulatorConfig):
    """Run a simulation and verify distribution."""
    print("\n" + "="*70)
    print("RUNNING DISTRIBUTED QUANTUM SIMULATION")
    print("="*70)
    
    # Use QFT circuit (dense state, good for distribution)
    n_qubits = 12  # 4096 amplitudes - should distribute well
    circuit = generate_qft_circuit(n_qubits)
    
    print(f"\nCircuit: QFT on {n_qubits} qubits")
    print(f"  Gates: {len(circuit['gates'])}")
    print(f"  Expected amplitudes: {2**n_qubits}")
    
    start_time = time.time()
    
    with SparkQuantumDriver(config) as driver:
        # Get Spark session to monitor
        spark = driver.spark
        
        # Check cluster info
        is_cluster = get_cluster_info(spark)
        
        if not is_cluster:
            print("\n⚠️  WARNING: Not running on a cluster!")
            print("   Only 1 executor detected. For real distribution,")
            print("   use: docker-compose -f docker-compose-cluster.yml up")
            return
        
        print("\n✓ Running on multi-worker cluster")
        
        # Run simulation
        print("\nRunning circuit simulation...")
        result = driver.run_circuit(circuit, resume=False)
        
        # Monitor distribution of final state
        final_df = result.final_state_df
        monitor_task_distribution(spark, final_df, "Final State Distribution")
        
        # Get final state info
        state_vector = driver.get_state_vector(result)
        non_zero = np.sum(np.abs(state_vector) > 1e-10)
        
        elapsed = time.time() - start_time
        
        print(f"\n{'='*70}")
        print("SIMULATION RESULTS")
        print("="*70)
        print(f"  Total time: {elapsed:.2f}s")
        print(f"  Amplitudes: {len(state_vector)}")
        print(f"  Non-zero amplitudes: {non_zero}")
        print(f"  Norm: {np.linalg.norm(state_vector):.6f}")
        
        # Verify correctness
        if abs(np.linalg.norm(state_vector) - 1.0) < 1e-10:
            print("  ✓ State is normalized correctly")
        else:
            print(f"  ⚠️  Norm should be 1.0, got {np.linalg.norm(state_vector)}")
    
    print(f"\n📊 To see detailed task distribution:")
    print(f"   1. Open http://localhost:8080 (Spark Master UI)")
    print(f"   2. Click on your application")
    print(f"   3. Go to 'Stages' tab")
    print(f"   4. Click on a stage to see task distribution across workers")


def test_partition_distribution(config: SimulatorConfig):
    """Test that partitions are actually distributed."""
    print("\n" + "="*70)
    print("TESTING PARTITION DISTRIBUTION")
    print("="*70)
    
    from src.state_manager import StateManager
    from src.gate_applicator import GateApplicator
    from src.gates import HadamardGate
    
    spark = SparkSession.builder \
        .appName("PartitionDistributionTest") \
        .master(config.spark_master) \
        .config("spark.sql.shuffle.partitions", str(config.spark_shuffle_partitions)) \
        .getOrCreate()
    
    try:
        state_manager = StateManager(spark, config)
        applicator = GateApplicator(spark, num_partitions=config.spark_shuffle_partitions)
        
        n_qubits = 14  # 16384 amplitudes
        gates = [HadamardGate(i) for i in range(n_qubits)]
        applicator.register_gates(gates)
        
        print(f"\nCreating dense state with {2**n_qubits} amplitudes")
        print(f"Using {config.spark_shuffle_partitions} partitions")
        
        state = state_manager.initialize_state(n_qubits)
        
        # Apply a few gates and check distribution
        for i in range(min(5, n_qubits)):
            state = applicator.apply_gate(state, gates[i])
            n_rows = state.count()
            n_parts = state.rdd.getNumPartitions()
            
            # Get partition distribution
            part_dist = state.rdd.mapPartitionsWithIndex(
                lambda idx, it: [(idx, sum(1 for _ in it))]
            ).collect()
            
            non_empty = [pid for pid, count in part_dist if count > 0]
            
            print(f"\n  After H[{i}]:")
            print(f"    Rows: {n_rows}")
            print(f"    Partitions: {n_parts}")
            print(f"    Non-empty partitions: {len(non_empty)}")
            
            if len(non_empty) > 1:
                print(f"    ✓ State is distributed across {len(non_empty)} partitions")
            else:
                print(f"    ⚠️  State is in single partition")
        
        applicator.cleanup()
        
    finally:
        spark.stop()


def main():
    master_url = os.environ.get("SPARK_MASTER_URL", "spark://spark-master:7077")
    
    print("="*70)
    print("REAL DISTRIBUTION VERIFICATION")
    print("="*70)
    print(f"\nConnecting to Spark cluster: {master_url}")
    print("\nMake sure the cluster is running:")
    print("  docker-compose -f docker-compose-cluster.yml up -d")
    
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        config = SimulatorConfig(
            run_id="distribution_test",
            base_path=temp_dir,
            batch_size=50,
            spark_master=master_url,
            spark_shuffle_partitions=16,  # More partitions = better distribution
        )
        config.ensure_paths()
        
        # Test partition distribution
        test_partition_distribution(config)
        
        # Test full simulation
        test_distributed_simulation(config)
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        print("\n" + "="*70)
        print("VERIFICATION COMPLETE")
        print("="*70)


if __name__ == "__main__":
    main()
