#!/usr/bin/env python3
"""
Demo script to verify Spark quantum simulator is working.
Run this inside the Docker container to test end-to-end.
"""
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.driver import SparkQuantumDriver
from src.config import SimulatorConfig
import numpy as np


def main():
    print("=" * 60)
    print("Spark Quantum Simulator Demo")
    print("=" * 60)
    
    # Create config with temp directory
    base_path = Path(__file__).parent.parent / "data" / "demo"
    config = SimulatorConfig(
        run_id="demo_run",
        base_path=base_path,
        batch_size=5,
        spark_master="local[2]",
        spark_shuffle_partitions=4,
        spark_driver_memory="1g",
    )
    
    print(f"\nConfiguration:")
    print(f"  Run ID: {config.run_id}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Spark master: {config.spark_master}")
    
    # Initialize driver
    print("\n" + "-" * 60)
    print("Initializing Spark...")
    print("-" * 60)
    
    with SparkQuantumDriver(config) as driver:
        print("✓ SparkSession created successfully")
        print(f"  Spark version: {driver.spark.version}")
        
        # Test 1: GHZ circuit
        print("\n" + "-" * 60)
        print("Test 1: GHZ(4) circuit")
        print("-" * 60)
        
        result = driver.run_ghz(4, resume=False)
        state = driver.get_state_dict(result)
        
        print(f"  Gates: {result.n_gates}")
        print(f"  Batches: {result.n_batches}")
        print(f"  Time: {result.elapsed_time:.3f}s")
        print(f"  Final version: {result.final_version}")
        print(f"  Non-zero amplitudes: {len(state)}")
        
        # Verify GHZ state
        sqrt2_inv = 1 / np.sqrt(2)
        expected_0000 = 0b0000
        expected_1111 = 0b1111
        
        if len(state) == 2 and expected_0000 in state and expected_1111 in state:
            if (np.isclose(abs(state[expected_0000]), sqrt2_inv, atol=1e-6) and
                np.isclose(abs(state[expected_1111]), sqrt2_inv, atol=1e-6)):
                print("  ✓ GHZ state verified: (|0000⟩ + |1111⟩)/√2")
            else:
                print("  ✗ Amplitude values incorrect")
                return 1
        else:
            print(f"  ✗ Expected 2 states, got {len(state)}: {state}")
            return 1
        
        # Test 2: QFT circuit
        print("\n" + "-" * 60)
        print("Test 2: QFT(3) circuit")
        print("-" * 60)
        
        result = driver.run_qft(3, resume=False)
        state_arr = driver.get_state_vector(result)
        
        print(f"  Gates: {result.n_gates}")
        print(f"  Batches: {result.n_batches}")
        print(f"  Time: {result.elapsed_time:.3f}s")
        
        # Verify normalization
        norm = np.linalg.norm(state_arr)
        if np.isclose(norm, 1.0, atol=1e-6):
            print(f"  ✓ State normalized: ||ψ|| = {norm:.6f}")
        else:
            print(f"  ✗ Normalization failed: ||ψ|| = {norm}")
            return 1
        
        # QFT|0⟩ should give uniform superposition
        expected_amp = 1 / np.sqrt(8)  # 1/√(2^3)
        if np.allclose(np.abs(state_arr), expected_amp, atol=1e-6):
            print(f"  ✓ Uniform superposition verified: all |amp| = {expected_amp:.4f}")
        else:
            print("  ✗ Not uniform superposition")
            return 1
        
        # Test 3: Checkpoint verification
        print("\n" + "-" * 60)
        print("Test 3: Checkpoint verification")
        print("-" * 60)
        
        checkpoints = driver.checkpoint_manager.list_checkpoints()
        print(f"  Total checkpoints: {len(checkpoints)}")
        
        if len(checkpoints) > 0:
            latest = checkpoints[-1]
            print(f"  Latest: version={latest.state_version}, gate_seq={latest.last_gate_seq}")
            print("  ✓ Checkpoints working")
        else:
            print("  ✗ No checkpoints found")
            return 1
        
        # Test 4: WAL verification
        print("\n" + "-" * 60)
        print("Test 4: WAL verification")
        print("-" * 60)
        
        pending = driver.metadata_store.wal_get_pending(config.run_id)
        last_committed = driver.metadata_store.wal_get_last_committed(config.run_id)
        
        print(f"  Pending entries: {len(pending)}")
        if last_committed:
            print(f"  Last committed: gates[{last_committed.gate_start}:{last_committed.gate_end}]")
        
        if len(pending) == 0:
            print("  ✓ No pending WAL entries (all committed)")
        else:
            print("  ✗ Found pending WAL entries")
            return 1
    
    print("\n" + "=" * 60)
    print("✓ ALL TESTS PASSED - Spark implementation verified!")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
