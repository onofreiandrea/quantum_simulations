#!/usr/bin/env python3
"""
Demo script for HiSVSIM + Spark integration.

Shows how to use circuit partitioning with Spark execution.
"""
from __future__ import annotations

import sys
from pathlib import Path
import tempfile
import shutil

# Add src to path
V3_SPARK = Path(__file__).parent.parent
SRC_PATH = V3_SPARK / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

# Import directly from src directory
from driver import SparkHiSVSIMDriver
from v2_common import config, circuits

SimulatorConfig = config.SimulatorConfig
generate_ghz_circuit = circuits.generate_ghz_circuit
generate_qft_circuit = circuits.generate_qft_circuit


def main():
    """Run demo."""
    print("="*70)
    print("HiSVSIM + Spark Integration Demo")
    print("="*70)
    print()
    
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        config = SimulatorConfig(
            run_id="demo_hisvsim_spark",
            base_path=temp_dir,
            batch_size=10,
            spark_master="local[2]",
            spark_shuffle_partitions=4,
        )
        
        # Test 1: GHZ circuit
        print("Test 1: GHZ Circuit (3 qubits)")
        print("-" * 70)
        circuit = generate_ghz_circuit(3)
        
        with SparkHiSVSIMDriver(config) as driver:
            result = driver.run_circuit(circuit, n_partitions=2)
            state = driver.get_state_vector(result)
            
            print(f"  Partitions: {result.n_partitions}")
            print(f"  Gates: {result.n_gates}")
            print(f"  Time: {result.elapsed_time:.3f}s")
            print(f"  Final state norm: {sum(abs(x)**2 for x in state):.6f}")
            print(f"  Non-zero amplitudes: {sum(1 for x in state if abs(x) > 1e-10)}")
            print()
        
        # Test 2: QFT circuit
        print("Test 2: QFT Circuit (4 qubits)")
        print("-" * 70)
        circuit = generate_qft_circuit(4)
        
        config2 = SimulatorConfig(
            run_id="demo_hisvsim_spark_qft",
            base_path=temp_dir / "qft",
            batch_size=10,
            spark_master="local[2]",
            spark_shuffle_partitions=4,
        )
        
        with SparkHiSVSIMDriver(config2) as driver:
            result = driver.run_circuit(circuit, n_partitions=3)
            state = driver.get_state_vector(result)
            
            print(f"  Partitions: {result.n_partitions}")
            print(f"  Gates: {result.n_gates}")
            print(f"  Time: {result.elapsed_time:.3f}s")
            print(f"  Final state norm: {sum(abs(x)**2 for x in state):.6f}")
            print(f"  Non-zero amplitudes: {sum(1 for x in state if abs(x) > 1e-10)}")
            print()
        
        print("="*70)
        print("Demo complete!")
        print("="*70)
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
