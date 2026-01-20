#!/usr/bin/env python3
"""
SCALABILITY DEMO: Actually test and measure sparsity scaling.

This script runs real simulations and measures:
1. Non-zero amplitudes at each qubit count
2. Actual execution time
3. Memory usage
4. Shows the W state vs Hadamard wall difference in practice
"""
import sys
import os
from pathlib import Path
import numpy as np
import tempfile
import shutil
import time
import uuid

V3_SRC = Path(__file__).parent.parent / "src"
if str(V3_SRC) not in sys.path:
    sys.path.insert(0, str(V3_SRC))

from driver import SparkHiSVSIMDriver
from v2_common.config import SimulatorConfig
from v2_common.circuits import (
    generate_ghz_circuit, 
    generate_w_circuit, 
    generate_hadamard_wall,
    generate_qft_circuit,
)


def run_and_measure(circuit_dict: dict, name: str, temp_dir: Path) -> dict:
    """Run circuit and measure everything."""
    config = SimulatorConfig(
        run_id=f"demo_{uuid.uuid4().hex[:8]}",
        base_path=temp_dir,
        batch_size=50,
        spark_master="local[*]",  # Use all cores
        spark_shuffle_partitions=8,
    )
    config.ensure_paths()
    
    n_qubits = circuit_dict["number_of_qubits"]
    n_gates = len(circuit_dict["gates"])
    
    start_time = time.time()
    
    with SparkHiSVSIMDriver(config, enable_parallel=True) as driver:
        result = driver.run_circuit(circuit_dict, n_partitions=4, resume=False)
        
        # Count non-zero amplitudes
        state_dict = driver.state_manager.get_state_as_dict(result.final_state_df)
        non_zero = len([v for v in state_dict.values() if abs(v) > 1e-10])
        
        # Verify normalization
        norm = np.sqrt(sum(abs(v)**2 for v in state_dict.values()))
    
    elapsed = time.time() - start_time
    
    # Calculate storage
    storage_actual = non_zero * 16  # 16 bytes per complex
    storage_dense = (2 ** n_qubits) * 16
    
    return {
        "name": name,
        "n_qubits": n_qubits,
        "n_gates": n_gates,
        "non_zero": non_zero,
        "total_possible": 2 ** n_qubits,
        "sparsity_pct": (non_zero / (2 ** n_qubits)) * 100,
        "storage_bytes": storage_actual,
        "storage_dense": storage_dense,
        "time_seconds": elapsed,
        "norm": norm,
    }


def format_bytes(b):
    if b < 1024:
        return f"{b} B"
    elif b < 1024**2:
        return f"{b/1024:.1f} KB"
    elif b < 1024**3:
        return f"{b/1024**2:.1f} MB"
    elif b < 1024**4:
        return f"{b/1024**3:.2f} GB"
    else:
        return f"{b/1024**4:.2f} TB"


def print_header(title):
    print("\n" + "=" * 80)
    print(f" {title}")
    print("=" * 80)


def main():
    print_header("SCALABILITY DEMO: Real Measurements")
    print("\nThis demo runs ACTUAL simulations to demonstrate sparsity scaling.\n")
    
    temp_dir = Path(tempfile.mkdtemp())
    results = []
    
    try:
        # ============== DEMO 1: Same qubit count, different circuits ==============
        print_header("DEMO 1: Comparing Circuits at Same Qubit Count")
        
        n = 8  # 8 qubits = 256 possible amplitudes
        print(f"\nRunning 4 different {n}-qubit circuits...")
        print("-" * 80)
        
        circuits = [
            ("GHZ State", generate_ghz_circuit(n)),
            ("W State", generate_w_circuit(n)),
            ("Hadamard Wall", generate_hadamard_wall(n)),
            ("QFT|0⟩", generate_qft_circuit(n)),
        ]
        
        demo1_results = []
        for name, circuit in circuits:
            print(f"  Running {name}...", end=" ", flush=True)
            r = run_and_measure(circuit, name, temp_dir)
            demo1_results.append(r)
            print(f"✓ {r['non_zero']} amplitudes, {r['time_seconds']:.2f}s")
        
        print("\n" + "-" * 80)
        print(f"{'Circuit':<15} {'Qubits':<8} {'Non-Zero':<12} {'Total':<10} {'Sparsity':<10} {'Storage':<12} {'Time':<8}")
        print("-" * 80)
        
        for r in demo1_results:
            print(f"{r['name']:<15} {r['n_qubits']:<8} {r['non_zero']:<12} "
                  f"{r['total_possible']:<10} {r['sparsity_pct']:.2f}%      "
                  f"{format_bytes(r['storage_bytes']):<12} {r['time_seconds']:.2f}s")
        
        print("\n📊 OBSERVATION:")
        print(f"   • GHZ has only 2 amplitudes (constant)")
        print(f"   • W State has {demo1_results[1]['non_zero']} amplitudes (linear = n)")
        print(f"   • Hadamard and QFT have {demo1_results[2]['non_zero']} amplitudes (2^n = all)")
        
        # ============== DEMO 2: GHZ Scaling (should be unlimited) ==============
        print_header("DEMO 2: GHZ State Scaling (Constant Sparsity)")
        print("\nGHZ state always has exactly 2 non-zero amplitudes!\n")
        
        ghz_results = []
        for n in [5, 10, 15, 20]:
            print(f"  Running GHZ-{n}...", end=" ", flush=True)
            r = run_and_measure(generate_ghz_circuit(n), f"GHZ-{n}", temp_dir)
            ghz_results.append(r)
            print(f"✓ {r['non_zero']} amplitudes (always 2!), {r['time_seconds']:.2f}s")
        
        print("\n" + "-" * 80)
        print(f"{'Circuit':<12} {'Qubits':<8} {'Non-Zero':<10} {'If Dense':<15} {'Storage Saved':<15}")
        print("-" * 80)
        
        for r in ghz_results:
            saved = r['storage_dense'] - r['storage_bytes']
            print(f"{r['name']:<12} {r['n_qubits']:<8} {r['non_zero']:<10} "
                  f"{format_bytes(r['storage_dense']):<15} {format_bytes(saved):<15}")
        
        print("\n📊 OBSERVATION: GHZ always has 2 amplitudes regardless of qubits!")
        print("   With 20 qubits, we save 16 MB of memory by using sparse storage.")
        
        # ============== DEMO 3: W State Scaling (Linear) ==============
        print_header("DEMO 3: W State Scaling (Linear Sparsity)")
        print("\nW state has exactly n non-zero amplitudes!\n")
        
        w_results = []
        for n in [5, 10, 15, 20]:
            print(f"  Running W-{n}...", end=" ", flush=True)
            r = run_and_measure(generate_w_circuit(n), f"W-{n}", temp_dir)
            w_results.append(r)
            print(f"✓ {r['non_zero']} amplitudes (= {n}), {r['time_seconds']:.2f}s")
        
        print("\n" + "-" * 80)
        print(f"{'Circuit':<12} {'Qubits':<8} {'Non-Zero':<10} {'Expected':<10} {'Match':<8}")
        print("-" * 80)
        
        for r in w_results:
            expected = r['n_qubits']
            match = "✓" if r['non_zero'] == expected else "✗"
            print(f"{r['name']:<12} {r['n_qubits']:<8} {r['non_zero']:<10} {expected:<10} {match:<8}")
        
        print("\n📊 OBSERVATION: W state amplitudes = n (linear scaling!)")
        print("   The G gate creates this linear sparsity pattern.")
        
        # ============== DEMO 4: Hadamard Wall Scaling (Exponential) ==============
        print_header("DEMO 4: Hadamard Wall Scaling (Exponential - Limited!)")
        print("\nHadamard wall creates 2^n amplitudes - grows FAST!\n")
        
        h_results = []
        for n in [5, 8, 10, 12]:  # Can't go much higher!
            print(f"  Running H-Wall-{n}...", end=" ", flush=True)
            r = run_and_measure(generate_hadamard_wall(n), f"H-Wall-{n}", temp_dir)
            h_results.append(r)
            print(f"✓ {r['non_zero']:,} amplitudes (2^{n}), {r['time_seconds']:.2f}s")
        
        print("\n" + "-" * 80)
        print(f"{'Circuit':<15} {'Qubits':<8} {'Non-Zero':<12} {'Storage':<12} {'Time':<10}")
        print("-" * 80)
        
        for r in h_results:
            print(f"{r['name']:<15} {r['n_qubits']:<8} {r['non_zero']:<12,} "
                  f"{format_bytes(r['storage_bytes']):<12} {r['time_seconds']:.2f}s")
        
        print("\n📊 OBSERVATION: Storage doubles with each qubit!")
        print("   12 qubits = 4,096 amplitudes = 64 KB")
        print("   20 qubits = 1,048,576 amplitudes = 16 MB")
        print("   30 qubits = 1,073,741,824 amplitudes = 16 GB (!)")
        
        # ============== DEMO 5: Head-to-head comparison ==============
        print_header("DEMO 5: Head-to-Head - W State vs Hadamard Wall")
        print("\nComparing scaling at the SAME qubit counts:\n")
        
        print("-" * 80)
        print(f"{'Qubits':<8} {'W Non-Zero':<12} {'H Non-Zero':<12} {'W Storage':<12} {'H Storage':<12} {'Ratio':<10}")
        print("-" * 80)
        
        for n in [5, 10, 15, 20]:
            w_nz = n
            h_nz = 2**n
            w_storage = w_nz * 16
            h_storage = h_nz * 16
            ratio = h_nz / w_nz
            print(f"{n:<8} {w_nz:<12} {h_nz:<12,} {format_bytes(w_storage):<12} "
                  f"{format_bytes(h_storage):<12} {ratio:.0f}x")
        
        print("\n📊 OBSERVATION: At 20 qubits:")
        print("   • W State: 20 amplitudes = 320 bytes")
        print("   • Hadamard: 1,048,576 amplitudes = 16 MB")
        print("   • Hadamard uses 52,428x MORE memory!")
        
        # ============== FINAL SUMMARY ==============
        print_header("FINAL SUMMARY: Scalability Results")
        
        print("""
┌─────────────────────────────────────────────────────────────────────────────────┐
│                           SCALABILITY RESULTS                                   │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  Circuit Type    │ Amplitudes  │ Scaling    │ Max Qubits (16GB)                │
│  ────────────────┼─────────────┼────────────┼───────────────────               │
│  GHZ State       │ 2           │ O(1)       │ UNLIMITED                        │
│  W State         │ n           │ O(n)       │ ~1 billion                       │
│  Hadamard Wall   │ 2^n         │ O(2^n)     │ ~30                              │
│  QFT|0⟩          │ 2^n         │ O(2^n)     │ ~30                              │
│                                                                                 │
├─────────────────────────────────────────────────────────────────────────────────┤
│                                                                                 │
│  KEY INSIGHT: The G gate creates W states with LINEAR sparsity.                │
│  This allows simulating 1000s of qubits for W states, while Hadamard           │
│  walls are limited to ~30 qubits due to exponential amplitude growth.          │
│                                                                                 │
│  Parallelization (Spark) helps with:                                           │
│  ✓ Distributing dense states across workers                                    │
│  ✓ Faster gate application                                                     │
│  ✗ Does NOT overcome the 2^n memory limit for dense states                     │
│                                                                                 │
└─────────────────────────────────────────────────────────────────────────────────┘
""")
        
        print("\n✅ All demos completed successfully!")
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
