#!/usr/bin/env python3
"""
Sparsity Analysis: Understanding quantum state sparsity.

Key questions answered:
1. How does sparsity affect the number of qubits we can simulate?
2. What's the difference between W state (sparse) and Hadamard wall (dense)?
3. How does storage scale for different circuits?
"""
import sys
from pathlib import Path
import numpy as np
import tempfile
import shutil

V3_SRC = Path(__file__).parent.parent / "src"
if str(V3_SRC) not in sys.path:
    sys.path.insert(0, str(V3_SRC))

from driver import SparkHiSVSIMDriver
from v2_common.config import SimulatorConfig


def analyze_circuit(circuit_dict: dict, name: str, temp_dir: Path) -> dict:
    """Run circuit and analyze sparsity."""
    import uuid
    
    config = SimulatorConfig(
        run_id=f"sparsity_{uuid.uuid4().hex[:8]}",
        base_path=temp_dir,
        batch_size=10,
        spark_master="local[2]",
        spark_shuffle_partitions=4,
    )
    config.ensure_paths()
    
    n_qubits = circuit_dict["number_of_qubits"]
    
    with SparkHiSVSIMDriver(config, enable_parallel=True) as driver:
        result = driver.run_circuit(circuit_dict, n_partitions=2, resume=False)
        
        # Count non-zero amplitudes
        state_dict = driver.state_manager.get_state_as_dict(result.final_state_df)
        non_zero = len([v for v in state_dict.values() if abs(v) > 1e-10])
        
        # Total possible amplitudes
        total_possible = 2 ** n_qubits
        
        # Storage (bytes): each amplitude = 16 bytes (complex128)
        storage_actual = non_zero * 16
        storage_dense = total_possible * 16
    
    return {
        "name": name,
        "n_qubits": n_qubits,
        "non_zero": non_zero,
        "total_possible": total_possible,
        "sparsity_ratio": non_zero / total_possible,
        "storage_bytes": storage_actual,
        "storage_if_dense": storage_dense,
        "storage_saved": storage_dense - storage_actual,
    }


# ============== Circuit Generators (use v2_common) ==============

from v2_common.circuits import (
    generate_ghz_circuit, 
    generate_w_circuit, 
    generate_hadamard_wall,
    generate_qft_circuit,
)


def format_bytes(b):
    """Format bytes as human-readable."""
    if b < 1024:
        return f"{b} B"
    elif b < 1024**2:
        return f"{b/1024:.1f} KB"
    elif b < 1024**3:
        return f"{b/1024**2:.1f} MB"
    else:
        return f"{b/1024**3:.1f} GB"


def main():
    print("=" * 80)
    print("SPARSITY ANALYSIS: Understanding Quantum State Sparsity")
    print("=" * 80)
    
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # ============== PART 1: Compare sparsity patterns ==============
        print("\n" + "=" * 80)
        print("PART 1: Sparsity Comparison at Same Qubit Count (5 qubits)")
        print("=" * 80)
        
        results = []
        n = 5
        
        print(f"\nAnalyzing {n}-qubit circuits...")
        
        # GHZ - constant sparsity
        r = analyze_circuit(generate_ghz_circuit(n), "GHZ State", temp_dir)
        results.append(r)
        print(f"  GHZ:     {r['non_zero']:,} non-zero amplitudes (constant)")
        
        # W state - linear sparsity  
        r = analyze_circuit(generate_w_circuit(n), "W State", temp_dir)
        results.append(r)
        print(f"  W State: {r['non_zero']:,} non-zero amplitudes (linear = n)")
        
        # Hadamard wall - exponential
        r = analyze_circuit(generate_hadamard_wall(n), "Hadamard Wall", temp_dir)
        results.append(r)
        print(f"  H-Wall:  {r['non_zero']:,} non-zero amplitudes (exponential = 2^n)")
        
        # QFT on |0⟩ - exponential
        r = analyze_circuit(generate_qft_circuit(n), "QFT|0⟩", temp_dir)
        results.append(r)
        print(f"  QFT|0⟩:  {r['non_zero']:,} non-zero amplitudes (exponential = 2^n)")
        
        print("\n" + "-" * 80)
        print(f"{'Circuit':<15} {'Non-Zero':<12} {'Total':<12} {'Sparsity':<12} {'Storage':<12}")
        print("-" * 80)
        for r in results:
            print(f"{r['name']:<15} {r['non_zero']:<12,} {r['total_possible']:<12,} "
                  f"{r['sparsity_ratio']:.6f}     {format_bytes(r['storage_bytes']):<12}")
        
        # ============== PART 2: Scaling analysis ==============
        print("\n" + "=" * 80)
        print("PART 2: How Sparsity Scales with Qubits")
        print("=" * 80)
        print("\nThis is the key insight about scalability:")
        print()
        print(f"{'Qubits':<8} {'GHZ':<12} {'W State':<12} {'H-Wall':<15} {'Storage (H-Wall)':<15}")
        print("-" * 80)
        
        for n in [5, 10, 15, 20, 25, 30]:
            ghz_nonzero = 2  # Always 2
            w_nonzero = n    # Linear: n
            hwall_nonzero = 2**n  # Exponential: 2^n
            
            # Storage for H-Wall (the limiting case)
            storage = hwall_nonzero * 16  # 16 bytes per complex number
            
            print(f"{n:<8} {ghz_nonzero:<12} {w_nonzero:<12} {hwall_nonzero:<15,} {format_bytes(storage):<15}")
        
        # ============== PART 3: Maximum qubits ==============
        print("\n" + "=" * 80)
        print("PART 3: Maximum Qubits You Can Simulate")
        print("=" * 80)
        print()
        print("Given 16 GB RAM, how many qubits can you simulate?")
        print()
        
        ram_gb = 16
        ram_bytes = ram_gb * (1024**3)
        
        print(f"{'Circuit Type':<20} {'Max Qubits':<15} {'Why':<40}")
        print("-" * 80)
        
        # GHZ: always 2 amplitudes
        print(f"{'GHZ State':<20} {'UNLIMITED':<15} {'Only 2 amplitudes regardless of n':<40}")
        
        # W State: n amplitudes
        max_w = ram_bytes // 16  # Can have this many amplitudes
        print(f"{'W State':<20} {max_w:,} qubits     {'Linear: n amplitudes':<40}")
        
        # Hadamard wall: 2^n amplitudes
        import math
        max_h = int(math.log2(ram_bytes / 16))
        print(f"{'Hadamard Wall':<20} {max_h} qubits        {'Exponential: 2^n amplitudes = 2^{max_h}':<40}")
        
        # QFT
        print(f"{'QFT on |0⟩':<20} {max_h} qubits        {'Same as Hadamard - dense output':<40}")
        
        # ============== PART 4: The key insight ==============
        print("\n" + "=" * 80)
        print("PART 4: KEY INSIGHT - What Determines Scalability")
        print("=" * 80)
        print("""
┌─────────────────────────────────────────────────────────────────────────────┐
│ The NUMBER OF NON-ZERO AMPLITUDES determines how many qubits you can       │
│ simulate, NOT just the number of qubits itself!                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│ SPARSE STATES (can scale to many qubits):                                   │
│   • GHZ: 2 amplitudes      → can simulate 1000s of qubits                  │
│   • W State: n amplitudes  → can simulate 1000s of qubits                  │
│                                                                             │
│ DENSE STATES (limited by exponential growth):                               │
│   • Hadamard Wall: 2^n amplitudes → limited to ~30 qubits                  │
│   • QFT|0⟩: 2^n amplitudes       → limited to ~30 qubits                   │
│   • Random circuits: 2^n amplitudes → limited to ~30 qubits                │
│                                                                             │
├─────────────────────────────────────────────────────────────────────────────┤
│ OTHER PARAMETERS THAT INFLUENCE SCALABILITY:                                │
│   1. Memory per worker (more RAM = more amplitudes)                        │
│   2. Number of Spark workers (distribute state across machines)            │
│   3. Disk storage (checkpoint intermediate states)                         │
│   4. Gate complexity (non-stabilizer gates can increase sparsity)          │
└─────────────────────────────────────────────────────────────────────────────┘
""")
        
        # ============== PART 5: Parallelization insight ==============
        print("=" * 80)
        print("PART 5: Parallelization vs Sparsity Trade-off")
        print("=" * 80)
        print("""
Note: "Hadamard wall is highly parallelizable"

This is the PARADOX:
┌─────────────────────────────────────────────────────────────────────────────┐
│ Hadamard Wall:                                                              │
│   ✅ Each H gate is independent → perfectly parallelizable                  │
│   ❌ Creates 2^n amplitudes → memory explodes                               │
│   Result: Parallelism doesn't help if you can't fit the state in memory!   │
├─────────────────────────────────────────────────────────────────────────────┤
│ W State:                                                                    │
│   ❌ G gates are sequential (dependencies)                                   │
│   ✅ Only n amplitudes → memory scales linearly                             │
│   Result: Even sequential execution can handle 1000s of qubits!            │
└─────────────────────────────────────────────────────────────────────────────┘
""")
        
        print("=" * 80)
        print("SUMMARY")
        print("=" * 80)
        print("""
What affects the number of qubits that can be simulated:

1. PRIMARY FACTOR: Number of non-zero amplitudes (sparsity)
   - Sparse states (GHZ, W): scale to 1000s of qubits
   - Dense states (H-wall, QFT|0⟩): limited to ~30 qubits

2. STORAGE: O(non_zero × 16 bytes per amplitude)
   - Sparse: O(n) or O(1)
   - Dense: O(2^n) - exponential explosion

3. The G gate creates W states with LINEAR sparsity (n amplitudes)
   This is why W states can scale far beyond Hadamard walls.

4. Parallelization helps compute faster, but doesn't solve memory limits.
   Even perfectly parallel Hadamard walls hit the 2^n memory wall.
""")
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
