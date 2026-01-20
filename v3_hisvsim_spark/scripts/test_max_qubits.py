#!/usr/bin/env python3
"""
MAX QUBITS TEST: Find the maximum number of qubits for each circuit type.

Key findings:
- Sparse states (GHZ, W) can scale to MANY qubits
- Dense states (Hadamard wall) are limited by exponential growth

Results will show:
- GHZ: Constant sparsity (2 amplitudes) → can simulate 100+ qubits
- W State: Linear sparsity (n amplitudes) → can simulate 100+ qubits
- Hadamard Wall: Exponential (2^n amplitudes) → limited to ~20 qubits on laptop
"""
import sys
from pathlib import Path
import numpy as np
import tempfile
import shutil
import uuid
import time
import gc

V3_SRC = Path(__file__).parent.parent / "src"
if str(V3_SRC) not in sys.path:
    sys.path.insert(0, str(V3_SRC))

from driver import SparkHiSVSIMDriver
from v2_common.config import SimulatorConfig
from v2_common.circuits import generate_ghz_circuit, generate_w_circuit, generate_hadamard_wall


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


def format_time(seconds):
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}min"
    else:
        return f"{seconds/3600:.1f}h"


def test_circuit(circuit_gen, name, n_qubits, temp_dir, timeout=120):
    """Test a single circuit and return results."""
    try:
        config = SimulatorConfig(
            run_id=f"max_{uuid.uuid4().hex[:8]}",
            base_path=temp_dir,
            batch_size=100,
            spark_master="local[*]",
            spark_shuffle_partitions=8,
            checkpoint_every_n_batches=100,  # Reduce checkpointing overhead
        )
        config.ensure_paths()
        
        circuit = circuit_gen(n_qubits)
        
        start = time.time()
        
        with SparkHiSVSIMDriver(config) as driver:
            result = driver.run_circuit(circuit, resume=False)
            state_dict = driver.state_manager.get_state_as_dict(result.final_state_df)
            non_zero = len(state_dict)
            
            # Verify normalization
            norm = np.sqrt(sum(abs(v)**2 for v in state_dict.values()))
        
        elapsed = time.time() - start
        
        if elapsed > timeout:
            return None, "TIMEOUT"
        
        return {
            "n_qubits": n_qubits,
            "non_zero": non_zero,
            "norm": norm,
            "time": elapsed,
            "parallel_groups": result.parallel_groups,
            "storage": non_zero * 16,  # 16 bytes per complex
        }, "OK"
        
    except Exception as e:
        return None, str(e)[:50]


def test_sparse_circuit(circuit_gen, name, max_qubits=100, step=10, temp_dir=None):
    """Test a sparse circuit scaling."""
    print(f"\n{'='*70}")
    print(f"Testing {name} (SPARSE - should scale to many qubits)")
    print(f"{'='*70}")
    
    own_temp = temp_dir is None
    if own_temp:
        temp_dir = Path(tempfile.mkdtemp())
    
    results = []
    last_success = 0
    
    try:
        # Test small sizes first
        for n in [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
            if n > max_qubits:
                break
            
            print(f"  Testing {name}-{n}...", end=" ", flush=True)
            result, status = test_circuit(circuit_gen, name, n, temp_dir)
            
            if result:
                last_success = n
                results.append(result)
                print(f"✓ {result['non_zero']} amplitudes, {format_time(result['time'])}")
            else:
                print(f"✗ {status}")
                if "memory" in status.lower() or "oom" in status.lower():
                    break
            
            # Force garbage collection
            gc.collect()
    
    finally:
        if own_temp:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    return results, last_success


def test_dense_circuit(circuit_gen, name, max_qubits=20, temp_dir=None):
    """Test a dense circuit scaling."""
    print(f"\n{'='*70}")
    print(f"Testing {name} (DENSE - limited by exponential growth)")
    print(f"{'='*70}")
    
    own_temp = temp_dir is None
    if own_temp:
        temp_dir = Path(tempfile.mkdtemp())
    
    results = []
    last_success = 0
    
    try:
        for n in range(5, max_qubits + 1):
            print(f"  Testing {name}-{n}...", end=" ", flush=True)
            
            # Estimate memory needed
            estimated_storage = (2**n) * 16
            print(f"(est. {format_bytes(estimated_storage)}) ", end="", flush=True)
            
            result, status = test_circuit(circuit_gen, name, n, temp_dir, timeout=60)
            
            if result:
                last_success = n
                results.append(result)
                print(f"✓ {result['non_zero']:,} amplitudes, {format_time(result['time'])}")
            else:
                print(f"✗ {status}")
                break
            
            gc.collect()
    
    finally:
        if own_temp:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    return results, last_success


def main():
    print("=" * 70)
    print("MAX QUBITS TEST: Finding the scaling limits")
    print("=" * 70)
    print("\nKey insight:")
    print("- Sparse states can scale to MANY qubits")
    print("- Dense states are limited by exponential memory growth")
    
    temp_dir = Path(tempfile.mkdtemp())
    
    try:
        # Test GHZ (constant sparsity - 2 amplitudes)
        ghz_results, ghz_max = test_sparse_circuit(
            generate_ghz_circuit, "GHZ", max_qubits=100, temp_dir=temp_dir
        )
        
        # Test W State (linear sparsity - n amplitudes)
        w_results, w_max = test_sparse_circuit(
            generate_w_circuit, "W State", max_qubits=50, temp_dir=temp_dir
        )
        
        # Test Hadamard Wall (exponential - 2^n amplitudes)
        h_results, h_max = test_dense_circuit(
            generate_hadamard_wall, "Hadamard Wall", max_qubits=22, temp_dir=temp_dir
        )
        
        # Summary
        print("\n" + "=" * 70)
        print("SUMMARY: Maximum Qubits Achieved")
        print("=" * 70)
        
        print(f"\n{'Circuit':<20} {'Max Qubits':<12} {'Amplitudes':<15} {'Storage':<12} {'Scaling':<15}")
        print("-" * 70)
        
        if ghz_results:
            last = ghz_results[-1]
            print(f"{'GHZ State':<20} {ghz_max:<12} {last['non_zero']:<15} "
                  f"{format_bytes(last['storage']):<12} {'O(1) - constant':<15}")
        
        if w_results:
            last = w_results[-1]
            print(f"{'W State':<20} {w_max:<12} {last['non_zero']:<15} "
                  f"{format_bytes(last['storage']):<12} {'O(n) - linear':<15}")
        
        if h_results:
            last = h_results[-1]
            print(f"{'Hadamard Wall':<20} {h_max:<12} {last['non_zero']:<15,} "
                  f"{format_bytes(last['storage']):<12} {'O(2^n) - exponential':<15}")
        
        # Theoretical limits
        print("\n" + "-" * 70)
        print("THEORETICAL LIMITS (with 16GB RAM):")
        print("-" * 70)
        print(f"  GHZ State:     UNLIMITED (always 2 amplitudes)")
        print(f"  W State:       ~1 billion qubits (n amplitudes)")
        print(f"  Hadamard Wall: ~30 qubits (2^30 = 1B amplitudes = 16GB)")
        
        # Key insight
        print("\n" + "=" * 70)
        print("KEY INSIGHT")
        print("=" * 70)
        print(f"""
The NUMBER OF NON-ZERO AMPLITUDES determines scalability, not qubit count!

Measured Results:
  • GHZ-{ghz_max}: {ghz_results[-1]['non_zero'] if ghz_results else 'N/A'} amplitudes (always 2!)
  • W-{w_max}: {w_results[-1]['non_zero'] if w_results else 'N/A'} amplitudes (= n)
  • H-Wall-{h_max}: {h_results[-1]['non_zero'] if h_results else 'N/A':,} amplitudes (= 2^n)

The G gate creates W states with LINEAR sparsity, allowing simulation of
many more qubits than dense states like Hadamard walls.

Parallelization helps with SPEED but doesn't change MEMORY limits.
Dense states will always hit the 2^n memory wall.
""")
        
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
