#!/usr/bin/env python3
"""
Scalability test for Spark quantum simulator.

Tests how many qubits can be simulated for different circuit types:
1. GHZ (sparse) - stays sparse, should scale well
2. QFT (dense) - becomes fully dense, limited by memory
3. Random circuits - varies
"""
from __future__ import annotations

import sys
import time
import traceback
from pathlib import Path
import tempfile
import shutil

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import SimulatorConfig
from src.driver import SparkQuantumDriver
from src.circuits import generate_ghz_circuit, generate_qft_circuit


def format_size(n_amplitudes: int) -> str:
    """Format number of amplitudes as memory size."""
    bytes_per_amp = 16  # 8 bytes real + 8 bytes imag
    total_bytes = n_amplitudes * bytes_per_amp
    
    if total_bytes < 1024:
        return f"{total_bytes} B"
    elif total_bytes < 1024**2:
        return f"{total_bytes/1024:.1f} KB"
    elif total_bytes < 1024**3:
        return f"{total_bytes/1024**2:.1f} MB"
    elif total_bytes < 1024**4:
        return f"{total_bytes/1024**3:.1f} GB"
    else:
        return f"{total_bytes/1024**4:.1f} TB"


def test_ghz_scalability():
    """Test GHZ circuit scalability (sparse state)."""
    print("\n" + "="*70)
    print("GHZ CIRCUIT SCALABILITY (Sparse State)")
    print("="*70)
    print("GHZ produces only 2 non-zero amplitudes regardless of qubit count")
    print("-"*70)
    print(f"{'Qubits':>8} | {'Gates':>8} | {'Time (s)':>10} | {'Dense Size':>12} | {'Status':>10}")
    print("-"*70)
    
    results = []
    
    for n_qubits in [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]:
        temp_dir = Path(tempfile.mkdtemp())
        
        try:
            config = SimulatorConfig(
                base_path=temp_dir,
                batch_size=n_qubits,  # One batch for all gates
                spark_master="local[*]",
                spark_driver_memory="4g",
            )
            
            circuit = generate_ghz_circuit(n_qubits)
            dense_size = format_size(2**n_qubits)
            
            start = time.time()
            with SparkQuantumDriver(config) as driver:
                result = driver.run_circuit(circuit, resume=False)
                state_dict = driver.get_state_dict(result)
            elapsed = time.time() - start
            
            # Verify correctness
            assert len(state_dict) == 2, f"Expected 2 amplitudes, got {len(state_dict)}"
            assert 0 in state_dict, "|0...0⟩ missing"
            assert (2**n_qubits - 1) in state_dict, "|1...1⟩ missing"
            
            status = "✓ PASS"
            results.append((n_qubits, True, elapsed))
            
            print(f"{n_qubits:>8} | {len(circuit['gates']):>8} | {elapsed:>10.2f} | {dense_size:>12} | {status:>10}")
            
        except Exception as e:
            status = "✗ FAIL"
            results.append((n_qubits, False, 0))
            print(f"{n_qubits:>8} | {n_qubits:>8} | {'N/A':>10} | {format_size(2**n_qubits):>12} | {status:>10}")
            print(f"  Error: {type(e).__name__}: {str(e)[:60]}")
            if n_qubits <= 30:
                traceback.print_exc()
            break
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    return results


def test_qft_scalability():
    """Test QFT circuit scalability (dense state)."""
    print("\n" + "="*70)
    print("QFT CIRCUIT SCALABILITY (Dense State)")
    print("="*70)
    print("QFT produces 2^n non-zero amplitudes - memory intensive!")
    print("-"*70)
    print(f"{'Qubits':>8} | {'Gates':>8} | {'Time (s)':>10} | {'State Size':>12} | {'Status':>10}")
    print("-"*70)
    
    results = []
    
    for n_qubits in [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]:
        temp_dir = Path(tempfile.mkdtemp())
        
        try:
            config = SimulatorConfig(
                base_path=temp_dir,
                batch_size=20,
                spark_master="local[*]",
                spark_driver_memory="4g",
            )
            
            circuit = generate_qft_circuit(n_qubits)
            n_gates = len(circuit['gates'])
            state_size = format_size(2**n_qubits)
            
            start = time.time()
            with SparkQuantumDriver(config) as driver:
                result = driver.run_circuit(circuit, resume=False)
                # Just count non-zero amplitudes, don't collect all
                count = result.final_state_df.count()
            elapsed = time.time() - start
            
            status = "✓ PASS"
            results.append((n_qubits, True, elapsed))
            
            print(f"{n_qubits:>8} | {n_gates:>8} | {elapsed:>10.2f} | {state_size:>12} | {status:>10}")
            
        except Exception as e:
            status = "✗ FAIL"
            results.append((n_qubits, False, 0))
            print(f"{n_qubits:>8} | {'?':>8} | {'N/A':>10} | {format_size(2**n_qubits):>12} | {status:>10}")
            print(f"  Error: {type(e).__name__}: {str(e)[:60]}")
            break
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
    
    return results


def test_hadamard_wall():
    """Test Hadamard on all qubits (maximally dense)."""
    print("\n" + "="*70)
    print("HADAMARD WALL (Maximally Dense)")
    print("="*70)
    print("H on all qubits creates uniform superposition of ALL 2^n states")
    print("-"*70)
    print(f"{'Qubits':>8} | {'Amplitudes':>12} | {'Time (s)':>10} | {'State Size':>12} | {'Status':>10}")
    print("-"*70)
    
    for n_qubits in [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]:
        temp_dir = Path(tempfile.mkdtemp())
        
        try:
            config = SimulatorConfig(
                base_path=temp_dir,
                batch_size=n_qubits,
                spark_master="local[*]",
                spark_driver_memory="4g",
            )
            
            # H on all qubits
            circuit = {
                "number_of_qubits": n_qubits,
                "gates": [{"qubits": [i], "gate": "H"} for i in range(n_qubits)]
            }
            
            state_size = format_size(2**n_qubits)
            
            start = time.time()
            with SparkQuantumDriver(config) as driver:
                result = driver.run_circuit(circuit, resume=False)
                count = result.final_state_df.count()
            elapsed = time.time() - start
            
            status = "✓ PASS"
            print(f"{n_qubits:>8} | {count:>12,} | {elapsed:>10.2f} | {state_size:>12} | {status:>10}")
            
        except Exception as e:
            status = "✗ FAIL"
            print(f"{n_qubits:>8} | {2**n_qubits:>12,} | {'N/A':>10} | {format_size(2**n_qubits):>12} | {status:>10}")
            print(f"  Error: {type(e).__name__}: {str(e)[:60]}")
            break
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)


def main():
    print("="*70)
    print("SPARK QUANTUM SIMULATOR - SCALABILITY TEST")
    print("="*70)
    print(f"Testing on local machine with Spark local[*] mode")
    print(f"Driver memory: 4GB")
    print()
    print("Memory requirements (dense state):")
    print("  10 qubits: 16 KB")
    print("  20 qubits: 16 MB")
    print("  30 qubits: 16 GB")
    print("  40 qubits: 16 TB")
    print("  50 qubits: 16 PB (!)")
    
    # Test sparse circuits first (GHZ)
    ghz_results = test_ghz_scalability()
    
    # Test dense circuits (QFT)
    qft_results = test_qft_scalability()
    
    # Test maximally dense (Hadamard wall)
    test_hadamard_wall()
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    max_ghz = max(n for n, success, _ in ghz_results if success)
    max_qft = max((n for n, success, _ in qft_results if success), default=0)
    
    print(f"\nGHZ (sparse): Successfully simulated up to {max_ghz} qubits")
    print(f"QFT (dense):  Successfully simulated up to {max_qft} qubits")
    
    print("\n" + "-"*70)
    print("ANALYSIS:")
    print("-"*70)
    print("""
For SPARSE circuits (like GHZ):
  - Can simulate 50+ qubits easily because only 2 amplitudes are stored
  - Limited by gate application time, not memory
  - Spark's distributed processing helps with larger circuits

For DENSE circuits (like QFT, Hadamard wall):
  - Limited by memory: 2^n amplitudes × 16 bytes each
  - ~15-18 qubits is typical limit on a single machine with 4GB
  - To reach 50 qubits would need ~16 PETABYTES of memory!
  
To simulate 50 qubits non-sparse:
  - Would need a massive Spark cluster (thousands of nodes)
  - OR use tensor network / MPS representations (different approach)
  - OR use stabilizer formalism (only for Clifford circuits)
""")


if __name__ == "__main__":
    main()
