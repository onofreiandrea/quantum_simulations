"""
MAX Test with Clean Memory - Fresh driver for each qubit count
Tests with 15 minute timeout to push limits further
"""
import sys
from pathlib import Path
import time
import numpy as np
import gc

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from v2_common.config import SimulatorConfig
from driver import SparkHiSVSIMDriver


def generate_ry(n_qubits):
    """RY gates: Creates dense states directly."""
    gates = []
    for q in range(n_qubits):
        gates.append({"gate": "RY", "qubits": [q], "params": {"theta": np.pi / 4}})
    return {"number_of_qubits": n_qubits, "gates": gates}


def generate_h_t(n_qubits):
    """H+T gates: Superposition + phase."""
    gates = []
    for q in range(n_qubits):
        gates.append({"gate": "H", "qubits": [q]})
        gates.append({"gate": "T", "qubits": [q]})
    return {"number_of_qubits": n_qubits, "gates": gates}


def generate_h_t_cr(n_qubits):
    """H+T+CR gates: Superposition + phase + entanglement."""
    gates = []
    for q in range(n_qubits):
        gates.append({"gate": "H", "qubits": [q]})
    for q in range(n_qubits):
        gates.append({"gate": "T", "qubits": [q]})
    for q in range(n_qubits - 1):
        gates.append({"gate": "CR", "qubits": [q, q+1], "params": {"k": 2}})
    return {"number_of_qubits": n_qubits, "gates": gates}


def generate_g(n_qubits):
    """G gates: General unitary gates."""
    gates = []
    for q in range(n_qubits):
        gates.append({"gate": "H", "qubits": [q]})
    for q in range(n_qubits):
        gates.append({"gate": "G", "qubits": [q], "params": {"p": 3}})
    return {"number_of_qubits": n_qubits, "gates": gates}


def generate_r(n_qubits):
    """R gates: Phase rotation gates."""
    gates = []
    for q in range(n_qubits):
        gates.append({"gate": "H", "qubits": [q]})
    for q in range(n_qubits):
        gates.append({"gate": "R", "qubits": [q], "params": {"k": 3}})
    return {"number_of_qubits": n_qubits, "gates": gates}


def generate_cu(n_qubits):
    """CU gates: Controlled unitaries."""
    gates = []
    for q in range(n_qubits):
        gates.append({"gate": "H", "qubits": [q]})
    U = np.array([[0.8, 0.6], [-0.6, 0.8]], dtype=complex)
    for q in range(n_qubits - 1):
        gates.append({"gate": "CU", "qubits": [q, q+1], "params": {"U": U, "exponent": 1}})
    return {"number_of_qubits": n_qubits, "gates": gates}


def test_circuit_type_clean_memory(name, generator, start=26, max_q=30, timeout=900):
    """
    Test with CLEAN MEMORY - fresh driver for each qubit count.
    This ensures no memory leaks or accumulated state.
    
    Returns:
        Tuple of (max_success_qubits, detailed_results_dict)
    """
    print(f"\n{'='*70}")
    print(f"Testing: {name} (CLEAN MEMORY - fresh driver per qubit)")
    print(f"{'='*70}")
    print(f"Timeout: {timeout}s (15 minutes)")
    print(f"Range: {start}-{max_q} qubits")
    
    max_success = None
    detailed_results = {}
    
    for n_qubits in range(start, max_q + 1):
        print(f"\n  {n_qubits} qubits... ", end="", flush=True)
        
        # Create fresh config for each test
        config = SimulatorConfig(
            base_path=Path(f"data/clean_test_{name.replace(' ', '_')}_{n_qubits}"),
            spark_master="local[*]",
            spark_shuffle_partitions=200,
            batch_size=10,
        )
        config.ensure_paths()
        
        driver = None
        start_time = time.time()
        
        try:
            circuit = generator(n_qubits)
            
            # Create FRESH driver for each qubit count
            driver = SparkHiSVSIMDriver(config, enable_parallel=True)
            
            result = driver.run_circuit(circuit, enable_parallel=True, resume=False)
            elapsed = time.time() - start_time
            state_size = result.final_state_df.count()
            theoretical = 2 ** n_qubits
            sparsity = (1 - state_size / theoretical) * 100
            
            print(f"✅ SUCCESS ({elapsed:.1f}s, {state_size:,} amplitudes, {sparsity:.1f}% sparse)")
            max_success = n_qubits
            detailed_results[n_qubits] = {
                "status": "SUCCESS",
                "elapsed_seconds": elapsed,
                "amplitudes": state_size,
                "theoretical": theoretical,
                "sparsity_percent": sparsity
            }
            
            # Clean up driver IMMEDIATELY
            driver.cleanup()
            if hasattr(driver, 'spark'):
                driver.spark.stop()
            driver = None
            
            # Force garbage collection
            gc.collect()
            
            if elapsed > timeout:
                print(f"  ⏱️  Timeout ({elapsed:.1f}s > {timeout}s)")
                break
                
        except Exception as e:
            elapsed = time.time() - start_time if start_time else 0
            error = str(e)[:80]
            print(f"❌ FAILED ({elapsed:.1f}s): {error}")
            
            error_type = "UNKNOWN"
            if "memory" in error.lower() or "oom" in error.lower() or "heap" in error.lower():
                error_type = "MEMORY"
            elif elapsed > timeout:
                error_type = "TIMEOUT"
            else:
                error_type = "OTHER"
            
            detailed_results[n_qubits] = {
                "status": "FAILED",
                "elapsed_seconds": elapsed,
                "error": error,
                "error_type": error_type
            }
            
            # Clean up on error
            if driver:
                try:
                    driver.cleanup()
                    if hasattr(driver, 'spark'):
                        driver.spark.stop()
                except:
                    pass
                driver = None
            
            gc.collect()
            
            if error_type == "MEMORY":
                print(f"  💥 MEMORY LIMIT: {max_success} qubits")
                break
            elif error_type == "TIMEOUT":
                print(f"  ⏱️  TIMEOUT LIMIT: {max_success} qubits")
                break
            else:
                # Other error - might be transient, try next
                print(f"  ⚠️  Error (not memory/timeout), continuing...")
                continue
    
    print(f"\n  🎯 MAX for {name} (clean memory): {max_success} qubits")
    return max_success, detailed_results


if __name__ == "__main__":
    import json
    from datetime import datetime
    
    results_file = Path("data/clean_memory_test_results.json")
    results_file.parent.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("NON-STABILIZER MAX TEST - CLEAN MEMORY APPROACH")
    print("="*70)
    print("Strategy: Fresh driver for each qubit count")
    print("Timeout: 15 minutes (900s) per test")
    print(f"Results will be saved to: {results_file}")
    print("="*70)
    
    results = {}
    detailed_results = {}
    
    # Test ALL non-stabilizer gate types comprehensively
    print("\n" + "="*70)
    print("COMPREHENSIVE TEST: All Non-Stabilizer Gate Types")
    print("="*70)
    print("Strategy: Clean memory (fresh driver per qubit)")
    print("Timeout: 15 minutes (900s) per test")
    print("="*70)
    
    # Test RY gates (we know 30 works, start from there)
    print("\n" + "="*70)
    print("1/6: RY gates")
    print("="*70)
    max_ry, details_ry = test_circuit_type_clean_memory("RY gates", generate_ry, start=30, max_q=32, timeout=900)
    results["RY"] = max_ry
    detailed_results["RY"] = details_ry
    
    # Test H+T gates
    print("\n" + "="*70)
    print("2/6: H+T gates")
    print("="*70)
    max_ht, details_ht = test_circuit_type_clean_memory("H+T gates", generate_h_t, start=25, max_q=30, timeout=900)
    results["H+T"] = max_ht
    detailed_results["H+T"] = details_ht
    
    # Test H+T+CR gates
    print("\n" + "="*70)
    print("3/6: H+T+CR gates")
    print("="*70)
    max_htcr, details_htcr = test_circuit_type_clean_memory("H+T+CR gates", generate_h_t_cr, start=25, max_q=30, timeout=900)
    results["H+T+CR"] = max_htcr
    detailed_results["H+T+CR"] = details_htcr
    
    # Test G gates
    print("\n" + "="*70)
    print("4/6: G gates")
    print("="*70)
    max_g, details_g = test_circuit_type_clean_memory("G gates", generate_g, start=25, max_q=30, timeout=900)
    results["G"] = max_g
    detailed_results["G"] = details_g
    
    # Test R gates
    print("\n" + "="*70)
    print("5/6: R gates")
    print("="*70)
    max_r, details_r = test_circuit_type_clean_memory("R gates", generate_r, start=25, max_q=30, timeout=900)
    results["R"] = max_r
    detailed_results["R"] = details_r
    
    # Test CU gates
    print("\n" + "="*70)
    print("6/6: CU gates")
    print("="*70)
    max_cu, details_cu = test_circuit_type_clean_memory("CU gates", generate_cu, start=25, max_q=30, timeout=900)
    results["CU"] = max_cu
    detailed_results["CU"] = details_cu
    
    print("\n" + "="*70)
    print("FINAL RESULTS (CLEAN MEMORY)")
    print("="*70)
    for name, max_q in results.items():
        if max_q:
            print(f"{name}: {max_q} qubits")
        else:
            print(f"{name}: Failed")
    
    overall_max = max([q for q in results.values() if q], default=0)
    print(f"\n🎯 OVERALL MAX (clean memory): {overall_max} qubits")
    
    if overall_max > 26:
        print(f"\n🎉 IMPROVEMENT: {overall_max} qubits > previous 26 qubits!")
    
    # Save results to JSON file
    output = {
        "timestamp": datetime.now().isoformat(),
        "strategy": "clean_memory_fresh_driver_per_qubit",
        "timeout_seconds": 900,
        "results": results,
        "detailed_results": detailed_results,
        "overall_max": overall_max,
        "improvement": overall_max > 26
    }
    
    with open(results_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✅ Results saved to: {results_file}")
