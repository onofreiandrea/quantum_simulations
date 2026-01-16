"""
COMPREHENSIVE MAX TEST - All Non-Stabilizer Gate Types
Clean Memory Approach: Fresh driver for each qubit count
"""
import sys
from pathlib import Path
import time
import numpy as np
import gc
import json
from datetime import datetime

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


def test_circuit_type_clean_memory(name, generator, start=25, max_q=30, timeout=900):
    """
    Test with CLEAN MEMORY - fresh driver for each qubit count.
    Returns: (max_success_qubits, detailed_results_dict)
    """
    print(f"\n{'='*70}")
    print(f"Testing: {name} (CLEAN MEMORY)")
    print(f"{'='*70}")
    print(f"Timeout: {timeout}s (15 minutes)")
    print(f"Range: {start}-{max_q} qubits")
    
    max_success = None
    detailed_results = {}
    
    for n_qubits in range(start, max_q + 1):
        print(f"\n  {n_qubits} qubits... ", end="", flush=True)
        
        config = SimulatorConfig(
            base_path=Path(f"data/comprehensive_test_{name.replace(' ', '_').replace('+', '_')}_{n_qubits}"),
            spark_master="local[*]",
            spark_shuffle_partitions=200,
            batch_size=10,
        )
        config.ensure_paths()
        
        driver = None
        start_time = time.time()
        
        try:
            circuit = generator(n_qubits)
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
            
            # Clean up driver properly
            try:
                driver.cleanup()
            except:
                pass
            
            # Stop Spark context completely
            try:
                if hasattr(driver, 'spark'):
                    driver.spark.stop()
                    driver.spark = None
            except:
                pass
            
            driver = None
            
            # Force garbage collection and wait a bit for cleanup
            gc.collect()
            time.sleep(2)  # Give Spark time to fully shut down
            
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
            
            if driver:
                try:
                    driver.cleanup()
                except:
                    pass
                
                try:
                    if hasattr(driver, 'spark'):
                        driver.spark.stop()
                        driver.spark = None
                except:
                    pass
                
                driver = None
            
            # Force garbage collection and wait for cleanup
            gc.collect()
            time.sleep(2)  # Give Spark time to fully shut down
            
            if error_type == "MEMORY":
                print(f"  💥 MEMORY LIMIT: {max_success} qubits")
                break
            elif error_type == "TIMEOUT":
                print(f"  ⏱️  TIMEOUT LIMIT: {max_success} qubits")
                break
    
    print(f"\n  🎯 MAX for {name}: {max_success} qubits")
    return max_success, detailed_results


if __name__ == "__main__":
    results_file = Path("data/comprehensive_non_stabilizer_max_results.json")
    results_file.parent.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("COMPREHENSIVE NON-STABILIZER MAX TEST")
    print("="*70)
    print("Strategy: Clean memory (fresh driver per qubit)")
    print("Timeout: 15 minutes (900s) per test")
    print(f"Results will be saved to: {results_file}")
    print("="*70)
    
    results = {}
    detailed_results = {}
    
    # Test ALL gate types
    gate_types = [
        ("RY gates", generate_ry, 30, 32),  # We know 30 works, try 31-32
        ("H+T gates", generate_h_t, 25, 30),
        ("H+T+CR gates", generate_h_t_cr, 25, 30),
        ("G gates", generate_g, 25, 30),
        ("R gates", generate_r, 25, 30),
        ("CU gates", generate_cu, 25, 30),
    ]
    
    for i, (name, generator, start, max_q) in enumerate(gate_types, 1):
        print(f"\n{'='*70}")
        print(f"{i}/{len(gate_types)}: {name}")
        print("="*70)
        
        # Ensure Spark is fully stopped before starting new test
        try:
            from pyspark import SparkContext
            sc = SparkContext._active_spark_context
            if sc:
                print("  Stopping existing Spark context...")
                sc.stop()
                time.sleep(3)  # Wait for Spark to fully shut down
        except Exception as e:
            print(f"  No existing Spark context (or error stopping): {e}")
        
        # Force garbage collection
        gc.collect()
        time.sleep(1)
        
        max_q_result, details = test_circuit_type_clean_memory(name, generator, start=start, max_q=max_q, timeout=900)
        results[name] = max_q_result
        detailed_results[name] = details
        
        # Clean up after each gate type
        try:
            from pyspark import SparkContext
            sc = SparkContext._active_spark_context
            if sc:
                sc.stop()
                time.sleep(2)
        except:
            pass
        gc.collect()
        time.sleep(1)
    
    print("\n" + "="*70)
    print("FINAL RESULTS - ALL NON-STABILIZER GATE TYPES")
    print("="*70)
    for name, max_q in results.items():
        if max_q:
            print(f"{name}: {max_q} qubits")
        else:
            print(f"{name}: Failed")
    
    overall_max = max([q for q in results.values() if q], default=0)
    print(f"\n🎯 OVERALL MAX: {overall_max} qubits")
    
    # Save results
    output = {
        "timestamp": datetime.now().isoformat(),
        "strategy": "clean_memory_fresh_driver_per_qubit",
        "timeout_seconds": 900,
        "results": results,
        "detailed_results": detailed_results,
        "overall_max": overall_max,
    }
    
    with open(results_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n✅ Results saved to: {results_file}")
