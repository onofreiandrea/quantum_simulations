"""Test beyond 30 qubits - RY gates"""
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


def test_qubits(start=31, max_q=35, timeout=1800):
    """Test RY gates from start to max_q qubits"""
    print(f"\n{'='*70}")
    print(f"Testing RY gates: {start}-{max_q} qubits")
    print(f"Timeout: {timeout}s (30 minutes)")
    print(f"{'='*70}")
    
    max_success = None
    
    for n_qubits in range(start, max_q + 1):
        print(f"\n{n_qubits} qubits... ", end="", flush=True)
        
        config = SimulatorConfig(
            base_path=Path(f"data/test_beyond_30_{n_qubits}"),
            spark_master="local[*]",
            spark_shuffle_partitions=200,
            batch_size=10,
        )
        config.ensure_paths()
        
        driver = None
        start_time = time.time()
        
        try:
            circuit = generate_ry(n_qubits)
            driver = SparkHiSVSIMDriver(config, enable_parallel=True)
            
            result = driver.run_circuit(circuit, enable_parallel=True, resume=False)
            elapsed = time.time() - start_time
            state_size = result.final_state_df.count()
            theoretical = 2 ** n_qubits
            sparsity = (1 - state_size / theoretical) * 100
            
            print(f"✅ SUCCESS ({elapsed:.1f}s, {state_size:,} amplitudes, {sparsity:.1f}% sparse)")
            max_success = n_qubits
            
            driver.cleanup()
            if hasattr(driver, 'spark'):
                driver.spark.stop()
            driver = None
            gc.collect()
            time.sleep(2)
            
            if elapsed > timeout:
                print(f"  ⏱️  Timeout ({elapsed:.1f}s > {timeout}s)")
                break
                
        except Exception as e:
            elapsed = time.time() - start_time if start_time else 0
            error = str(e)[:100]
            print(f"❌ FAILED ({elapsed:.1f}s): {error}")
            
            if driver:
                try:
                    driver.cleanup()
                    if hasattr(driver, 'spark'):
                        driver.spark.stop()
                except:
                    pass
                driver = None
            
            gc.collect()
            time.sleep(2)
            
            if "memory" in error.lower() or "oom" in error.lower() or "heap" in error.lower():
                print(f"  💥 MEMORY LIMIT: {max_success} qubits")
                break
            elif elapsed > timeout:
                print(f"  ⏱️  TIMEOUT LIMIT: {max_success} qubits")
                break
    
    print(f"\n🎯 MAX SUCCESS: {max_success} qubits")
    return max_success


if __name__ == "__main__":
    print("="*70)
    print("TESTING BEYOND 30 QUBITS")
    print("="*70)
    
    max_q = test_qubits(start=31, max_q=35, timeout=1800)
    
    if max_q and max_q > 30:
        print(f"\n🎉 BREAKTHROUGH: {max_q} qubits > 30 qubits!")
    elif max_q == 30:
        print(f"\n✅ Confirmed: 30 qubits still max")
    else:
        print(f"\n⚠️  Could not exceed 30 qubits")
