"""
Scalability Test: Find practical qubit limits for quantum simulation.

Tests:
1. Sparse states (GHZ) - should scale to many qubits
2. Dense states (QFT) - limited by memory
3. Non-stabilizer circuits (T, CR gates) - limited by memory

Reports:
- Maximum qubits for each circuit type
- Memory usage
- Execution time
- Failure points and reasons
"""
import sys
from pathlib import Path
import time
import traceback
from typing import Dict, List, Tuple, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from dataclasses import dataclass

from v2_common.config import SimulatorConfig
from driver import SparkHiSVSIMDriver
from v2_common.circuits import (
    generate_ghz_circuit,
    generate_qft_circuit,
    generate_w_circuit,
)
from v2_common.frontend import circuit_dict_to_gates


@dataclass
class ScalabilityResult:
    """Result of a scalability test."""
    circuit_type: str
    n_qubits: int
    success: bool
    execution_time: float
    memory_usage_mb: Optional[float] = None
    state_size: Optional[int] = None
    error_message: Optional[str] = None


def get_memory_usage() -> float:
    """Get current memory usage in MB."""
    try:
        import psutil
        import os
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except ImportError:
        return None


def test_circuit_scalability(
    driver: SparkHiSVSIMDriver,
    circuit_generator,
    circuit_name: str,
    start_qubits: int = 5,
    max_qubits: int = 50,
    step: int = 1,
    timeout_seconds: int = 300,
) -> List[ScalabilityResult]:
    """
    Test scalability of a circuit type by incrementing qubit count.
    
    Args:
        driver: SparkHiSVSIMDriver instance
        circuit_generator: Function that generates circuit dict (n_qubits) -> dict
        circuit_name: Name of circuit type
        start_qubits: Starting number of qubits
        max_qubits: Maximum qubits to test
        step: Increment step
        timeout_seconds: Timeout per test
        
    Returns:
        List of ScalabilityResult objects
    """
    results = []
    
    print(f"\n{'='*70}")
    print(f"Testing {circuit_name} Circuit Scalability")
    print(f"{'='*70}")
    print(f"Start: {start_qubits} qubits, Max: {max_qubits} qubits, Step: {step}")
    print()
    
    for n_qubits in range(start_qubits, max_qubits + 1, step):
        print(f"Testing {circuit_name} with {n_qubits} qubits...", end=" ", flush=True)
        
        # Calculate theoretical state size
        theoretical_size = 2 ** n_qubits
        theoretical_memory_mb = theoretical_size * 16 / 1024 / 1024  # 16 bytes per amplitude
        
        start_time = time.time()
        start_memory = get_memory_usage()
        
        try:
            # Generate circuit
            circuit_dict = circuit_generator(n_qubits)
            n_gates = len(circuit_dict["gates"])
            
            # Run simulation with timeout
            result = driver.run_circuit(
                circuit_dict,
                enable_parallel=True,
                resume=False
            )
            
            execution_time = time.time() - start_time
            end_memory = get_memory_usage()
            
            # Get actual state size
            state_df = result.final_state_df
            actual_state_size = state_df.count()
            
            memory_used = (end_memory - start_memory) if (start_memory and end_memory) else None
            
            success = True
            error_msg = None
            
            print(f"✅ SUCCESS ({execution_time:.2f}s, {actual_state_size} amplitudes)")
            
            results.append(ScalabilityResult(
                circuit_type=circuit_name,
                n_qubits=n_qubits,
                success=True,
                execution_time=execution_time,
                memory_usage_mb=memory_used,
                state_size=actual_state_size,
            ))
            
            # Check if we're hitting memory limits
            if theoretical_memory_mb > 100_000:  # > 100 GB theoretical
                print(f"  ⚠️  Theoretical memory: {theoretical_memory_mb:.1f} GB")
                print(f"  ⚠️  Actual state size: {actual_state_size} amplitudes")
            
            # Stop if execution time is too long
            if execution_time > timeout_seconds:
                print(f"  ⚠️  Execution time exceeded {timeout_seconds}s, stopping")
                break
                
        except MemoryError as e:
            execution_time = time.time() - start_time
            print(f"❌ MEMORY ERROR ({execution_time:.2f}s)")
            print(f"   Error: {str(e)}")
            
            results.append(ScalabilityResult(
                circuit_type=circuit_name,
                n_qubits=n_qubits,
                success=False,
                execution_time=execution_time,
                error_message=f"MemoryError: {str(e)}",
            ))
            break
            
        except Exception as e:
            execution_time = time.time() - start_time
            print(f"❌ ERROR ({execution_time:.2f}s)")
            print(f"   Error: {str(e)}")
            
            # Print full traceback for debugging
            if "memory" in str(e).lower() or "oom" in str(e).lower():
                print(f"   ⚠️  Likely memory-related failure")
            
            results.append(ScalabilityResult(
                circuit_type=circuit_name,
                n_qubits=n_qubits,
                success=False,
                execution_time=execution_time,
                error_message=f"{type(e).__name__}: {str(e)}",
            ))
            
            # Stop on critical errors
            if "memory" in str(e).lower() or "oom" in str(e).lower():
                break
    
    return results


def test_non_stabilizer_circuit(
    driver: SparkHiSVSIMDriver,
    n_qubits: int,
) -> Tuple[bool, Optional[str], float]:
    """
    Test a non-stabilizer circuit (T gates, CR gates).
    
    Creates a circuit with T gates on all qubits (creates dense state).
    """
    # Create a circuit with T gates (non-stabilizer)
    # T gate creates superposition, so this will be dense
    gates = []
    for q in range(n_qubits):
        gates.append({"gate": "T", "qubits": [q]})
    
    # Add some CR gates for more complexity
    for q in range(n_qubits - 1):
        gates.append({"gate": "CR", "qubits": [q, q+1], "params": {"k": 2}})
    
    circuit_dict = {
        "number_of_qubits": n_qubits,
        "gates": gates
    }
    
    try:
        start_time = time.time()
        result = driver.run_circuit(circuit_dict, enable_parallel=True, resume=False)
        execution_time = time.time() - start_time
        
        state_size = result.final_state_df.count()
        return True, None, execution_time
        
    except Exception as e:
        execution_time = time.time() - start_time
        return False, str(e), execution_time


def print_summary(results: Dict[str, List[ScalabilityResult]]):
    """Print summary of scalability test results."""
    print("\n" + "="*70)
    print("SCALABILITY TEST SUMMARY")
    print("="*70)
    
    for circuit_type, circuit_results in results.items():
        print(f"\n{circuit_type.upper()} CIRCUIT:")
        
        successful = [r for r in circuit_results if r.success]
        failed = [r for r in circuit_results if not r.success]
        
        if successful:
            max_successful = max(successful, key=lambda r: r.n_qubits)
            print(f"  ✅ Maximum successful: {max_successful.n_qubits} qubits")
            print(f"     Execution time: {max_successful.execution_time:.2f}s")
            if max_successful.state_size:
                print(f"     State size: {max_successful.state_size:,} amplitudes")
                theoretical = 2 ** max_successful.n_qubits
                sparsity = (1 - max_successful.state_size / theoretical) * 100
                print(f"     Sparsity: {sparsity:.2f}% (theoretical: {theoretical:,})")
        
        if failed:
            first_failure = min(failed, key=lambda r: r.n_qubits)
            print(f"  ❌ First failure: {first_failure.n_qubits} qubits")
            print(f"     Error: {first_failure.error_message}")
        
        # Memory analysis
        if successful:
            memory_results = [r for r in successful if r.memory_usage_mb]
            if memory_results:
                avg_memory = sum(r.memory_usage_mb for r in memory_results) / len(memory_results)
                max_memory = max(r.memory_usage_mb for r in memory_results)
                print(f"  📊 Memory usage: avg {avg_memory:.1f} MB, max {max_memory:.1f} MB")
    
    print("\n" + "="*70)
    print("RECOMMENDATIONS:")
    print("="*70)
    
    # Find limits
    ghz_results = results.get("GHZ", [])
    qft_results = results.get("QFT", [])
    
    if ghz_results:
        ghz_max = max([r for r in ghz_results if r.success], key=lambda r: r.n_qubits, default=None)
        if ghz_max:
            print(f"✅ Sparse circuits (GHZ): Up to {ghz_max.n_qubits} qubits")
    
    if qft_results:
        qft_max = max([r for r in qft_results if r.success], key=lambda r: r.n_qubits, default=None)
        if qft_max:
            print(f"⚠️  Dense circuits (QFT): Up to {qft_max.n_qubits} qubits")
            theoretical_memory = (2 ** qft_max.n_qubits) * 16 / 1024 / 1024 / 1024
            print(f"   Theoretical memory for {qft_max.n_qubits + 1} qubits: {theoretical_memory:.1f} GB")
    
    print("\n💡 For 50 qubits:")
    theoretical_50 = (2 ** 50) * 16 / 1024 / 1024 / 1024 / 1024  # PB
    print(f"   Theoretical memory: {theoretical_50:.2f} PB (Petabytes)")
    print(f"   This is IMPOSSIBLE with current hardware")
    print(f"   Need: Tensor networks, Stabilizer formalism, or specialized hardware")


def main():
    """Run scalability tests."""
    print("="*70)
    print("QUANTUM SIMULATOR SCALABILITY TEST")
    print("="*70)
    print("\nTesting practical limits for:")
    print("  1. Sparse states (GHZ circuit)")
    print("  2. Dense states (QFT circuit)")
    print("  3. Non-stabilizer circuits (T + CR gates)")
    print()
    
    # Create config
    config = SimulatorConfig(
        base_path=Path("data/scalability_test"),
        spark_master="local[*]",
        spark_shuffle_partitions=200,
        batch_size=10,
    )
    config.ensure_paths()
    
    results = {}
    
    try:
        with SparkHiSVSIMDriver(config, enable_parallel=True) as driver:
            # Test 1: GHZ (sparse - should scale well)
            print("\n" + "="*70)
            print("TEST 1: GHZ CIRCUIT (Sparse State)")
            print("="*70)
            ghz_results = test_circuit_scalability(
                driver,
                generate_ghz_circuit,
                "GHZ",
                start_qubits=5,
                max_qubits=30,  # Start conservative
                step=1,
                timeout_seconds=60,
            )
            results["GHZ"] = ghz_results
            
            # Test 2: QFT (dense - will hit memory limits)
            print("\n" + "="*70)
            print("TEST 2: QFT CIRCUIT (Dense State)")
            print("="*70)
            qft_results = test_circuit_scalability(
                driver,
                generate_qft_circuit,
                "QFT",
                start_qubits=5,
                max_qubits=25,  # Dense states hit limits earlier
                step=1,
                timeout_seconds=120,
            )
            results["QFT"] = qft_results
            
            # Test 3: Non-stabilizer (T + CR gates)
            print("\n" + "="*70)
            print("TEST 3: NON-STABILIZER CIRCUIT (T + CR gates)")
            print("="*70)
            non_stab_results = []
            for n_qubits in range(5, 21, 1):  # Test up to 20 qubits
                print(f"Testing non-stabilizer circuit with {n_qubits} qubits...", end=" ", flush=True)
                success, error, exec_time = test_non_stabilizer_circuit(driver, n_qubits)
                
                if success:
                    state_size = driver.run_circuit(
                        {"number_of_qubits": n_qubits, "gates": []},
                        enable_parallel=True,
                        resume=False
                    ).final_state_df.count()
                    print(f"✅ SUCCESS ({exec_time:.2f}s)")
                    non_stab_results.append(ScalabilityResult(
                        circuit_type="Non-Stabilizer",
                        n_qubits=n_qubits,
                        success=True,
                        execution_time=exec_time,
                    ))
                else:
                    print(f"❌ FAILED ({exec_time:.2f}s): {error}")
                    non_stab_results.append(ScalabilityResult(
                        circuit_type="Non-Stabilizer",
                        n_qubits=n_qubits,
                        success=False,
                        execution_time=exec_time,
                        error_message=error,
                    ))
                    if "memory" in error.lower():
                        break
            
            results["Non-Stabilizer"] = non_stab_results
            
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        traceback.print_exc()
    
    # Print summary
    print_summary(results)
    
    # Save results to file
    results_file = Path("data/scalability_test/scalability_results.txt")
    with open(results_file, "w") as f:
        f.write("SCALABILITY TEST RESULTS\n")
        f.write("="*70 + "\n\n")
        for circuit_type, circuit_results in results.items():
            f.write(f"{circuit_type}:\n")
            for r in circuit_results:
                f.write(f"  {r.n_qubits} qubits: {'✅' if r.success else '❌'} "
                       f"({r.execution_time:.2f}s")
                if r.state_size:
                    f.write(f", {r.state_size} amplitudes")
                if r.error_message:
                    f.write(f", Error: {r.error_message}")
                f.write(")\n")
            f.write("\n")
    
    print(f"\n📄 Results saved to: {results_file}")


if __name__ == "__main__":
    main()
