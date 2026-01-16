"""
Performance benchmarks to measure speedup from partitioning and parallelism.
"""
from __future__ import annotations

import pytest
import numpy as np
import tempfile
import shutil
import time
import sys
from pathlib import Path

# Add paths
V3_SPARK = Path(__file__).parent.parent
if str(V3_SPARK / "src") not in sys.path:
    sys.path.insert(0, str(V3_SPARK / "src"))

from driver import SparkHiSVSIMDriver
from v2_common import config, circuits

SimulatorConfig = config.SimulatorConfig
generate_ghz_circuit = circuits.generate_ghz_circuit
generate_qft_circuit = circuits.generate_qft_circuit
generate_w_circuit = circuits.generate_w_circuit


@pytest.fixture
def config_v3():
    """Create a test configuration for v3."""
    temp_dir = Path(tempfile.mkdtemp())
    cfg = SimulatorConfig(
        run_id="test_perf",
        base_path=temp_dir,
        batch_size=10,
        spark_master="local[2]",
        spark_shuffle_partitions=4,
    )
    cfg.ensure_paths()
    yield cfg
    shutil.rmtree(temp_dir, ignore_errors=True)


class TestPerformance:
    """Performance benchmarks."""
    
    def test_partitioning_overhead(self, config_v3):
        """Measure overhead of partitioning."""
        circuit = generate_qft_circuit(5)
        
        # Sequential execution (no partitioning overhead)
        start = time.time()
        with SparkHiSVSIMDriver(config_v3, enable_parallel=False) as driver:
            result_seq = driver.run_circuit(circuit, n_partitions=1, enable_parallel=False)
        time_seq = time.time() - start
        
        # With partitioning
        start = time.time()
        with SparkHiSVSIMDriver(config_v3, enable_parallel=False) as driver:
            result_part = driver.run_circuit(circuit, n_partitions=3, enable_parallel=False)
        time_part = time.time() - start
        
        # Verify correctness
        state_seq = driver.get_state_vector(result_seq)
        state_part = driver.get_state_vector(result_part)
        np.testing.assert_allclose(state_seq, state_part, atol=1e-10)
        
        print(f"\nSequential (1 partition): {time_seq:.3f}s")
        print(f"With partitioning (3 partitions): {time_part:.3f}s")
        print(f"Overhead: {((time_part / time_seq - 1) * 100):.1f}%")
        
        # Partitioning should have minimal overhead
        assert time_part < time_seq * 2  # Less than 2x overhead
    
    def test_parallel_vs_sequential(self, config_v3):
        """Compare parallel vs sequential execution."""
        circuit = generate_qft_circuit(4)
        
        # Sequential execution
        start = time.time()
        with SparkHiSVSIMDriver(config_v3, enable_parallel=False) as driver:
            result_seq = driver.run_circuit(circuit, n_partitions=3, enable_parallel=False)
        time_seq = time.time() - start
        
        # Parallel execution (level-based)
        start = time.time()
        with SparkHiSVSIMDriver(config_v3, enable_parallel=True) as driver:
            result_par = driver.run_circuit(circuit, n_partitions=3, enable_parallel=True)
        time_par = time.time() - start
        
        # Verify correctness
        state_seq = driver.get_state_vector(result_seq)
        state_par = driver.get_state_vector(result_par)
        np.testing.assert_allclose(state_seq, state_par, atol=1e-10)
        
        print(f"\nSequential: {time_seq:.3f}s")
        print(f"Parallel (level-based): {time_par:.3f}s")
        print(f"Speedup: {time_seq / time_par:.2f}x")
    
    def test_scaling_with_partitions(self, config_v3):
        """Test how performance scales with number of partitions."""
        circuit = generate_qft_circuit(5)
        
        times = []
        for n_partitions in [1, 2, 3, 4]:
            start = time.time()
            with SparkHiSVSIMDriver(config_v3, enable_parallel=False) as driver:
                result = driver.run_circuit(circuit, n_partitions=n_partitions, enable_parallel=False)
            elapsed = time.time() - start
            times.append((n_partitions, elapsed))
            print(f"  {n_partitions} partitions: {elapsed:.3f}s")
        
        print(f"\nScaling results:")
        for n_part, t in times:
            print(f"  {n_part} partitions: {t:.3f}s")
        
        # All should produce same result
        states = []
        for n_partitions in [1, 2, 3, 4]:
            with SparkHiSVSIMDriver(config_v3, enable_parallel=False) as driver:
                result = driver.run_circuit(circuit, n_partitions=n_partitions, enable_parallel=False)
                states.append(driver.get_state_vector(result))
        
        for i in range(1, len(states)):
            np.testing.assert_allclose(states[0], states[i], atol=1e-10)
    
    def test_large_circuit_performance(self, config_v3):
        """Test performance on larger circuits."""
        circuits_to_test = [
            ("GHZ", generate_ghz_circuit, 8),
            ("QFT", generate_qft_circuit, 6),
            ("W", generate_w_circuit, 6),
        ]
        
        results = []
        for name, generator, n_qubits in circuits_to_test:
            circuit = generator(n_qubits)
            
            # Sequential
            start = time.time()
            with SparkHiSVSIMDriver(config_v3, enable_parallel=False) as driver:
                result = driver.run_circuit(circuit, n_partitions=2, enable_parallel=False)
            time_seq = time.time() - start
            
            # Parallel
            start = time.time()
            with SparkHiSVSIMDriver(config_v3, enable_parallel=True) as driver:
                result_par = driver.run_circuit(circuit, n_partitions=2, enable_parallel=True)
            time_par = time.time() - start
            
            # Verify correctness
            state_seq = driver.get_state_vector(result)
            state_par = driver.get_state_vector(result_par)
            np.testing.assert_allclose(state_seq, state_par, atol=1e-10)
            
            results.append({
                'circuit': name,
                'n_qubits': n_qubits,
                'n_gates': len(circuit['gates']),
                'time_seq': time_seq,
                'time_par': time_par,
                'speedup': time_seq / time_par if time_par > 0 else 1.0,
            })
            
            print(f"\n{name} ({n_qubits} qubits, {len(circuit['gates'])} gates):")
            print(f"  Sequential: {time_seq:.3f}s")
            print(f"  Parallel: {time_par:.3f}s")
            print(f"  Speedup: {time_seq / time_par:.2f}x")
        
        # Print summary
        print("\n" + "="*60)
        print("Performance Summary")
        print("="*60)
        for r in results:
            print(f"{r['circuit']:6s} | {r['n_qubits']:2d} qubits | {r['n_gates']:3d} gates | "
                  f"Seq: {r['time_seq']:6.3f}s | Par: {r['time_par']:6.3f}s | "
                  f"Speedup: {r['speedup']:5.2f}x")
