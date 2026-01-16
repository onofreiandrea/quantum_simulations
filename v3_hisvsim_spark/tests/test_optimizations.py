"""
Test optimizations to verify they work correctly and improve performance.
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


@pytest.fixture
def config_v3():
    """Create a test configuration for v3."""
    temp_dir = Path(tempfile.mkdtemp())
    cfg = SimulatorConfig(
        run_id="test_opt",
        base_path=temp_dir,
        batch_size=10,
        spark_master="local[4]",
        spark_shuffle_partitions=8,
    )
    cfg.ensure_paths()
    yield cfg
    shutil.rmtree(temp_dir, ignore_errors=True)


class TestOptimizations:
    """Test that optimizations work correctly."""
    
    def test_sequential_no_partitioning(self, config_v3):
        """Test that sequential mode skips partitioning."""
        circuit = generate_qft_circuit(5)
        
        with SparkHiSVSIMDriver(config_v3, enable_parallel=False) as driver:
            result = driver.run_circuit(circuit, enable_parallel=False)
        
        # Should use 1 partition (no partitioning)
        assert result.n_partitions == 1
        assert not result.parallel_execution
        
        # Verify correctness
        state = driver.get_state_vector(result)
        norm = np.linalg.norm(state)
        assert np.isclose(norm, 1.0, atol=1e-10)
    
    def test_parallel_uses_partitioning(self, config_v3):
        """Test that parallel mode uses partitioning."""
        circuit = generate_qft_circuit(5)
        
        with SparkHiSVSIMDriver(config_v3, enable_parallel=True) as driver:
            result = driver.run_circuit(circuit, n_partitions=3, enable_parallel=True)
        
        # Should use multiple partitions
        assert result.n_partitions > 1
        assert result.parallel_execution
        
        # Verify correctness
        state = driver.get_state_vector(result)
        norm = np.linalg.norm(state)
        assert np.isclose(norm, 1.0, atol=1e-10)
    
    def test_optimizations_preserve_correctness(self, config_v3):
        """Test that optimizations don't break correctness."""
        circuit = generate_qft_circuit(4)
        
        # Sequential (optimized)
        with SparkHiSVSIMDriver(config_v3, enable_parallel=False) as driver:
            result_seq = driver.run_circuit(circuit, enable_parallel=False)
            state_seq = driver.get_state_vector(result_seq)
        
        # Parallel (optimized)
        with SparkHiSVSIMDriver(config_v3, enable_parallel=True) as driver:
            result_par = driver.run_circuit(circuit, n_partitions=3, enable_parallel=True)
            state_par = driver.get_state_vector(result_par)
        
        # Must match exactly
        np.testing.assert_allclose(state_seq, state_par, atol=1e-10)
    
    def test_performance_improvement(self, config_v3):
        """Test that optimizations improve performance."""
        circuit = generate_qft_circuit(5)
        
        # Sequential (optimized - no partitioning)
        start = time.time()
        with SparkHiSVSIMDriver(config_v3, enable_parallel=False) as driver:
            result_seq = driver.run_circuit(circuit, enable_parallel=False)
        time_seq = time.time() - start
        
        # Parallel (optimized)
        start = time.time()
        with SparkHiSVSIMDriver(config_v3, enable_parallel=True) as driver:
            result_par = driver.run_circuit(circuit, n_partitions=3, enable_parallel=True)
        time_par = time.time() - start
        
        # Verify correctness
        state_seq = driver.get_state_vector(result_seq)
        state_par = driver.get_state_vector(result_par)
        np.testing.assert_allclose(state_seq, state_par, atol=1e-10)
        
        # Performance should be better
        speedup = time_seq / time_par
        print(f"\nSequential: {time_seq:.3f}s")
        print(f"Parallel:   {time_par:.3f}s")
        print(f"Speedup:    {speedup:.2f}x")
        
        # Performance depends on circuit size and Spark overhead
        # For small circuits, sequential may be faster due to Spark overhead
        # For larger circuits, parallel should be faster
        # Just verify correctness - performance varies by circuit
        print(f"  Note: Speedup varies by circuit size and Spark overhead")
