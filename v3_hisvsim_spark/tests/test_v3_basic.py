"""
Basic tests for v3 HiSVSIM + Spark integration.
"""
from __future__ import annotations

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
import sys

# Add paths
V2_SPARK = Path(__file__).parent.parent.parent / "v2_spark"
V3_SPARK = Path(__file__).parent.parent
if str(V2_SPARK / "src") not in sys.path:
    sys.path.insert(0, str(V2_SPARK / "src"))
if str(V3_SPARK / "src") not in sys.path:
    sys.path.insert(0, str(V3_SPARK / "src"))

from src.driver import SparkHiSVSIMDriver
from v2_spark.src.config import SimulatorConfig
from v2_spark.src.circuits import generate_ghz_circuit, generate_qft_circuit


@pytest.fixture
def config():
    """Create a test configuration."""
    temp_dir = Path(tempfile.mkdtemp())
    cfg = SimulatorConfig(
        run_id="test_v3",
        base_path=temp_dir,
        batch_size=10,
        spark_master="local[2]",
        spark_shuffle_partitions=4,
    )
    cfg.ensure_paths()
    yield cfg
    shutil.rmtree(temp_dir, ignore_errors=True)


class TestBasicSimulation:
    """Basic simulation tests."""
    
    def test_ghz_2_qubits(self, config):
        """Test GHZ circuit with 2 qubits."""
        circuit = generate_ghz_circuit(2)
        
        with SparkHiSVSIMDriver(config) as driver:
            result = driver.run_circuit(circuit, n_partitions=2)
            state = driver.get_state_vector(result)
        
        # GHZ(2) = (|00⟩ + |11⟩) / √2
        expected = np.array([1/np.sqrt(2), 0, 0, 1/np.sqrt(2)], dtype=complex)
        
        np.testing.assert_allclose(state, expected, atol=1e-10)
        assert result.n_partitions == 2
    
    def test_ghz_3_qubits(self, config):
        """Test GHZ circuit with 3 qubits."""
        circuit = generate_ghz_circuit(3)
        
        with SparkHiSVSIMDriver(config) as driver:
            result = driver.run_circuit(circuit, n_partitions=2)
            state = driver.get_state_vector(result)
        
        # GHZ(3) = (|000⟩ + |111⟩) / √2
        expected = np.zeros(8, dtype=complex)
        expected[0] = 1/np.sqrt(2)
        expected[7] = 1/np.sqrt(2)
        
        np.testing.assert_allclose(state, expected, atol=1e-10)
    
    def test_qft_3_qubits(self, config):
        """Test QFT circuit with 3 qubits."""
        circuit = generate_qft_circuit(3)
        
        with SparkHiSVSIMDriver(config) as driver:
            result = driver.run_circuit(circuit, n_partitions=3)
            state = driver.get_state_vector(result)
        
        # QFT should preserve normalization
        norm = np.linalg.norm(state)
        assert np.isclose(norm, 1.0, atol=1e-10)
        
        # QFT|0⟩ should give uniform superposition
        expected_mag = 1 / np.sqrt(8)
        for amp in state:
            assert np.isclose(np.abs(amp), expected_mag, atol=1e-10)
    
    def test_partitioning_stats(self, config):
        """Test that partitioning produces reasonable stats."""
        circuit = generate_qft_circuit(4)
        
        with SparkHiSVSIMDriver(config) as driver:
            result = driver.run_circuit(circuit, n_partitions=3)
            
            # Should have correct number of partitions
            assert result.n_partitions == 3
            assert result.n_gates > 0
            assert result.elapsed_time > 0


class TestCorrectness:
    """Correctness tests comparing with v2."""
    
    def test_ghz_matches_v2(self, config):
        """Test that GHZ results match v2 implementation."""
        circuit = generate_ghz_circuit(3)
        
        # Run v3
        with SparkHiSVSIMDriver(config) as driver:
            v3_result = driver.run_circuit(circuit, n_partitions=2)
            v3_state = driver.get_state_vector(v3_result)
        
        # Run v2 (reference)
        from v2_spark.src.driver import SparkQuantumDriver
        config_v2 = SimulatorConfig(
            run_id="test_v2_ref",
            base_path=config.base_path.parent / "v2_ref",
            batch_size=10,
            spark_master="local[2]",
            spark_shuffle_partitions=4,
        )
        config_v2.ensure_paths()
        
        with SparkQuantumDriver(config_v2) as driver_v2:
            v2_result = driver_v2.run_circuit(circuit, resume=False)
            v2_state = driver_v2.get_state_vector(v2_result)
        
        # Compare
        np.testing.assert_allclose(v3_state, v2_state, atol=1e-10)
        
        shutil.rmtree(config_v2.base_path, ignore_errors=True)
