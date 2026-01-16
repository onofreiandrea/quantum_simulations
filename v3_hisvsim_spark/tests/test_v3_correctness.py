"""
Correctness tests for v3 HiSVSIM + Spark integration.

Compares v3 results with v2 (reference implementation).
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
        run_id="test_v3",
        base_path=temp_dir,
        batch_size=10,
        spark_master="local[2]",
        spark_shuffle_partitions=4,
    )
    cfg.ensure_paths()
    yield cfg
    shutil.rmtree(temp_dir, ignore_errors=True)


@pytest.fixture
def config_v2():
    """Create a test configuration for v2 (reference)."""
    temp_dir = Path(tempfile.mkdtemp())
    cfg = SimulatorConfig(
        run_id="test_v2_ref",
        base_path=temp_dir,
        batch_size=10,
        spark_master="local[2]",
        spark_shuffle_partitions=4,
    )
    cfg.ensure_paths()
    yield cfg
    shutil.rmtree(temp_dir, ignore_errors=True)


def run_v2_simulation(circuit_dict, config):
    """Run simulation using v2 (reference)."""
    # Import v2 driver
    V2_SPARK = Path(__file__).parent.parent.parent / "v2_spark"
    if str(V2_SPARK / "src") not in sys.path:
        sys.path.insert(0, str(V2_SPARK / "src"))
    
    # Import using module loading
    import importlib.util
    import importlib
    
    # Load config first
    spec_config = importlib.util.spec_from_file_location(
        "config", V2_SPARK / "src" / "config.py"
    )
    config_module = importlib.util.module_from_spec(spec_config)
    spec_config.loader.exec_module(config_module)
    
    # Load driver
    spec_driver = importlib.util.spec_from_file_location(
        "driver", V2_SPARK / "src" / "driver.py"
    )
    driver_module = importlib.util.module_from_spec(spec_driver)
    spec_driver.loader.exec_module(driver_module)
    SparkQuantumDriver = driver_module.SparkQuantumDriver
    
    with SparkQuantumDriver(config) as driver:
        result = driver.run_circuit(circuit_dict, resume=False)
        state = driver.get_state_vector(result)
    return state


def run_v3_simulation(circuit_dict, config, n_partitions=2):
    """Run simulation using v3."""
    with SparkHiSVSIMDriver(config) as driver:
        result = driver.run_circuit(circuit_dict, n_partitions=n_partitions)
        state = driver.get_state_vector(result)
    return state


class TestV3VsV2Parity:
    """Test that v3 produces same results as v2."""
    
    @pytest.mark.parametrize("n_qubits", [2, 3, 4])
    def test_ghz_parity(self, config_v3, config_v2, n_qubits):
        """Test GHZ circuit parity between v2 and v3."""
        circuit = generate_ghz_circuit(n_qubits)
        
        v3_state = run_v3_simulation(circuit, config_v3, n_partitions=2)
        v2_state = run_v2_simulation(circuit, config_v2)
        
        np.testing.assert_allclose(v3_state, v2_state, atol=1e-10)
    
    @pytest.mark.parametrize("n_qubits", [2, 3, 4])
    def test_qft_parity(self, config_v3, config_v2, n_qubits):
        """Test QFT circuit parity between v2 and v3."""
        circuit = generate_qft_circuit(n_qubits)
        
        v3_state = run_v3_simulation(circuit, config_v3, n_partitions=3)
        v2_state = run_v2_simulation(circuit, config_v2)
        
        np.testing.assert_allclose(v3_state, v2_state, atol=1e-10)
    
    @pytest.mark.parametrize("n_qubits", [3, 4])
    def test_w_state_parity(self, config_v3, config_v2, n_qubits):
        """Test W-state circuit parity between v2 and v3."""
        circuit = generate_w_circuit(n_qubits)
        
        v3_state = run_v3_simulation(circuit, config_v3, n_partitions=2)
        v2_state = run_v2_simulation(circuit, config_v2)
        
        np.testing.assert_allclose(v3_state, v2_state, atol=1e-10)


class TestPartitioning:
    """Test partitioning functionality."""
    
    def test_partitioning_creates_partitions(self, config_v3):
        """Test that partitioning creates expected number of partitions."""
        circuit = generate_qft_circuit(4)
        
        with SparkHiSVSIMDriver(config_v3) as driver:
            result = driver.run_circuit(circuit, n_partitions=3)
            
            assert result.n_partitions == 3
            assert result.n_gates > 0
    
    def test_different_partition_counts(self, config_v3):
        """Test that different partition counts work."""
        circuit = generate_ghz_circuit(3)
        
        for n_partitions in [1, 2, 3]:
            with SparkHiSVSIMDriver(config_v3) as driver:
                result = driver.run_circuit(circuit, n_partitions=n_partitions)
                state = driver.get_state_vector(result)
                
                # Should produce same result regardless of partition count
                norm = np.linalg.norm(state)
                assert np.isclose(norm, 1.0, atol=1e-10)


class TestCorrectness:
    """Test correctness of results."""
    
    def test_ghz_normalization(self, config_v3):
        """Test that GHZ state is normalized."""
        circuit = generate_ghz_circuit(3)
        
        with SparkHiSVSIMDriver(config_v3) as driver:
            result = driver.run_circuit(circuit, n_partitions=2)
            state = driver.get_state_vector(result)
        
        norm = np.linalg.norm(state)
        assert np.isclose(norm, 1.0, atol=1e-10)
        
        # GHZ(3) should have 2 non-zero amplitudes
        non_zero = np.sum(np.abs(state) > 1e-10)
        assert non_zero == 2
    
    def test_qft_normalization(self, config_v3):
        """Test that QFT preserves normalization."""
        circuit = generate_qft_circuit(4)
        
        with SparkHiSVSIMDriver(config_v3) as driver:
            result = driver.run_circuit(circuit, n_partitions=3)
            state = driver.get_state_vector(result)
        
        norm = np.linalg.norm(state)
        assert np.isclose(norm, 1.0, atol=1e-10)
        
        # QFT|0⟩ should give uniform superposition
        expected_mag = 1 / np.sqrt(16)
        for amp in state:
            assert np.isclose(np.abs(amp), expected_mag, atol=1e-10)
