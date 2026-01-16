"""
DIRECT COMPARISON: v3 vs v1 implementation.

This test directly runs v1 and v3 and compares results to ensure correctness.
"""
from __future__ import annotations

import pytest
import numpy as np
import tempfile
import shutil
import sqlite3
import sys
from pathlib import Path

# Add paths
V1_IMPL = Path(__file__).parent.parent.parent / "v1_implementation"
V3_SPARK = Path(__file__).parent.parent
if str(V1_IMPL / "src") not in sys.path:
    sys.path.insert(0, str(V1_IMPL / "src"))
if str(V3_SPARK / "src") not in sys.path:
    sys.path.insert(0, str(V3_SPARK / "src"))

# Import v1 modules using direct file loading
import importlib.util

# Load v1 modules
spec_db = importlib.util.spec_from_file_location("db", V1_IMPL / "src" / "db.py")
db_module = importlib.util.module_from_spec(spec_db)
sys.modules['db'] = db_module
spec_db.loader.exec_module(db_module)
db = db_module

spec_sim = importlib.util.spec_from_file_location("simulator", V1_IMPL / "src" / "simulator.py")
sim_module = importlib.util.module_from_spec(spec_sim)
sys.modules['simulator'] = sim_module
spec_sim.loader.exec_module(sim_module)
run_circuit_v1 = sim_module.run_circuit

spec_state = importlib.util.spec_from_file_location("state_manager", V1_IMPL / "src" / "state_manager.py")
state_module = importlib.util.module_from_spec(spec_state)
sys.modules['state_manager'] = state_module
spec_state.loader.exec_module(state_module)
fetch_state_v1 = state_module.fetch_state

spec_circuits = importlib.util.spec_from_file_location("circuits", V1_IMPL / "src" / "circuits.py")
circuits_module = importlib.util.module_from_spec(spec_circuits)
sys.modules['circuits'] = circuits_module
spec_circuits.loader.exec_module(circuits_module)
generate_ghz_circuit_v1 = circuits_module.generate_ghz_circuit
generate_qft_circuit_v1 = circuits_module.generate_qft_circuit
generate_w_circuit_v1 = circuits_module.generate_w_circuit

# Import v3
from driver import SparkHiSVSIMDriver
from v2_common import config

SimulatorConfig = config.SimulatorConfig

SCHEMA = V1_IMPL / "sql" / "schema.sql"


def run_v1_simulation(circuit_dict):
    """Run simulation using v1 (reference implementation)."""
    conn = sqlite3.connect(":memory:")
    db.initialize_schema(conn, str(SCHEMA))
    version = run_circuit_v1(conn, circuit_dict)
    
    # Convert to numpy array
    n_qubits = circuit_dict["number_of_qubits"]
    state = np.zeros(2**n_qubits, dtype=complex)
    rows = fetch_state_v1(conn, version)
    for row in rows:
        idx, real, imag = row[0], row[1], row[2]
        state[idx] = complex(real, imag)
    
    conn.close()
    return state


def run_v3_simulation(circuit_dict, config_v3, n_partitions=2, enable_parallel=False):
    """Run simulation using v3."""
    with SparkHiSVSIMDriver(config_v3, enable_parallel=enable_parallel) as driver:
        result = driver.run_circuit(circuit_dict, n_partitions=n_partitions, enable_parallel=enable_parallel)
        state = driver.get_state_vector(result)
    return state


@pytest.fixture
def config_v3():
    """Create a test configuration for v3."""
    temp_dir = Path(tempfile.mkdtemp())
    cfg = SimulatorConfig(
        run_id="test_v3_vs_v1",
        base_path=temp_dir,
        batch_size=10,
        spark_master="local[2]",
        spark_shuffle_partitions=4,
    )
    cfg.ensure_paths()
    yield cfg
    shutil.rmtree(temp_dir, ignore_errors=True)


class TestV3VsV1Direct:
    """CRITICAL: Direct comparison of v3 vs v1 results."""
    
    @pytest.mark.parametrize("n_qubits", [2, 3, 4])
    def test_ghz_v3_matches_v1(self, config_v3, n_qubits):
        """CRITICAL: GHZ circuit - v3 must match v1 exactly."""
        circuit_v1 = generate_ghz_circuit_v1(n_qubits)
        
        v1_state = run_v1_simulation(circuit_v1)
        v3_state = run_v3_simulation(circuit_v1, config_v3, n_partitions=2, enable_parallel=False)
        
        # Must match exactly
        np.testing.assert_allclose(v3_state, v1_state, atol=1e-10, rtol=1e-10)
        
        # Also check non-zero amplitudes match
        v1_nonzero = np.sum(np.abs(v1_state) > 1e-10)
        v3_nonzero = np.sum(np.abs(v3_state) > 1e-10)
        assert v1_nonzero == v3_nonzero, f"Non-zero amplitudes differ: v1={v1_nonzero}, v3={v3_nonzero}"
    
    @pytest.mark.parametrize("n_qubits", [2, 3, 4])
    def test_qft_v3_matches_v1(self, config_v3, n_qubits):
        """CRITICAL: QFT circuit - v3 must match v1 exactly."""
        circuit_v1 = generate_qft_circuit_v1(n_qubits)
        
        v1_state = run_v1_simulation(circuit_v1)
        v3_state = run_v3_simulation(circuit_v1, config_v3, n_partitions=3, enable_parallel=False)
        
        # Must match exactly
        np.testing.assert_allclose(v3_state, v1_state, atol=1e-10, rtol=1e-10)
        
        # Check normalization
        v1_norm = np.linalg.norm(v1_state)
        v3_norm = np.linalg.norm(v3_state)
        assert np.isclose(v1_norm, v3_norm, atol=1e-10), f"Norms differ: v1={v1_norm}, v3={v3_norm}"
    
    @pytest.mark.parametrize("n_qubits", [3, 4])
    def test_w_state_v3_matches_v1(self, config_v3, n_qubits):
        """CRITICAL: W-state circuit - v3 must match v1 exactly."""
        circuit_v1 = generate_w_circuit_v1(n_qubits)
        
        v1_state = run_v1_simulation(circuit_v1)
        v3_state = run_v3_simulation(circuit_v1, config_v3, n_partitions=2, enable_parallel=False)
        
        # Must match exactly
        np.testing.assert_allclose(v3_state, v1_state, atol=1e-10, rtol=1e-10)
    
    def test_parallel_mode_matches_v1(self, config_v3):
        """CRITICAL: Parallel mode must also match v1."""
        circuit_v1 = generate_qft_circuit_v1(4)
        
        v1_state = run_v1_simulation(circuit_v1)
        
        # Test both sequential and parallel modes
        v3_seq = run_v3_simulation(circuit_v1, config_v3, n_partitions=3, enable_parallel=False)
        v3_par = run_v3_simulation(circuit_v1, config_v3, n_partitions=3, enable_parallel=True)
        
        # Both must match v1
        np.testing.assert_allclose(v3_seq, v1_state, atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose(v3_par, v1_state, atol=1e-10, rtol=1e-10)
        np.testing.assert_allclose(v3_seq, v3_par, atol=1e-10, rtol=1e-10)
    
    def test_different_partition_counts_match_v1(self, config_v3):
        """CRITICAL: Different partition counts must all match v1."""
        circuit_v1 = generate_ghz_circuit_v1(4)
        
        v1_state = run_v1_simulation(circuit_v1)
        
        # Test with different partition counts
        for n_partitions in [1, 2, 3]:
            v3_state = run_v3_simulation(circuit_v1, config_v3, n_partitions=n_partitions, enable_parallel=False)
            np.testing.assert_allclose(v3_state, v1_state, atol=1e-10, rtol=1e-10)
