"""
Critical test: Verify v3 produces identical results to v1 implementation.

This is the most important test - if v3 doesn't match v1, it's broken.
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

# Import v1 modules directly
import importlib.util

spec_db = importlib.util.spec_from_file_location("db", V1_IMPL / "src" / "db.py")
db_module = importlib.util.module_from_spec(spec_db)
spec_db.loader.exec_module(db_module)
db = db_module

spec_sim = importlib.util.spec_from_file_location("simulator", V1_IMPL / "src" / "simulator.py")
sim_module = importlib.util.module_from_spec(spec_sim)
spec_sim.loader.exec_module(sim_module)
run_circuit = sim_module.run_circuit

spec_state = importlib.util.spec_from_file_location("state_manager", V1_IMPL / "src" / "state_manager.py")
state_module = importlib.util.module_from_spec(spec_state)
spec_state.loader.exec_module(state_module)
fetch_state = state_module.fetch_state

spec_circuits = importlib.util.spec_from_file_location("circuits", V1_IMPL / "src" / "circuits.py")
circuits_module = importlib.util.module_from_spec(spec_circuits)
spec_circuits.loader.exec_module(circuits_module)
generate_ghz_circuit = circuits_module.generate_ghz_circuit
generate_qft_circuit = circuits_module.generate_qft_circuit
generate_w_circuit = circuits_module.generate_w_circuit
generate_qpe_circuit = circuits_module.generate_qpe_circuit

from driver import SparkHiSVSIMDriver
from v2_common import config

SimulatorConfig = config.SimulatorConfig

SCHEMA = V1_IMPL / "sql" / "schema.sql"


def run_v1_simulation(circuit_dict):
    """Run simulation using v1 (reference implementation)."""
    conn = sqlite3.connect(":memory:")
    db.initialize_schema(conn, str(SCHEMA))
    version = run_circuit(conn, circuit_dict)
    
    # Convert to numpy array
    n_qubits = circuit_dict["number_of_qubits"]
    state = np.zeros(2**n_qubits, dtype=complex)
    rows = fetch_state(conn, version)
    for row in rows:
        idx, real, imag = row[0], row[1], row[2]
        state[idx] = complex(real, imag)
    
    conn.close()
    return state


def run_v3_simulation(circuit_dict, config, n_partitions=2):
    """Run simulation using v3."""
    with SparkHiSVSIMDriver(config) as driver:
        result = driver.run_circuit(circuit_dict, n_partitions=n_partitions)
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


class TestV3VsV1Parity:
    """Critical tests: v3 must match v1 exactly."""
    
    @pytest.mark.parametrize("n_qubits", [3, 4, 5])
    def test_ghz_parity(self, config_v3, n_qubits):
        """Test GHZ circuit parity - CRITICAL."""
        circuit = generate_ghz_circuit(n_qubits)
        
        v1_state = run_v1_simulation(circuit)
        v3_state = run_v3_simulation(circuit, config_v3, n_partitions=2)
        
        # Must match exactly
        np.testing.assert_allclose(v3_state, v1_state, atol=1e-10, rtol=1e-10)
    
    @pytest.mark.parametrize("n_qubits", [3, 4])
    def test_qft_parity(self, config_v3, n_qubits):
        """Test QFT circuit parity - CRITICAL."""
        circuit = generate_qft_circuit(n_qubits)
        
        v1_state = run_v1_simulation(circuit)
        v3_state = run_v3_simulation(circuit, config_v3, n_partitions=3)
        
        # Must match exactly
        np.testing.assert_allclose(v3_state, v1_state, atol=1e-10, rtol=1e-10)
    
    @pytest.mark.parametrize("n_qubits", [3, 4])
    def test_w_state_parity(self, config_v3, n_qubits):
        """Test W-state circuit parity - CRITICAL."""
        circuit = generate_w_circuit(n_qubits)
        
        v1_state = run_v1_simulation(circuit)
        v3_state = run_v3_simulation(circuit, config_v3, n_partitions=2)
        
        # Must match exactly
        np.testing.assert_allclose(v3_state, v1_state, atol=1e-10, rtol=1e-10)
    
    @pytest.mark.parametrize("n_qubits", [3, 4])
    def test_qpe_parity(self, config_v3, n_qubits):
        """Test QPE circuit parity - CRITICAL."""
        circuit = generate_qpe_circuit(n_qubits)
        
        v1_state = run_v1_simulation(circuit)
        v3_state = run_v3_simulation(circuit, config_v3, n_partitions=3)
        
        # Must match exactly
        np.testing.assert_allclose(v3_state, v1_state, atol=1e-10, rtol=1e-10)
    
    def test_partition_independence(self, config_v3):
        """Test that different partition counts produce same result."""
        circuit = generate_qft_circuit(4)
        
        v1_state = run_v1_simulation(circuit)
        
        # Test with different partition counts
        for n_partitions in [1, 2, 3, 4]:
            v3_state = run_v3_simulation(circuit, config_v3, n_partitions=n_partitions)
            np.testing.assert_allclose(v3_state, v1_state, atol=1e-10, rtol=1e-10)


class TestGateOrderPreservation:
    """Test that gate order is preserved correctly."""
    
    def test_gate_matching_correctness(self, config_v3):
        """Test that gate matching logic correctly identifies gates."""
        circuit = generate_qft_circuit(4)
        
        # Run v3 and check that gates are applied in correct order
        v1_state = run_v1_simulation(circuit)
        v3_state = run_v3_simulation(circuit, config_v3, n_partitions=3)
        
        # If gate order is wrong, states will differ
        np.testing.assert_allclose(v3_state, v1_state, atol=1e-10)
    
    def test_duplicate_gates_handled(self, config_v3):
        """Test that duplicate gates (same type, same qubits) are handled correctly."""
        # Create a circuit with duplicate gates
        circuit = {
            "number_of_qubits": 2,
            "gates": [
                {"gate": "H", "qubits": [0]},
                {"gate": "H", "qubits": [0]},  # Duplicate
                {"gate": "CNOT", "qubits": [0, 1]},
            ]
        }
        
        v1_state = run_v1_simulation(circuit)
        v3_state = run_v3_simulation(circuit, config_v3, n_partitions=2)
        
        np.testing.assert_allclose(v3_state, v1_state, atol=1e-10)
