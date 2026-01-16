"""
Tests for circuits with intermediate sparsity levels.

GHZ states have exactly 2 non-zero amplitudes (sparse).
Hadamard walls have 2^n non-zero amplitudes (dense).

But what about circuits with 4, 8, 16, etc. non-zero amplitudes?
These test intermediate sparsity levels.
"""
from __future__ import annotations

import pytest
import numpy as np
import tempfile
import shutil
import sys
from pathlib import Path

# Add paths
V3_SPARK = Path(__file__).parent.parent
if str(V3_SPARK / "src") not in sys.path:
    sys.path.insert(0, str(V3_SPARK / "src"))

from driver import SparkHiSVSIMDriver
from v2_common import config

SimulatorConfig = config.SimulatorConfig


@pytest.fixture
def config_v3():
    """Create a test configuration for v3."""
    temp_dir = Path(tempfile.mkdtemp())
    cfg = SimulatorConfig(
        run_id="test_intermediate_sparsity",
        base_path=temp_dir,
        batch_size=10,
        spark_master="local[2]",
        spark_shuffle_partitions=4,
    )
    cfg.ensure_paths()
    yield cfg
    shutil.rmtree(temp_dir, ignore_errors=True)


def generate_ghz_like_4_amplitudes(n_qubits: int):
    """Create a state with 4 non-zero amplitudes.
    
    Apply H to first two qubits, then entangle with CNOTs.
    This creates: (|00...0⟩ + |01...0⟩ + |10...0⟩ + |11...1⟩) / 2
    """
    gates = [
        {"gate": "H", "qubits": [0]},
        {"gate": "H", "qubits": [1]},
    ]
    # Add CNOTs to entangle remaining qubits
    for q in range(2, n_qubits):
        gates.append({"gate": "CNOT", "qubits": [q-1, q]})
    
    return {"number_of_qubits": n_qubits, "gates": gates}


def generate_8_amplitude_state(n_qubits: int):
    """Create a state with 8 non-zero amplitudes.
    
    Apply H to first three qubits, then entangle.
    """
    gates = [
        {"gate": "H", "qubits": [0]},
        {"gate": "H", "qubits": [1]},
        {"gate": "H", "qubits": [2]},
    ]
    # Add CNOTs to entangle remaining qubits
    for q in range(3, n_qubits):
        gates.append({"gate": "CNOT", "qubits": [q-1, q]})
    
    return {"number_of_qubits": n_qubits, "gates": gates}


def generate_16_amplitude_state(n_qubits: int):
    """Create a state with 16 non-zero amplitudes.
    
    Apply H to first four qubits, then entangle.
    """
    gates = [
        {"gate": "H", "qubits": [0]},
        {"gate": "H", "qubits": [1]},
        {"gate": "H", "qubits": [2]},
        {"gate": "H", "qubits": [3]},
    ]
    # Add CNOTs to entangle remaining qubits
    for q in range(4, n_qubits):
        gates.append({"gate": "CNOT", "qubits": [q-1, q]})
    
    return {"number_of_qubits": n_qubits, "gates": gates}


def generate_partial_superposition(n_qubits: int, num_hadamards: int):
    """Create a state with 2^num_hadamards non-zero amplitudes.
    
    Apply H to first num_hadamards qubits, then entangle rest.
    """
    gates = []
    for i in range(num_hadamards):
        gates.append({"gate": "H", "qubits": [i]})
    
    # Add CNOTs to entangle remaining qubits
    for q in range(num_hadamards, n_qubits):
        gates.append({"gate": "CNOT", "qubits": [q-1, q]})
    
    return {"number_of_qubits": n_qubits, "gates": gates}


def generate_cluster_state(n_qubits: int):
    """Create a cluster state with many non-zero amplitudes.
    
    Cluster states have 2^(n/2) to 2^n non-zero amplitudes depending on structure.
    """
    gates = []
    # Apply H to all qubits
    for i in range(n_qubits):
        gates.append({"gate": "H", "qubits": [i]})
    
    # Apply CZ gates in a linear chain
    for i in range(n_qubits - 1):
        gates.append({"gate": "CZ", "qubits": [i, i+1]})
    
    return {"number_of_qubits": n_qubits, "gates": gates}


def generate_ring_state(n_qubits: int):
    """Create a ring state with many non-zero amplitudes.
    
    Apply H to all, then CZ in a ring structure.
    """
    gates = []
    # Apply H to all qubits
    for i in range(n_qubits):
        gates.append({"gate": "H", "qubits": [i]})
    
    # Apply CZ gates in a ring
    for i in range(n_qubits):
        gates.append({"gate": "CZ", "qubits": [i, (i+1) % n_qubits]})
    
    return {"number_of_qubits": n_qubits, "gates": gates}


class TestIntermediateSparsity:
    """Test circuits with intermediate sparsity levels."""
    
    def test_4_amplitude_state(self, config_v3):
        """Test state with 4 non-zero amplitudes."""
        circuit = generate_ghz_like_4_amplitudes(4)
        
        with SparkHiSVSIMDriver(config_v3, enable_parallel=False) as driver:
            result = driver.run_circuit(circuit, enable_parallel=False)
            state = driver.get_state_vector(result)
        
        nonzero = np.sum(np.abs(state) > 1e-10)
        assert nonzero == 4, f"Expected 4 non-zero amplitudes, got {nonzero}"
        
        # Check normalization
        norm = np.linalg.norm(state)
        assert np.isclose(norm, 1.0, atol=1e-10)
    
    def test_8_amplitude_state(self, config_v3):
        """Test state with 8 non-zero amplitudes."""
        circuit = generate_8_amplitude_state(4)
        
        with SparkHiSVSIMDriver(config_v3, enable_parallel=False) as driver:
            result = driver.run_circuit(circuit, enable_parallel=False)
            state = driver.get_state_vector(result)
        
        nonzero = np.sum(np.abs(state) > 1e-10)
        assert nonzero == 8, f"Expected 8 non-zero amplitudes, got {nonzero}"
        
        # Check normalization
        norm = np.linalg.norm(state)
        assert np.isclose(norm, 1.0, atol=1e-10)
    
    def test_16_amplitude_state(self, config_v3):
        """Test state with 16 non-zero amplitudes."""
        circuit = generate_16_amplitude_state(5)
        
        with SparkHiSVSIMDriver(config_v3, enable_parallel=False) as driver:
            result = driver.run_circuit(circuit, enable_parallel=False)
            state = driver.get_state_vector(result)
        
        nonzero = np.sum(np.abs(state) > 1e-10)
        assert nonzero == 16, f"Expected 16 non-zero amplitudes, got {nonzero}"
        
        # Check normalization
        norm = np.linalg.norm(state)
        assert np.isclose(norm, 1.0, atol=1e-10)
    
    @pytest.mark.parametrize("n_qubits,num_hadamards,expected_nonzero", [
        (4, 1, 2),   # 1 H -> 2 amplitudes
        (4, 2, 4),   # 2 H -> 4 amplitudes
        (4, 3, 8),   # 3 H -> 8 amplitudes
        (5, 2, 4),   # 2 H -> 4 amplitudes
        (5, 3, 8),   # 3 H -> 8 amplitudes
        (5, 4, 16),  # 4 H -> 16 amplitudes
        (6, 3, 8),   # 3 H -> 8 amplitudes
        (6, 4, 16),  # 4 H -> 16 amplitudes
    ])
    def test_partial_superposition(self, config_v3, n_qubits, num_hadamards, expected_nonzero):
        """Test partial superpositions with varying sparsity."""
        circuit = generate_partial_superposition(n_qubits, num_hadamards)
        
        with SparkHiSVSIMDriver(config_v3, enable_parallel=False) as driver:
            result = driver.run_circuit(circuit, enable_parallel=False)
            state = driver.get_state_vector(result)
        
        nonzero = np.sum(np.abs(state) > 1e-10)
        assert nonzero == expected_nonzero, \
            f"Expected {expected_nonzero} non-zero amplitudes, got {nonzero}"
        
        # Check normalization
        norm = np.linalg.norm(state)
        assert np.isclose(norm, 1.0, atol=1e-10)
    
    def test_cluster_state_4_qubits(self, config_v3):
        """Test cluster state (can be dense or intermediate sparsity)."""
        circuit = generate_cluster_state(4)
        
        with SparkHiSVSIMDriver(config_v3, enable_parallel=False) as driver:
            result = driver.run_circuit(circuit, enable_parallel=False)
            state = driver.get_state_vector(result)
        
        nonzero = np.sum(np.abs(state) > 1e-10)
        # Cluster state with H on all qubits + CZ chain can be dense
        assert nonzero >= 4, f"Cluster state should have >=4 non-zero amplitudes, got {nonzero}"
        assert nonzero <= 16, f"Cluster state should have <=16 non-zero amplitudes, got {nonzero}"
        
        # Check normalization
        norm = np.linalg.norm(state)
        assert np.isclose(norm, 1.0, atol=1e-10)
    
    def test_ring_state_5_qubits(self, config_v3):
        """Test ring state (intermediate sparsity)."""
        circuit = generate_ring_state(5)
        
        with SparkHiSVSIMDriver(config_v3, enable_parallel=False) as driver:
            result = driver.run_circuit(circuit, enable_parallel=False)
            state = driver.get_state_vector(result)
        
        nonzero = np.sum(np.abs(state) > 1e-10)
        # Ring state should have many non-zero amplitudes
        assert nonzero > 8, f"Ring state should have >8 non-zero amplitudes, got {nonzero}"
        assert nonzero <= 32, f"Ring state should have <=32 non-zero amplitudes, got {nonzero}"
        
        # Check normalization
        norm = np.linalg.norm(state)
        assert np.isclose(norm, 1.0, atol=1e-10)
    
    def test_many_amplitudes_6_qubits(self, config_v3):
        """Test state with many non-zero amplitudes (but not all)."""
        # Apply H to 5 out of 6 qubits -> 32 amplitudes
        circuit = generate_partial_superposition(6, 5)
        
        with SparkHiSVSIMDriver(config_v3, enable_parallel=False) as driver:
            result = driver.run_circuit(circuit, enable_parallel=False)
            state = driver.get_state_vector(result)
        
        nonzero = np.sum(np.abs(state) > 1e-10)
        assert nonzero == 32, f"Expected 32 non-zero amplitudes, got {nonzero}"
        
        # Check normalization
        norm = np.linalg.norm(state)
        assert np.isclose(norm, 1.0, atol=1e-10)
    
    def test_many_amplitudes_7_qubits(self, config_v3):
        """Test state with many non-zero amplitudes on larger system."""
        # Apply H to 4 out of 7 qubits -> 16 amplitudes
        circuit = generate_partial_superposition(7, 4)
        
        with SparkHiSVSIMDriver(config_v3, enable_parallel=False) as driver:
            result = driver.run_circuit(circuit, enable_parallel=False)
            state = driver.get_state_vector(result)
        
        nonzero = np.sum(np.abs(state) > 1e-10)
        assert nonzero == 16, f"Expected 16 non-zero amplitudes, got {nonzero}"
        
        # Check normalization
        norm = np.linalg.norm(state)
        assert np.isclose(norm, 1.0, atol=1e-10)
    
    def test_many_amplitudes_8_qubits(self, config_v3):
        """Test state with many non-zero amplitudes on 8-qubit system."""
        # Apply H to 5 out of 8 qubits -> 32 amplitudes
        circuit = generate_partial_superposition(8, 5)
        
        with SparkHiSVSIMDriver(config_v3, enable_parallel=False) as driver:
            result = driver.run_circuit(circuit, enable_parallel=False)
            state = driver.get_state_vector(result)
        
        nonzero = np.sum(np.abs(state) > 1e-10)
        assert nonzero == 32, f"Expected 32 non-zero amplitudes, got {nonzero}"
        
        # Check normalization
        norm = np.linalg.norm(state)
        assert np.isclose(norm, 1.0, atol=1e-10)


class TestSparsitySpectrum:
    """Test the full spectrum of sparsity levels."""
    
    def test_sparsity_spectrum_4_qubits(self, config_v3):
        """Test all sparsity levels for 4 qubits."""
        sparsity_levels = [
            (1, 2),   # 1 H -> 2 amplitudes (GHZ-like)
            (2, 4),   # 2 H -> 4 amplitudes
            (3, 8),   # 3 H -> 8 amplitudes
            (4, 16),  # 4 H -> 16 amplitudes (full superposition)
        ]
        
        for num_hadamards, expected_nonzero in sparsity_levels:
            circuit = generate_partial_superposition(4, num_hadamards)
            
            with SparkHiSVSIMDriver(config_v3, enable_parallel=False) as driver:
                result = driver.run_circuit(circuit, enable_parallel=False)
                state = driver.get_state_vector(result)
            
            nonzero = np.sum(np.abs(state) > 1e-10)
            assert nonzero == expected_nonzero, \
                f"{num_hadamards} H gates: Expected {expected_nonzero}, got {nonzero}"
            
            # Check normalization
            norm = np.linalg.norm(state)
            assert np.isclose(norm, 1.0, atol=1e-10)
    
    def test_sparsity_spectrum_5_qubits(self, config_v3):
        """Test sparsity spectrum for 5 qubits."""
        sparsity_levels = [
            (1, 2),   # 1 H -> 2 amplitudes
            (2, 4),   # 2 H -> 4 amplitudes
            (3, 8),   # 3 H -> 8 amplitudes
            (4, 16),  # 4 H -> 16 amplitudes
            (5, 32),  # 5 H -> 32 amplitudes (full superposition)
        ]
        
        for num_hadamards, expected_nonzero in sparsity_levels:
            circuit = generate_partial_superposition(5, num_hadamards)
            
            with SparkHiSVSIMDriver(config_v3, enable_parallel=False) as driver:
                result = driver.run_circuit(circuit, enable_parallel=False)
                state = driver.get_state_vector(result)
            
            nonzero = np.sum(np.abs(state) > 1e-10)
            assert nonzero == expected_nonzero, \
                f"{num_hadamards} H gates: Expected {expected_nonzero}, got {nonzero}"
            
            # Check normalization
            norm = np.linalg.norm(state)
            assert np.isclose(norm, 1.0, atol=1e-10)
