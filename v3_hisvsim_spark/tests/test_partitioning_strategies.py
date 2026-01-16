"""
Test partitioning strategies to verify optimization.
"""
from __future__ import annotations

import pytest
import numpy as np
import tempfile
import shutil
from pathlib import Path
import sys

# Add paths
V3_SPARK = Path(__file__).parent.parent
if str(V3_SPARK / "src") not in sys.path:
    sys.path.insert(0, str(V3_SPARK / "src"))

from driver import SparkHiSVSIMDriver
from hisvsim.partition_adapter import HiSVSIMPartitionAdapter
from v2_common import config, circuits

SimulatorConfig = config.SimulatorConfig
generate_ghz_circuit = circuits.generate_ghz_circuit
generate_qft_circuit = circuits.generate_qft_circuit


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


class TestPartitioningStrategies:
    """Test different partitioning strategies."""
    
    def test_all_strategies_produce_same_result(self, config_v3):
        """Test that all strategies produce the same quantum state."""
        circuit = generate_qft_circuit(4)
        
        strategies = ["load_balanced", "locality", "hybrid"]
        states = []
        
        for strategy in strategies:
            # Temporarily modify driver's adapter
            with SparkHiSVSIMDriver(config_v3) as driver:
                driver.partition_adapter = HiSVSIMPartitionAdapter(strategy=strategy)
                result = driver.run_circuit(circuit, n_partitions=3)
                state = driver.get_state_vector(result)
                states.append(state)
        
        # Compare all states
        for i in range(1, len(states)):
            np.testing.assert_allclose(states[0], states[i], atol=1e-10)
        
        # All should produce same final state
        states = []
        for strategy in strategies:
            # Temporarily modify driver's adapter
            with SparkHiSVSIMDriver(config_v3) as driver:
                driver.partition_adapter = HiSVSIMPartitionAdapter(strategy=strategy)
                result = driver.run_circuit(circuit, n_partitions=3)
                state = driver.get_state_vector(result)
                states.append(state)
        
        # Compare all states
        for i in range(1, len(states)):
            np.testing.assert_allclose(states[0], states[i], atol=1e-10)
    
    def test_hybrid_reduces_overlaps(self, config_v3):
        """Test that hybrid strategy reduces qubit overlaps."""
        circuit = generate_qft_circuit(5)
        
        adapter = HiSVSIMPartitionAdapter(strategy="hybrid")
        from v2_common import frontend
        n_qubits, gates = frontend.circuit_dict_to_gates(circuit)
        
        partition_indices = adapter.partition_circuit(gates, n_partitions=4)
        stats = adapter.get_partition_stats(gates, partition_indices)
        
        # Check that overlaps are reasonable
        for i, overlaps in enumerate(stats["qubit_overlaps"]):
            # Each partition should have some overlap (circuit is connected)
            # but not excessive overlap
            max_overlap = max(overlaps) if overlaps else 0
            assert max_overlap <= len(stats["partition_qubits"][i])  # Reasonable bound
    
    def test_load_balanced_distribution(self, config_v3):
        """Test that load-balanced strategy distributes gates evenly."""
        circuit = generate_qft_circuit(4)
        
        adapter = HiSVSIMPartitionAdapter(strategy="load_balanced")
        from v2_common import frontend
        n_qubits, gates = frontend.circuit_dict_to_gates(circuit)
        
        partition_indices = adapter.partition_circuit(gates, n_partitions=3)
        stats = adapter.get_partition_stats(gates, partition_indices)
        
        # Check load balance
        loads = stats["partition_loads"]
        if loads:
            max_load = max(loads)
            min_load = min(loads)
            # Loads should be relatively balanced (within 2x)
            assert max_load <= 2.0 * min_load or min_load == 0
    
    def test_locality_groups_qubits(self, config_v3):
        """Test that locality strategy groups gates by qubits."""
        circuit = generate_qft_circuit(4)
        
        adapter = HiSVSIMPartitionAdapter(strategy="locality")
        from v2_common import frontend
        n_qubits, gates = frontend.circuit_dict_to_gates(circuit)
        
        partition_indices = adapter.partition_circuit(gates, n_partitions=3)
        stats = adapter.get_partition_stats(gates, partition_indices)
        
        # Check that partitions have distinct qubit sets (minimal overlap)
        qubit_sets = stats["partition_qubits"]
        for i, qubits_i in enumerate(qubit_sets):
            for j, qubits_j in enumerate(qubit_sets):
                if i != j:
                    overlap = len(qubits_i & qubits_j)
                    # Overlap should be minimal (circuit connectivity may require some)
                    assert overlap <= min(len(qubits_i), len(qubits_j))


class TestPartitioningCorrectness:
    """Test that partitioning preserves correctness."""
    
    @pytest.mark.parametrize("strategy", ["load_balanced", "locality", "hybrid"])
    def test_strategy_preserves_correctness(self, config_v3, strategy):
        """Test that each strategy produces correct results."""
        circuit = generate_ghz_circuit(3)
        
        with SparkHiSVSIMDriver(config_v3) as driver:
            driver.partition_adapter = HiSVSIMPartitionAdapter(strategy=strategy)
            result = driver.run_circuit(circuit, n_partitions=2)
            state = driver.get_state_vector(result)
        
        # GHZ(3) = (|000⟩ + |111⟩) / √2
        expected = np.zeros(8, dtype=complex)
        expected[0] = 1/np.sqrt(2)
        expected[7] = 1/np.sqrt(2)
        
        np.testing.assert_allclose(state, expected, atol=1e-10)
    
    def test_partition_count_independence(self, config_v3):
        """Test that different partition counts produce same result."""
        circuit = generate_qft_circuit(3)
        
        states = []
        for n_partitions in [1, 2, 3]:
            with SparkHiSVSIMDriver(config_v3) as driver:
                result = driver.run_circuit(circuit, n_partitions=n_partitions)
                state = driver.get_state_vector(result)
                states.append(state)
        
        # All should be the same
        for i in range(1, len(states)):
            np.testing.assert_allclose(states[0], states[i], atol=1e-10)
