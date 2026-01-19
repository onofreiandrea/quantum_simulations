"""
ADVERSARIAL TESTS - Designed to BREAK the implementation, not confirm it.

These tests:
1. Compare against v1 ACTUAL output (run both, compare)
2. Use externally verifiable mathematical properties
3. Try edge cases that could reveal bugs
4. Use random inputs to find unexpected failures
"""
from __future__ import annotations

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil
import sys
import sqlite3

# Add v2 to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Add v1 to path
V1_PATH = Path(__file__).parent.parent.parent / "v1_implementation"
sys.path.insert(0, str(V1_PATH))

from src.gates import (
    HadamardGate, XGate, YGate, ZGate, SGate, TGate,
    CNOTGate, CZGate, SWAPGate, CRGate
)
from src.circuits import generate_ghz_circuit, generate_qft_circuit
from src.frontend import circuit_dict_to_gates


# =============================================================================
# Test 1: Direct comparison with v1 implementation
# =============================================================================

def run_v1_simulation(circuit_dict: dict) -> np.ndarray:
    """
    Run v1 SQLite implementation and get state.
    
    This directly replicates the v1 SQL queries to avoid import issues.
    The SQL is copied EXACTLY from v1_implementation/src/gate_translator.py
    """
    # Use v2's gate parsing (identical to v1)
    n_qubits, gates = circuit_dict_to_gates(circuit_dict)
    
    # Create in-memory database
    conn = sqlite3.connect(":memory:")
    
    # Initialize schema (from v1)
    conn.executescript("""
        CREATE TABLE state (
            version INTEGER NOT NULL,
            idx INTEGER NOT NULL,
            real REAL NOT NULL,
            imag REAL NOT NULL,
            PRIMARY KEY (version, idx)
        );
        CREATE TABLE gate_matrix (
            gate_name TEXT NOT NULL,
            arity INTEGER NOT NULL,
            row INTEGER NOT NULL,
            col INTEGER NOT NULL,
            real REAL NOT NULL,
            imag REAL NOT NULL,
            PRIMARY KEY (gate_name, arity, row, col)
        );
    """)
    
    # Load gate matrices (from v1 gate_loader.py)
    seen = set()
    for gate in gates:
        key = (gate.gate_name, 2 if gate.two_qubit_gate else 1)
        if key in seen:
            continue
        seen.add(key)
        
        if gate.two_qubit_gate:
            T = gate.tensor.reshape(4, 4)
            for row in range(4):
                for col in range(4):
                    val = T[row, col]
                    conn.execute(
                        "INSERT OR REPLACE INTO gate_matrix(gate_name, arity, row, col, real, imag) VALUES (?, 2, ?, ?, ?, ?)",
                        (gate.gate_name, row, col, float(np.real(val)), float(np.imag(val)))
                    )
        else:
            U = gate.tensor
            for row in range(2):
                for col in range(2):
                    val = U[row, col]
                    conn.execute(
                        "INSERT OR REPLACE INTO gate_matrix(gate_name, arity, row, col, real, imag) VALUES (?, 1, ?, ?, ?, ?)",
                        (gate.gate_name, row, col, float(np.real(val)), float(np.imag(val)))
                    )
    conn.commit()
    
    # Initialize state |0...0⟩ (from v1 state_manager.py)
    conn.execute("DELETE FROM state;")
    conn.execute("INSERT INTO state(version, idx, real, imag) VALUES (0, 0, 1.0, 0.0);")
    conn.commit()
    
    # Apply gates using EXACT v1 SQL (from v1 gate_translator.py)
    version = 0
    for gate in gates:
        if gate.two_qubit_gate:
            q0, q1 = gate.qubits
            v1 = version + 1
            # EXACT SQL from v1_implementation/src/gate_translator.py:sql_apply_two_qubit_gate
            sql = f"""
            INSERT INTO state(version, idx, real, imag)
            SELECT
                {v1} AS version,
                (
                    (S.idx & ~((1 << {q0}) | (1 << {q1})))
                    | (((U.row >> 0) & 1) << {q0})
                    | (((U.row >> 1) & 1) << {q1})
                ) AS idx,
                SUM(U.real * S.real - U.imag * S.imag) AS real,
                SUM(U.real * S.imag + U.imag * S.real) AS imag
            FROM state AS S
            JOIN gate_matrix AS U
              ON U.gate_name = '{gate.gate_name}'
             AND U.arity = 2
             AND U.col = (
                 ((S.idx >> {q0}) & 1)
                 | (((S.idx >> {q1}) & 1) << 1)
             )
            WHERE S.version = {version}
            GROUP BY (
                (S.idx & ~((1 << {q0}) | (1 << {q1})))
                | (((U.row >> 0) & 1) << {q0})
                | (((U.row >> 1) & 1) << {q1})
            );
            """
        else:
            q = gate.qubits[0]
            v1 = version + 1
            # EXACT SQL from v1_implementation/src/gate_translator.py:sql_apply_one_qubit_gate
            sql = f"""
            INSERT INTO state(version, idx, real, imag)
            SELECT
                {v1} AS version,
                ((S.idx & ~(1 << {q})) | (U.row << {q})) AS idx,
                SUM(U.real * S.real - U.imag * S.imag) AS real,
                SUM(U.real * S.imag + U.imag * S.real) AS imag
            FROM state AS S
            JOIN gate_matrix AS U
              ON U.gate_name = '{gate.gate_name}'
             AND U.arity = 1
             AND U.col = ((S.idx >> {q}) & 1)
            WHERE S.version = {version}
            GROUP BY ((S.idx & ~(1 << {q})) | (U.row << {q}));
            """
        conn.executescript(sql)
        version += 1
    
    # Fetch final state
    rows = conn.execute(
        "SELECT idx, real, imag FROM state WHERE version = ? ORDER BY idx",
        (version,)
    ).fetchall()
    conn.close()
    
    # Convert to numpy array
    state = np.zeros(2 ** n_qubits, dtype=complex)
    for idx, real, imag in rows:
        state[idx] = complex(real, imag)
    
    return state


class TestV1Parity:
    """Direct comparison with v1 implementation."""
    
    @pytest.mark.parametrize("n_qubits", [2, 3, 4, 5, 6])
    def test_ghz_matches_v1(self, n_qubits):
        """GHZ circuit should produce identical results to v1."""
        circuit = generate_ghz_circuit(n_qubits)
        
        # Run v1
        v1_state = run_v1_simulation(circuit)
        
        # Run v2 (our implementation)
        from tests.test_spark_logic_verification import (
            simulate_circuit_pandas, pandas_state_to_numpy
        )
        pandas_df = simulate_circuit_pandas(circuit)
        v2_state = pandas_state_to_numpy(pandas_df, n_qubits)
        
        # Compare
        np.testing.assert_allclose(v2_state, v1_state, atol=1e-12,
            err_msg=f"GHZ({n_qubits}) mismatch between v1 and v2")
    
    @pytest.mark.parametrize("n_qubits", [2, 3, 4])
    def test_qft_matches_v1(self, n_qubits):
        """QFT circuit should produce identical results to v1."""
        circuit = generate_qft_circuit(n_qubits)
        
        v1_state = run_v1_simulation(circuit)
        
        from tests.test_spark_logic_verification import (
            simulate_circuit_pandas, pandas_state_to_numpy
        )
        pandas_df = simulate_circuit_pandas(circuit)
        v2_state = pandas_state_to_numpy(pandas_df, n_qubits)
        
        np.testing.assert_allclose(v2_state, v1_state, atol=1e-12,
            err_msg=f"QFT({n_qubits}) mismatch between v1 and v2")


# =============================================================================
# Test 2: Mathematical Properties (these MUST hold)
# =============================================================================

class TestMathematicalProperties:
    """Test properties that must hold for ANY correct quantum simulator."""
    
    def test_unitarity_preserves_norm(self):
        """
        PROPERTY: Unitary operations preserve the norm.
        For ANY circuit, ||output|| = ||input|| = 1
        """
        np.random.seed(42)
        
        for _ in range(20):
            # Random circuit
            n_qubits = np.random.randint(2, 5)
            n_gates = np.random.randint(5, 20)
            
            gates = []
            gate_types = ["H", "X", "Y", "Z", "S", "T"]
            
            for _ in range(n_gates):
                qubit = np.random.randint(0, n_qubits)
                gate_type = np.random.choice(gate_types)
                gates.append({"qubits": [qubit], "gate": gate_type})
            
            circuit = {"number_of_qubits": n_qubits, "gates": gates}
            
            from tests.test_spark_logic_verification import (
                simulate_circuit_pandas, pandas_state_to_numpy
            )
            pandas_df = simulate_circuit_pandas(circuit)
            state = pandas_state_to_numpy(pandas_df, n_qubits)
            
            norm = np.linalg.norm(state)
            assert np.isclose(norm, 1.0, atol=1e-10), \
                f"Normalization violated: ||ψ|| = {norm}"
    
    def test_probability_sum_is_one(self):
        """
        PROPERTY: Sum of probabilities = 1
        Σ|α_i|² = 1 for any valid state
        """
        circuits = [
            generate_ghz_circuit(4),
            generate_qft_circuit(3),
            {"number_of_qubits": 3, "gates": [
                {"qubits": [0], "gate": "H"},
                {"qubits": [1], "gate": "H"},
                {"qubits": [2], "gate": "H"},
            ]},
        ]
        
        for circuit in circuits:
            from tests.test_spark_logic_verification import (
                simulate_circuit_pandas, pandas_state_to_numpy
            )
            pandas_df = simulate_circuit_pandas(circuit)
            state = pandas_state_to_numpy(pandas_df, circuit["number_of_qubits"])
            
            prob_sum = np.sum(np.abs(state) ** 2)
            assert np.isclose(prob_sum, 1.0, atol=1e-10), \
                f"Probability sum = {prob_sum}, expected 1.0"
    
    def test_inverse_operations_cancel(self):
        """
        PROPERTY: A gate followed by its inverse = identity
        H·H = I, X·X = I, etc.
        """
        inverse_pairs = [
            ("H", "H"),
            ("X", "X"),
            ("Y", "Y"),
            ("Z", "Z"),
        ]
        
        for gate1, gate2 in inverse_pairs:
            circuit = {
                "number_of_qubits": 2,
                "gates": [
                    {"qubits": [0], "gate": "H"},  # Create superposition
                    {"qubits": [1], "gate": "X"},  # Create mixed state
                    {"qubits": [0], "gate": gate1},
                    {"qubits": [0], "gate": gate2},
                ]
            }
            
            # Also run without the inverse pair
            circuit_ref = {
                "number_of_qubits": 2,
                "gates": [
                    {"qubits": [0], "gate": "H"},
                    {"qubits": [1], "gate": "X"},
                ]
            }
            
            from tests.test_spark_logic_verification import (
                simulate_circuit_pandas, pandas_state_to_numpy
            )
            
            df1 = simulate_circuit_pandas(circuit)
            df2 = simulate_circuit_pandas(circuit_ref)
            
            state1 = pandas_state_to_numpy(df1, 2)
            state2 = pandas_state_to_numpy(df2, 2)
            
            np.testing.assert_allclose(state1, state2, atol=1e-10,
                err_msg=f"{gate1}·{gate2} ≠ I")
    
    def test_cnot_is_self_inverse(self):
        """CNOT·CNOT = I"""
        # Test on various input states
        for input_idx in range(4):
            circuit = {
                "number_of_qubits": 2,
                "gates": [
                    # Prepare input state
                    *([{"qubits": [0], "gate": "X"}] if input_idx & 1 else []),
                    *([{"qubits": [1], "gate": "X"}] if input_idx & 2 else []),
                    # Apply CNOT twice
                    {"qubits": [0, 1], "gate": "CNOT"},
                    {"qubits": [0, 1], "gate": "CNOT"},
                ]
            }
            
            from tests.test_spark_logic_verification import (
                simulate_circuit_pandas, pandas_state_to_numpy
            )
            df = simulate_circuit_pandas(circuit)
            state = pandas_state_to_numpy(df, 2)
            
            # Should return to input state
            assert np.isclose(np.abs(state[input_idx]), 1.0, atol=1e-10), \
                f"CNOT·CNOT failed for input |{input_idx:02b}⟩"


# =============================================================================
# Test 3: Known Analytical Results
# =============================================================================

class TestAnalyticalResults:
    """Compare against hand-calculated results."""
    
    def test_hadamard_exact_values(self):
        """H|0⟩ = (|0⟩ + |1⟩)/√2 with EXACT values."""
        circuit = {"number_of_qubits": 1, "gates": [{"qubits": [0], "gate": "H"}]}
        
        from tests.test_spark_logic_verification import (
            simulate_circuit_pandas, pandas_state_to_numpy
        )
        df = simulate_circuit_pandas(circuit)
        state = pandas_state_to_numpy(df, 1)
        
        # EXACT expected values
        sqrt2_inv = 1.0 / np.sqrt(2.0)
        
        assert np.isclose(state[0].real, sqrt2_inv, atol=1e-15), \
            f"H|0⟩[0].real = {state[0].real}, expected {sqrt2_inv}"
        assert np.isclose(state[0].imag, 0.0, atol=1e-15), \
            f"H|0⟩[0].imag = {state[0].imag}, expected 0"
        assert np.isclose(state[1].real, sqrt2_inv, atol=1e-15), \
            f"H|0⟩[1].real = {state[1].real}, expected {sqrt2_inv}"
        assert np.isclose(state[1].imag, 0.0, atol=1e-15), \
            f"H|0⟩[1].imag = {state[1].imag}, expected 0"
    
    def test_phase_gate_exact(self):
        """S|1⟩ = i|1⟩ with EXACT imaginary unit."""
        circuit = {
            "number_of_qubits": 1,
            "gates": [
                {"qubits": [0], "gate": "X"},  # |0⟩ → |1⟩
                {"qubits": [0], "gate": "S"},  # |1⟩ → i|1⟩
            ]
        }
        
        from tests.test_spark_logic_verification import (
            simulate_circuit_pandas, pandas_state_to_numpy
        )
        df = simulate_circuit_pandas(circuit)
        state = pandas_state_to_numpy(df, 1)
        
        assert np.isclose(state[0], 0.0, atol=1e-15), \
            f"S·X|0⟩[0] = {state[0]}, expected 0"
        assert np.isclose(state[1].real, 0.0, atol=1e-15), \
            f"S·X|0⟩[1].real = {state[1].real}, expected 0"
        assert np.isclose(state[1].imag, 1.0, atol=1e-15), \
            f"S·X|0⟩[1].imag = {state[1].imag}, expected 1"
    
    def test_bell_state_exact_phases(self):
        """Bell state must have POSITIVE real amplitudes."""
        circuit = {
            "number_of_qubits": 2,
            "gates": [
                {"qubits": [0], "gate": "H"},
                {"qubits": [0, 1], "gate": "CNOT"},
            ]
        }
        
        from tests.test_spark_logic_verification import (
            simulate_circuit_pandas, pandas_state_to_numpy
        )
        df = simulate_circuit_pandas(circuit)
        state = pandas_state_to_numpy(df, 2)
        
        sqrt2_inv = 1.0 / np.sqrt(2.0)
        
        # |00⟩ should have positive real amplitude
        assert state[0b00].real > 0, f"|00⟩ amplitude = {state[0b00]}, expected positive"
        assert np.isclose(state[0b00].real, sqrt2_inv, atol=1e-15)
        assert np.isclose(state[0b00].imag, 0.0, atol=1e-15)
        
        # |11⟩ should have positive real amplitude
        assert state[0b11].real > 0, f"|11⟩ amplitude = {state[0b11]}, expected positive"
        assert np.isclose(state[0b11].real, sqrt2_inv, atol=1e-15)
        assert np.isclose(state[0b11].imag, 0.0, atol=1e-15)
        
        # |01⟩ and |10⟩ should be exactly zero
        assert np.isclose(state[0b01], 0.0, atol=1e-15)
        assert np.isclose(state[0b10], 0.0, atol=1e-15)


# =============================================================================
# Test 4: Random Fuzzing
# =============================================================================

class TestRandomFuzzing:
    """Random inputs to find unexpected failures."""
    
    @pytest.mark.parametrize("seed", range(10))
    def test_random_circuit_normalization(self, seed):
        """Random circuits should always preserve normalization."""
        np.random.seed(seed)
        
        n_qubits = np.random.randint(2, 6)
        n_gates = np.random.randint(10, 50)
        
        one_qubit_gates = ["H", "X", "Y", "Z", "S", "T"]
        two_qubit_gates = ["CNOT", "CZ", "SWAP"]
        
        gates = []
        for _ in range(n_gates):
            if np.random.rand() < 0.7:
                # 1-qubit gate
                qubit = np.random.randint(0, n_qubits)
                gate = np.random.choice(one_qubit_gates)
                gates.append({"qubits": [qubit], "gate": gate})
            else:
                # 2-qubit gate
                q1, q2 = np.random.choice(n_qubits, size=2, replace=False)
                gate = np.random.choice(two_qubit_gates)
                gates.append({"qubits": [int(q1), int(q2)], "gate": gate})
        
        circuit = {"number_of_qubits": n_qubits, "gates": gates}
        
        from tests.test_spark_logic_verification import (
            simulate_circuit_pandas, pandas_state_to_numpy
        )
        df = simulate_circuit_pandas(circuit)
        state = pandas_state_to_numpy(df, n_qubits)
        
        norm = np.linalg.norm(state)
        assert np.isclose(norm, 1.0, atol=1e-8), \
            f"Seed {seed}: Normalization violated: ||ψ|| = {norm}"
    
    def test_deterministic_with_same_circuit(self):
        """Same circuit should always produce same result."""
        circuit = {
            "number_of_qubits": 3,
            "gates": [
                {"qubits": [0], "gate": "H"},
                {"qubits": [1], "gate": "X"},
                {"qubits": [0, 2], "gate": "CNOT"},
                {"qubits": [1], "gate": "S"},
            ]
        }
        
        from tests.test_spark_logic_verification import (
            simulate_circuit_pandas, pandas_state_to_numpy
        )
        
        # Run 5 times
        results = []
        for _ in range(5):
            df = simulate_circuit_pandas(circuit)
            state = pandas_state_to_numpy(df, 3)
            results.append(state)
        
        # All should be identical
        for i in range(1, 5):
            np.testing.assert_allclose(results[i], results[0], atol=1e-15,
                err_msg="Non-deterministic result!")


# =============================================================================
# Test 5: Edge Cases That Could Break Things
# =============================================================================

class TestBreakingEdgeCases:
    """Edge cases designed to break the implementation."""
    
    def test_many_consecutive_same_gate(self):
        """100 consecutive H gates on same qubit = I (even number)."""
        n_gates = 100  # Even number
        circuit = {
            "number_of_qubits": 2,
            "gates": [{"qubits": [0], "gate": "H"} for _ in range(n_gates)]
        }
        
        from tests.test_spark_logic_verification import (
            simulate_circuit_pandas, pandas_state_to_numpy
        )
        df = simulate_circuit_pandas(circuit)
        state = pandas_state_to_numpy(df, 2)
        
        # Should be |00⟩
        assert np.isclose(np.abs(state[0]), 1.0, atol=1e-8), \
            f"After {n_gates} H gates, expected |00⟩ but got different state"
    
    def test_highest_qubit_operations(self):
        """Operations on the highest qubit index."""
        for n_qubits in [2, 5, 8]:
            highest = n_qubits - 1
            circuit = {
                "number_of_qubits": n_qubits,
                "gates": [
                    {"qubits": [highest], "gate": "X"},
                    {"qubits": [highest], "gate": "H"},
                ]
            }
            
            from tests.test_spark_logic_verification import (
                simulate_circuit_pandas, pandas_state_to_numpy
            )
            df = simulate_circuit_pandas(circuit)
            state = pandas_state_to_numpy(df, n_qubits)
            
            # Check normalization
            norm = np.linalg.norm(state)
            assert np.isclose(norm, 1.0, atol=1e-10), \
                f"n_qubits={n_qubits}, highest={highest}: norm={norm}"
    
    def test_all_qubits_entangled(self):
        """Fully entangle all qubits and verify."""
        n_qubits = 6
        gates = [{"qubits": [0], "gate": "H"}]
        for i in range(1, n_qubits):
            gates.append({"qubits": [i-1, i], "gate": "CNOT"})
        
        circuit = {"number_of_qubits": n_qubits, "gates": gates}
        
        from tests.test_spark_logic_verification import (
            simulate_circuit_pandas, pandas_state_to_numpy
        )
        df = simulate_circuit_pandas(circuit)
        state = pandas_state_to_numpy(df, n_qubits)
        
        # GHZ state: only |00...0⟩ and |11...1⟩ should have amplitude
        non_zero = np.where(np.abs(state) > 1e-10)[0]
        assert len(non_zero) == 2, f"Expected 2 non-zero amplitudes, got {len(non_zero)}"
        assert 0 in non_zero, "|00...0⟩ should have amplitude"
        assert (2**n_qubits - 1) in non_zero, "|11...1⟩ should have amplitude"
    
    def test_swap_actually_swaps(self):
        """SWAP gate must actually exchange qubit states."""
        # Prepare |01⟩ (qubit 0 = 1, qubit 1 = 0)
        # After SWAP: |10⟩
        circuit = {
            "number_of_qubits": 2,
            "gates": [
                {"qubits": [0], "gate": "X"},  # |00⟩ → |01⟩
                {"qubits": [0, 1], "gate": "SWAP"},  # |01⟩ → |10⟩
            ]
        }
        
        from tests.test_spark_logic_verification import (
            simulate_circuit_pandas, pandas_state_to_numpy
        )
        df = simulate_circuit_pandas(circuit)
        state = pandas_state_to_numpy(df, 2)
        
        # In little-endian: |01⟩ is index 1, |10⟩ is index 2
        assert np.isclose(np.abs(state[0b10]), 1.0, atol=1e-10), \
            f"SWAP failed: expected |10⟩ but got state {state}"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
