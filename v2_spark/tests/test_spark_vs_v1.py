"""
Direct comparison: Actual Spark execution vs v1 SQL execution.

This is the ultimate parity test - runs both implementations
and compares the results bit-for-bit.
"""
from __future__ import annotations

import pytest
import numpy as np
from pathlib import Path
import tempfile
import shutil
import sqlite3

from pyspark.sql import SparkSession

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import SimulatorConfig
from src.driver import SparkQuantumDriver
from src.circuits import generate_ghz_circuit, generate_qft_circuit, generate_w_circuit
from src.frontend import circuit_dict_to_gates


def run_v1_sql(circuit_dict: dict) -> np.ndarray:
    """Run v1 SQL implementation."""
    n_qubits, gates = circuit_dict_to_gates(circuit_dict)
    
    conn = sqlite3.connect(":memory:")
    conn.executescript("""
        CREATE TABLE state (version INTEGER, idx INTEGER, real REAL, imag REAL, PRIMARY KEY (version, idx));
        CREATE TABLE gate_matrix (gate_name TEXT, arity INTEGER, row INTEGER, col INTEGER, real REAL, imag REAL, PRIMARY KEY (gate_name, arity, row, col));
    """)
    
    # Load gates
    seen = set()
    for gate in gates:
        key = (gate.gate_name, 2 if gate.two_qubit_gate else 1)
        if key in seen:
            continue
        seen.add(key)
        
        if gate.two_qubit_gate:
            T = gate.tensor.reshape(4, 4)
            for r in range(4):
                for c in range(4):
                    conn.execute("INSERT OR REPLACE INTO gate_matrix VALUES (?,2,?,?,?,?)",
                        (gate.gate_name, r, c, float(np.real(T[r,c])), float(np.imag(T[r,c]))))
        else:
            U = gate.tensor
            for r in range(2):
                for c in range(2):
                    conn.execute("INSERT OR REPLACE INTO gate_matrix VALUES (?,1,?,?,?,?)",
                        (gate.gate_name, r, c, float(np.real(U[r,c])), float(np.imag(U[r,c]))))
    
    conn.execute("INSERT INTO state VALUES (0,0,1.0,0.0)")
    conn.commit()
    
    version = 0
    for gate in gates:
        v1 = version + 1
        if gate.two_qubit_gate:
            q0, q1 = gate.qubits
            sql = f"""
            INSERT INTO state(version, idx, real, imag) SELECT {v1},
                (S.idx & ~((1<<{q0})|(1<<{q1}))) | ((U.row&1)<<{q0}) | (((U.row>>1)&1)<<{q1}),
                SUM(U.real*S.real-U.imag*S.imag), SUM(U.real*S.imag+U.imag*S.real)
            FROM state S JOIN gate_matrix U ON U.gate_name='{gate.gate_name}' AND U.arity=2
                AND U.col=((S.idx>>{q0})&1)|(((S.idx>>{q1})&1)<<1) WHERE S.version={version}
            GROUP BY (S.idx & ~((1<<{q0})|(1<<{q1}))) | ((U.row&1)<<{q0}) | (((U.row>>1)&1)<<{q1})
            """
        else:
            q = gate.qubits[0]
            sql = f"""
            INSERT INTO state(version, idx, real, imag) SELECT {v1},
                (S.idx & ~(1<<{q})) | (U.row<<{q}),
                SUM(U.real*S.real-U.imag*S.imag), SUM(U.real*S.imag+U.imag*S.real)
            FROM state S JOIN gate_matrix U ON U.gate_name='{gate.gate_name}' AND U.arity=1
                AND U.col=((S.idx>>{q})&1) WHERE S.version={version}
            GROUP BY (S.idx & ~(1<<{q})) | (U.row<<{q})
            """
        conn.executescript(sql)
        version += 1
    
    rows = conn.execute(f"SELECT idx, real, imag FROM state WHERE version={version}").fetchall()
    conn.close()
    
    state = np.zeros(2**n_qubits, dtype=complex)
    for idx, real, imag in rows:
        state[idx] = complex(real, imag)
    return state


@pytest.fixture
def temp_dir():
    path = Path(tempfile.mkdtemp())
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def config(temp_dir):
    return SimulatorConfig(base_path=temp_dir, batch_size=10)


class TestSparkVsV1:
    """Compare actual Spark execution against v1 SQL."""
    
    @pytest.mark.parametrize("n_qubits", [2, 3, 4, 5])
    def test_ghz_spark_matches_v1(self, config, n_qubits):
        """GHZ: Spark result should exactly match v1 SQL result."""
        circuit = generate_ghz_circuit(n_qubits)
        
        # Run v1 SQL
        v1_state = run_v1_sql(circuit)
        
        # Run Spark
        with SparkQuantumDriver(config) as driver:
            result = driver.run_circuit(circuit, resume=False)
            spark_state = driver.get_state_vector(result)
        
        # Compare
        np.testing.assert_allclose(spark_state, v1_state, atol=1e-12,
            err_msg=f"GHZ({n_qubits}): Spark != v1 SQL")
    
    @pytest.mark.parametrize("n_qubits", [2, 3, 4])
    def test_qft_spark_matches_v1(self, config, n_qubits):
        """QFT: Spark result should exactly match v1 SQL result."""
        circuit = generate_qft_circuit(n_qubits)
        
        v1_state = run_v1_sql(circuit)
        
        with SparkQuantumDriver(config) as driver:
            result = driver.run_circuit(circuit, resume=False)
            spark_state = driver.get_state_vector(result)
        
        np.testing.assert_allclose(spark_state, v1_state, atol=1e-12,
            err_msg=f"QFT({n_qubits}): Spark != v1 SQL")
    
    def test_random_circuit_spark_matches_v1(self, config):
        """Random circuit: Spark should match v1 SQL."""
        np.random.seed(12345)
        
        n_qubits = 4
        gates = []
        gate_types = ["H", "X", "Y", "Z", "S", "T"]
        two_qubit_types = ["CNOT", "CZ"]
        
        for _ in range(15):
            if np.random.rand() < 0.6:
                gates.append({
                    "qubits": [int(np.random.randint(0, n_qubits))],
                    "gate": np.random.choice(gate_types)
                })
            else:
                q1, q2 = np.random.choice(n_qubits, 2, replace=False)
                gates.append({
                    "qubits": [int(q1), int(q2)],
                    "gate": np.random.choice(two_qubit_types)
                })
        
        circuit = {"number_of_qubits": n_qubits, "gates": gates}
        
        v1_state = run_v1_sql(circuit)
        
        with SparkQuantumDriver(config) as driver:
            result = driver.run_circuit(circuit, resume=False)
            spark_state = driver.get_state_vector(result)
        
        np.testing.assert_allclose(spark_state, v1_state, atol=1e-12,
            err_msg="Random circuit: Spark != v1 SQL")
    
    def test_w_state_spark_matches_v1(self, config):
        """W-state: Spark should match v1 SQL."""
        circuit = generate_w_circuit(4)
        
        v1_state = run_v1_sql(circuit)
        
        with SparkQuantumDriver(config) as driver:
            result = driver.run_circuit(circuit, resume=False)
            spark_state = driver.get_state_vector(result)
        
        np.testing.assert_allclose(spark_state, v1_state, atol=1e-12,
            err_msg="W-state: Spark != v1 SQL")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
