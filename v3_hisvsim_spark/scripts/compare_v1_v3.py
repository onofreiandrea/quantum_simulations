#!/usr/bin/env python3
"""
Compare v1 (SQLite) vs v3 (Spark+HiSVSIM) simulation results.

This script properly imports both versions and compares their outputs
for various quantum circuits.
"""
import sys
import os
from pathlib import Path
import numpy as np
import tempfile
import shutil
import sqlite3

# Setup paths
SCRIPT_DIR = Path(__file__).parent.resolve()
V1_DIR = SCRIPT_DIR.parent.parent / "v1_implementation"
V3_DIR = SCRIPT_DIR.parent

# Add v3/src to path for imports
V3_SRC = V3_DIR / "src"
if str(V3_SRC) not in sys.path:
    sys.path.insert(0, str(V3_SRC))


# ============== V1 Implementation (inlined) ==============

def create_gate_matrix(gate_name: str, tensor: np.ndarray, two_qubit: bool) -> list:
    """Create gate matrix rows for SQL insertion."""
    rows = []
    if two_qubit:
        # 4x4 matrix
        tensor = tensor.reshape(4, 4)
        for row in range(4):
            for col in range(4):
                val = tensor[row, col]
                rows.append((gate_name, 2, row, col, float(np.real(val)), float(np.imag(val))))
    else:
        # 2x2 matrix
        for row in range(2):
            for col in range(2):
                val = tensor[row, col]
                rows.append((gate_name, 1, row, col, float(np.real(val)), float(np.imag(val))))
    return rows


def get_gate_tensor(gate_name: str, params: dict):
    """Get gate tensor and whether it's a 2-qubit gate."""
    PERM = np.array([0, 2, 1, 3])
    
    def to_little_endian(m):
        return m[np.ix_(PERM, PERM)]
    
    if gate_name == "H":
        matrix = (1 / np.sqrt(2)) * np.array([[1.0, 1.0], [1.0, -1.0]], dtype=complex)
        return matrix, False
    elif gate_name == "X":
        matrix = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
        return matrix, False
    elif gate_name == "Y":
        matrix = np.array([[0.0, -1.0j], [1.0j, 0.0]], dtype=complex)
        return matrix, False
    elif gate_name == "Z":
        matrix = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
        return matrix, False
    elif gate_name == "S":
        matrix = np.array([[1.0, 0.0], [0.0, 1.0j]], dtype=complex)
        return matrix, False
    elif gate_name == "T":
        matrix = np.array([[1.0, 0.0], [0.0, np.exp(1j * np.pi / 4)]], dtype=complex)
        return matrix, False
    elif gate_name == "RY":
        theta = params.get("theta", np.pi/4)
        matrix = np.array([
            [np.cos(theta / 2), -np.sin(theta / 2)],
            [np.sin(theta / 2), np.cos(theta / 2)]
        ], dtype=complex)
        return matrix, False
    elif gate_name == "R":
        k = params.get("k", 2)
        matrix = np.array([[1.0, 0.0], [0.0, np.exp(2j * np.pi / 2**k)]], dtype=complex)
        return matrix, False
    elif gate_name == "G":
        p = params.get("p", 2)
        matrix = np.array([
            [np.sqrt(1 / p), -np.sqrt(1 - (1 / p))],
            [np.sqrt(1 - (1 / p)), np.sqrt(1 / p)]
        ], dtype=complex)
        return matrix, False
    elif gate_name == "CNOT":
        matrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0]
        ], dtype=complex)
        return to_little_endian(matrix).reshape(2, 2, 2, 2), True
    elif gate_name == "CZ":
        matrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, -1]
        ], dtype=complex)
        return to_little_endian(matrix).reshape(2, 2, 2, 2), True
    elif gate_name == "CY":
        matrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 0, -1j],
            [0, 0, 1j, 0]
        ], dtype=complex)
        return to_little_endian(matrix).reshape(2, 2, 2, 2), True
    elif gate_name == "SWAP":
        matrix = np.array([
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [0, 1, 0, 0],
            [0, 0, 0, 1]
        ], dtype=complex)
        return to_little_endian(matrix).reshape(2, 2, 2, 2), True
    elif gate_name == "CR":
        k = params.get("k", 2)
        matrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, np.exp(2j * np.pi / 2**k)]
        ], dtype=complex)
        return to_little_endian(matrix).reshape(2, 2, 2, 2), True
    elif gate_name == "CU":
        U = params.get("U")
        exponent = params.get("exponent", 1)
        U_power = np.linalg.matrix_power(U, exponent)
        matrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, U_power[0, 0], U_power[0, 1]],
            [0, 0, U_power[1, 0], U_power[1, 1]]
        ], dtype=complex)
        return to_little_endian(matrix).reshape(2, 2, 2, 2), True
    else:
        raise ValueError(f"Unknown gate: {gate_name}")


def get_full_gate_name(gate_name: str, params: dict) -> str:
    """Get the full gate name including parameters."""
    if gate_name == "CR":
        return f"CR{params.get('k', 2)}"
    elif gate_name == "R":
        return f"R{params.get('k', 2)}"
    elif gate_name == "G":
        return f"G{params.get('p', 2)}"
    elif gate_name == "CU":
        return params.get("name", f"CU{params.get('exponent', 1)}")
    return gate_name


def sql_apply_one_qubit_gate(gate_name: str, q: int, v: int) -> str:
    v1 = v + 1
    return f"""
    INSERT INTO state(version, idx, real, imag)
    SELECT
        {v1} AS version,
        ((S.idx & ~(1 << {q})) | (U.row << {q})) AS idx,
        SUM(U.real * S.real - U.imag * S.imag) AS real,
        SUM(U.real * S.imag + U.imag * S.real) AS imag
    FROM state AS S
    JOIN gate_matrix AS U
      ON U.gate_name = '{gate_name}'
     AND U.arity = 1
     AND U.col = ((S.idx >> {q}) & 1)
    WHERE S.version = {v}
    GROUP BY ((S.idx & ~(1 << {q})) | (U.row << {q}));
    """


def sql_apply_two_qubit_gate(gate_name: str, q0: int, q1: int, v: int) -> str:
    v1 = v + 1
    return f"""
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
      ON U.gate_name = '{gate_name}'
     AND U.arity = 2
     AND U.col = (
         ((S.idx >> {q0}) & 1)
         | (((S.idx >> {q1}) & 1) << 1)
     )
    WHERE S.version = {v}
    GROUP BY (
        (S.idx & ~((1 << {q0}) | (1 << {q1})))
        | (((U.row >> 0) & 1) << {q0})
        | (((U.row >> 1) & 1) << {q1})
    );
    """


def run_v1_circuit(circuit_dict: dict) -> np.ndarray:
    """Run circuit using v1 (SQLite-based) implementation - inlined version."""
    n_qubits = circuit_dict["number_of_qubits"]
    gates = circuit_dict["gates"]
    
    # Create in-memory database with schema
    conn = sqlite3.connect(":memory:")
    schema = """
    CREATE TABLE IF NOT EXISTS state (
        version INTEGER NOT NULL,
        idx INTEGER NOT NULL,
        real REAL NOT NULL,
        imag REAL NOT NULL,
        PRIMARY KEY (version, idx)
    );
    
    CREATE TABLE IF NOT EXISTS gate_matrix (
        gate_name TEXT NOT NULL,
        arity INTEGER NOT NULL,
        row INTEGER NOT NULL,
        col INTEGER NOT NULL,
        real REAL NOT NULL,
        imag REAL NOT NULL,
        PRIMARY KEY (gate_name, arity, row, col)
    );
    """
    conn.executescript(schema)
    
    # Initialize |0...0> state
    conn.execute("INSERT INTO state(version, idx, real, imag) VALUES (0, 0, 1.0, 0.0);")
    conn.commit()
    
    # Register gate matrices and apply gates
    version = 0
    registered_gates = set()
    
    for gate_dict in gates:
        gate_name = gate_dict["gate"]
        qubits = tuple(gate_dict["qubits"])
        params = gate_dict.get("params", {})
        
        # Get gate tensor
        tensor, two_qubit = get_gate_tensor(gate_name, params)
        full_name = get_full_gate_name(gate_name, params)
        
        # Register gate matrix if not already registered
        if full_name not in registered_gates:
            matrix_rows = create_gate_matrix(full_name, tensor, two_qubit)
            conn.executemany(
                "INSERT OR REPLACE INTO gate_matrix(gate_name, arity, row, col, real, imag) VALUES (?, ?, ?, ?, ?, ?)",
                matrix_rows
            )
            conn.commit()
            registered_gates.add(full_name)
        
        # Apply gate
        if two_qubit:
            sql = sql_apply_two_qubit_gate(full_name, qubits[0], qubits[1], version)
        else:
            sql = sql_apply_one_qubit_gate(full_name, qubits[0], version)
        
        conn.execute("BEGIN;")
        conn.execute(f"DELETE FROM state WHERE version = {version + 1};")
        conn.executescript(sql)
        conn.commit()
        version += 1
    
    # Extract final state
    rows = conn.execute(
        "SELECT idx, real, imag FROM state WHERE version = ? ORDER BY idx", 
        (version,)
    ).fetchall()
    
    state = np.zeros(2**n_qubits, dtype=complex)
    for idx, real, imag in rows:
        state[idx] = complex(real, imag)
    
    conn.close()
    return state


# ============== V3 Implementation ==============

def run_v3_circuit(circuit_dict: dict, temp_dir: Path, test_name: str = "test") -> np.ndarray:
    """Run circuit using v3 (Spark+HiSVSIM) implementation."""
    import uuid
    from driver import SparkHiSVSIMDriver
    from v2_common.config import SimulatorConfig
    
    # Use unique run_id to avoid state reuse
    run_id = f"compare_{test_name}_{uuid.uuid4().hex[:8]}"
    
    config = SimulatorConfig(
        run_id=run_id,
        base_path=temp_dir,
        batch_size=10,
        spark_master="local[2]",
        spark_shuffle_partitions=4,
    )
    config.ensure_paths()
    
    with SparkHiSVSIMDriver(config, enable_parallel=True) as driver:
        # Disable resume to ensure fresh start
        result = driver.run_circuit(circuit_dict, n_partitions=2, resume=False)
        state = driver.get_state_vector(result)
    
    return state


# ============== Circuit Generators ==============

def generate_ghz_circuit(num_qubits: int):
    gates = [{"qubits": [0], "gate": "H"}]
    for q in range(1, num_qubits):
        gates.append({"qubits": [q - 1, q], "gate": "CNOT"})
    return {"number_of_qubits": num_qubits, "gates": gates}


def generate_qft_circuit(num_qubits: int):
    gates = []
    for j in range(num_qubits):
        gates.append({"qubits": [j], "gate": "H"})
        for k in range(j + 1, num_qubits):
            exponent = k - j + 1
            gates.append({"qubits": [k, j], "gate": "CR", "params": {"k": exponent}})
    return {"number_of_qubits": num_qubits, "gates": gates}


def generate_bell_circuit():
    return {
        "number_of_qubits": 2,
        "gates": [
            {"qubits": [0], "gate": "H"},
            {"qubits": [0, 1], "gate": "CNOT"},
        ]
    }


def generate_hadamard_wall(n_qubits: int):
    return {
        "number_of_qubits": n_qubits, 
        "gates": [{"qubits": [i], "gate": "H"} for i in range(n_qubits)]
    }


# ============== Comparison ==============

def compare_states(v1_state: np.ndarray, v3_state: np.ndarray, name: str, atol: float = 1e-10):
    """Compare two state vectors."""
    try:
        np.testing.assert_allclose(v3_state, v1_state, atol=atol, rtol=1e-10)
        print(f"✅ {name}: MATCH (max diff: {np.max(np.abs(v3_state - v1_state)):.2e})")
        return True
    except AssertionError as e:
        print(f"❌ {name}: MISMATCH")
        print(f"   v1 norm: {np.linalg.norm(v1_state):.6f}")
        print(f"   v3 norm: {np.linalg.norm(v3_state):.6f}")
        print(f"   max diff: {np.max(np.abs(v3_state - v1_state)):.2e}")
        # Show first few differences
        diff = np.abs(v3_state - v1_state)
        if np.max(diff) > atol:
            for i in range(min(5, len(diff))):
                if diff[i] > atol:
                    print(f"   idx {i}: v1={v1_state[i]:.6f}, v3={v3_state[i]:.6f}")
        return False


def main():
    print("=" * 60)
    print("V1 vs V3 Comparison Test")
    print("=" * 60)
    
    # Create temp directory for v3
    temp_dir = Path(tempfile.mkdtemp())
    
    tests = []
    
    try:
        # Test 1: Bell State
        print("\n--- Bell State ---")
        circuit = generate_bell_circuit()
        v1_state = run_v1_circuit(circuit)
        v3_state = run_v3_circuit(circuit, temp_dir, "bell")
        tests.append(("Bell State", compare_states(v1_state, v3_state, "Bell State")))
        
        # Test 2: GHZ States
        for n in [3, 4, 5]:
            print(f"\n--- GHZ-{n} ---")
            circuit = generate_ghz_circuit(n)
            v1_state = run_v1_circuit(circuit)
            v3_state = run_v3_circuit(circuit, temp_dir, f"ghz{n}")
            tests.append((f"GHZ-{n}", compare_states(v1_state, v3_state, f"GHZ-{n}")))
        
        # Test 3: QFT States  
        for n in [2, 3, 4]:
            print(f"\n--- QFT-{n} ---")
            circuit = generate_qft_circuit(n)
            v1_state = run_v1_circuit(circuit)
            v3_state = run_v3_circuit(circuit, temp_dir, f"qft{n}")
            tests.append((f"QFT-{n}", compare_states(v1_state, v3_state, f"QFT-{n}")))
        
        # Test 4: Hadamard Wall (dense state)
        for n in [3, 4, 5]:
            print(f"\n--- Hadamard Wall-{n} ---")
            circuit = generate_hadamard_wall(n)
            v1_state = run_v1_circuit(circuit)
            v3_state = run_v3_circuit(circuit, temp_dir, f"hwall{n}")
            tests.append((f"H-Wall-{n}", compare_states(v1_state, v3_state, f"H-Wall-{n}")))
        
        # Test 5: Custom circuit with multiple gate types
        print("\n--- Mixed Gates Circuit ---")
        circuit = {
            "number_of_qubits": 3,
            "gates": [
                {"qubits": [0], "gate": "H"},
                {"qubits": [1], "gate": "H"},
                {"qubits": [0, 1], "gate": "CNOT"},
                {"qubits": [1, 2], "gate": "CNOT"},
                {"qubits": [2], "gate": "T"},
                {"qubits": [0], "gate": "S"},
            ]
        }
        v1_state = run_v1_circuit(circuit)
        v3_state = run_v3_circuit(circuit, temp_dir, "mixed")
        tests.append(("Mixed Gates", compare_states(v1_state, v3_state, "Mixed Gates")))
        
        # Test 6: Non-trivial circuit
        print("\n--- Non-trivial Circuit ---")
        circuit = {
            "number_of_qubits": 3,
            "gates": [
                {"qubits": [0], "gate": "H"},
                {"qubits": [1], "gate": "H"},
                {"qubits": [2], "gate": "H"},
                {"qubits": [0, 1], "gate": "CNOT"},
                {"qubits": [1, 2], "gate": "CNOT"},
                {"qubits": [0], "gate": "T"},
                {"qubits": [1], "gate": "S"},
                {"qubits": [2], "gate": "Z"},
                {"qubits": [2, 0], "gate": "CZ"},
            ]
        }
        v1_state = run_v1_circuit(circuit)
        v3_state = run_v3_circuit(circuit, temp_dir, "nontrivial")
        tests.append(("Non-trivial", compare_states(v1_state, v3_state, "Non-trivial")))
        
        # Test 7: 6-qubit GHZ (larger scale)
        print("\n--- GHZ-6 (larger scale) ---")
        circuit = generate_ghz_circuit(6)
        v1_state = run_v1_circuit(circuit)
        v3_state = run_v3_circuit(circuit, temp_dir, "ghz6")
        tests.append(("GHZ-6", compare_states(v1_state, v3_state, "GHZ-6")))
        
        # Test 8: 5-qubit QFT
        print("\n--- QFT-5 ---")
        circuit = generate_qft_circuit(5)
        v1_state = run_v1_circuit(circuit)
        v3_state = run_v3_circuit(circuit, temp_dir, "qft5")
        tests.append(("QFT-5", compare_states(v1_state, v3_state, "QFT-5")))
        
        # Test 9: CZ and SWAP gates
        print("\n--- CZ/SWAP Circuit ---")
        circuit = {
            "number_of_qubits": 3,
            "gates": [
                {"qubits": [0], "gate": "H"},
                {"qubits": [0, 1], "gate": "CZ"},
                {"qubits": [1], "gate": "H"},
                {"qubits": [1, 2], "gate": "SWAP"},
                {"qubits": [0, 2], "gate": "CNOT"},
            ]
        }
        v1_state = run_v1_circuit(circuit)
        v3_state = run_v3_circuit(circuit, temp_dir, "czswap")
        tests.append(("CZ/SWAP", compare_states(v1_state, v3_state, "CZ/SWAP")))
        
        # Test 10: Y and Z gates
        print("\n--- Y/Z Gates ---")
        circuit = {
            "number_of_qubits": 2,
            "gates": [
                {"qubits": [0], "gate": "H"},
                {"qubits": [1], "gate": "H"},
                {"qubits": [0], "gate": "Y"},
                {"qubits": [1], "gate": "Z"},
                {"qubits": [0, 1], "gate": "CY"},
            ]
        }
        v1_state = run_v1_circuit(circuit)
        v3_state = run_v3_circuit(circuit, temp_dir, "yz")
        tests.append(("Y/Z Gates", compare_states(v1_state, v3_state, "Y/Z Gates")))
        
        # Summary
        print("\n" + "=" * 60)
        print("SUMMARY")
        print("=" * 60)
        passed = sum(1 for _, result in tests if result)
        total = len(tests)
        print(f"\nPassed: {passed}/{total}")
        
        if passed == total:
            print("\n🎉 ALL TESTS PASSED - V3 matches V1 exactly!")
        else:
            print("\n⚠️  Some tests failed")
            for name, result in tests:
                status = "✅" if result else "❌"
                print(f"  {status} {name}")
        
        return passed == total
        
    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
