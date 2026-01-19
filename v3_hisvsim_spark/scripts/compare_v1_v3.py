#!/usr/bin/env python3
"""
Direct comparison script: Run v1 and v3 and compare results.

This script directly compares v1 and v3 implementations to verify correctness.
"""
import sys
import sqlite3
import numpy as np
from pathlib import Path

# Add paths
ROOT = Path(__file__).parent.parent.parent
V1_IMPL = ROOT / "v1_implementation"
V3_SPARK = ROOT / "v3_hisvsim_spark"

sys.path.insert(0, str(V3_SPARK / "src"))

# Import v3
from driver import SparkHiSVSIMDriver
from v2_common import config
import tempfile
import shutil
import subprocess
import json

SimulatorConfig = config.SimulatorConfig

# Helper to run v1 via subprocess
def run_v1_helper(circuit_name, n_qubits):
    """Run v1 simulation via helper script."""
    script_path = Path(__file__).parent / "run_v1_helper.py"
    result = subprocess.run(
        [sys.executable, str(script_path), circuit_name, str(n_qubits)],
        capture_output=True,
        text=True,
        cwd=str(ROOT)
    )
    if result.returncode != 0:
        raise RuntimeError(f"V1 simulation failed: {result.stderr}")
    
    data = json.loads(result.stdout)
    n_qubits = data["n_qubits"]
    state = np.zeros(2**n_qubits, dtype=complex)
    for item in data["state"]:
        state[item["idx"]] = complex(item["real"], item["imag"])
    return state

# Import v3 circuit generators (they should match v1)
from v2_common import circuits
generate_ghz_v1 = circuits.generate_ghz_circuit
generate_qft_v1 = circuits.generate_qft_circuit
generate_w_v1 = circuits.generate_w_circuit

# Import v3
from driver import SparkHiSVSIMDriver
from v2_common import config
import tempfile
import shutil

SimulatorConfig = config.SimulatorConfig
SCHEMA = V1_IMPL / "sql" / "schema.sql"


def run_v1(circuit_dict):
    """Run v1 simulation."""
    # Determine circuit name and n_qubits from circuit_dict
    n_qubits = circuit_dict["number_of_qubits"]
    gates = circuit_dict["gates"]
    
    # Try to identify circuit type
    if len(gates) == n_qubits and all(g["gate"] == "H" for g in gates):
        circuit_name = "QFT"  # QFT starts with H gates
    elif gates[0]["gate"] == "H" and all(g["gate"] == "CNOT" for g in gates[1:]):
        circuit_name = "GHZ"
    elif gates[0]["gate"] == "X":
        circuit_name = "W"
    else:
        # Default to GHZ for now
        circuit_name = "GHZ"
    
    return run_v1_helper(circuit_name, n_qubits)


def run_v3(circuit_dict, n_partitions=2, enable_parallel=False):
    """Run v3 simulation."""
    temp_dir = Path(tempfile.mkdtemp())
    cfg = SimulatorConfig(
        run_id="compare_v1_v3",
        base_path=temp_dir,
        batch_size=10,
        spark_master="local[2]",
        spark_shuffle_partitions=4,
    )
    cfg.ensure_paths()
    
    try:
        with SparkHiSVSIMDriver(cfg, enable_parallel=enable_parallel) as driver:
            result = driver.run_circuit(circuit_dict, n_partitions=n_partitions, enable_parallel=enable_parallel)
            state = driver.get_state_vector(result)
        return state
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def compare_circuits(name, circuit_v1, n_partitions=2):
    """Compare v1 and v3 results for a circuit."""
    print(f"\n{'='*60}")
    print(f"Testing: {name}")
    print(f"{'='*60}")
    
    # Run v1
    print("Running v1...")
    v1_state = run_v1(circuit_v1)
    v1_nonzero = np.sum(np.abs(v1_state) > 1e-10)
    v1_norm = np.linalg.norm(v1_state)
    print(f"  Non-zero amplitudes: {v1_nonzero}")
    print(f"  Norm: {v1_norm:.10f}")
    
    # Run v3 sequential
    print("Running v3 (sequential)...")
    v3_seq_state = run_v3(circuit_v1, n_partitions=n_partitions, enable_parallel=False)
    v3_seq_nonzero = np.sum(np.abs(v3_seq_state) > 1e-10)
    v3_seq_norm = np.linalg.norm(v3_seq_state)
    print(f"  Non-zero amplitudes: {v3_seq_nonzero}")
    print(f"  Norm: {v3_seq_norm:.10f}")
    
    # Run v3 parallel
    print("Running v3 (parallel)...")
    v3_par_state = run_v3(circuit_v1, n_partitions=n_partitions, enable_parallel=True)
    v3_par_nonzero = np.sum(np.abs(v3_par_state) > 1e-10)
    v3_par_norm = np.linalg.norm(v3_par_state)
    print(f"  Non-zero amplitudes: {v3_par_nonzero}")
    print(f"  Norm: {v3_par_norm:.10f}")
    
    # Compare
    print("\nComparison:")
    max_diff_v1_v3_seq = np.max(np.abs(v1_state - v3_seq_state))
    max_diff_v1_v3_par = np.max(np.abs(v1_state - v3_par_state))
    max_diff_seq_par = np.max(np.abs(v3_seq_state - v3_par_state))
    
    print(f"  v1 vs v3 (sequential): max_diff = {max_diff_v1_v3_seq:.2e}")
    print(f"  v1 vs v3 (parallel):   max_diff = {max_diff_v1_v3_par:.2e}")
    print(f"  v3 seq vs v3 par:      max_diff = {max_diff_seq_par:.2e}")
    
    # Check if they match
    tolerance = 1e-10
    v1_v3_seq_match = max_diff_v1_v3_seq < tolerance
    v1_v3_par_match = max_diff_v1_v3_par < tolerance
    seq_par_match = max_diff_seq_par < tolerance
    
    print(f"\nResults:")
    print(f"  v1 == v3 (sequential): {'✓ PASS' if v1_v3_seq_match else '✗ FAIL'}")
    print(f"  v1 == v3 (parallel):   {'✓ PASS' if v1_v3_par_match else '✗ FAIL'}")
    print(f"  v3 seq == v3 par:      {'✓ PASS' if seq_par_match else '✗ FAIL'}")
    
    if not (v1_v3_seq_match and v1_v3_par_match and seq_par_match):
        print("\n⚠️  MISMATCH DETECTED!")
        # Show differences
        diff_indices = np.where(np.abs(v1_state - v3_seq_state) > tolerance)[0]
        if len(diff_indices) > 0:
            print(f"  First 5 differing indices:")
            for idx in diff_indices[:5]:
                print(f"    idx={idx}: v1={v1_state[idx]}, v3_seq={v3_seq_state[idx]}, diff={np.abs(v1_state[idx] - v3_seq_state[idx]):.2e}")
        return False
    
    return True


def main():
    """Run all comparisons."""
    print("="*60)
    print("V3 vs V1 Direct Comparison")
    print("="*60)
    
    results = []
    
    # Test GHZ
    for n in [2, 3, 4]:
        circuit = generate_ghz_v1(n)
        passed = compare_circuits(f"GHZ({n})", circuit, n_partitions=2)
        results.append(("GHZ", n, passed))
    
    # Test QFT
    for n in [2, 3, 4]:
        circuit = generate_qft_v1(n)
        passed = compare_circuits(f"QFT({n})", circuit, n_partitions=3)
        results.append(("QFT", n, passed))
    
    # Test W-state
    for n in [3, 4]:
        circuit = generate_w_v1(n)
        passed = compare_circuits(f"W({n})", circuit, n_partitions=2)
        results.append(("W", n, passed))
    
    # Summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    passed_count = sum(1 for _, _, p in results if p)
    total_count = len(results)
    
    for name, n, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {name}({n}): {status}")
    
    print(f"\nTotal: {passed_count}/{total_count} passed")
    
    if passed_count == total_count:
        print("\n✅ All tests passed! V3 matches V1 exactly.")
        return 0
    else:
        print(f"\n❌ {total_count - passed_count} test(s) failed!")
        return 1


if __name__ == "__main__":
    import os
    os.environ.setdefault("JAVA_HOME", "/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home")
    sys.exit(main())
