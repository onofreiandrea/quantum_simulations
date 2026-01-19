#!/usr/bin/env python3
"""
Helper script to run v1 simulation and return results as JSON.
"""
import sys
import sqlite3
import json
import numpy as np
from pathlib import Path

# Add v1 to path
ROOT = Path(__file__).parent.parent.parent
V1_IMPL = ROOT / "v1_implementation"
V1_SRC = V1_IMPL / "src"
sys.path.insert(0, str(V1_SRC))

# Set up package structure for relative imports
import types
import importlib.util

# Create package modules
v1_pkg = types.ModuleType('v1_implementation')
v1_src_pkg = types.ModuleType('v1_implementation.src')
sys.modules['v1_implementation'] = v1_pkg
sys.modules['v1_implementation.src'] = v1_src_pkg

# Import modules in dependency order
spec_db = importlib.util.spec_from_file_location("db", V1_SRC / "db.py")
db = importlib.util.module_from_spec(spec_db)
sys.modules['v1_implementation.src.db'] = db
spec_db.loader.exec_module(db)

spec_gates = importlib.util.spec_from_file_location("gates", V1_SRC / "gates.py")
gates_module = importlib.util.module_from_spec(spec_gates)
sys.modules['v1_implementation.src.gates'] = gates_module
spec_gates.loader.exec_module(gates_module)

spec_frontend = importlib.util.spec_from_file_location("frontend", V1_SRC / "frontend.py")
frontend_module = importlib.util.module_from_spec(spec_frontend)
sys.modules['v1_implementation.src.frontend'] = frontend_module
spec_frontend.loader.exec_module(frontend_module)

spec_gate_loader = importlib.util.spec_from_file_location("gate_loader", V1_SRC / "gate_loader.py")
gate_loader_module = importlib.util.module_from_spec(spec_gate_loader)
sys.modules['v1_implementation.src.gate_loader'] = gate_loader_module
spec_gate_loader.loader.exec_module(gate_loader_module)

spec_checkpoint = importlib.util.spec_from_file_location("checkpoint", V1_SRC / "checkpoint.py")
checkpoint_module = importlib.util.module_from_spec(spec_checkpoint)
sys.modules['v1_implementation.src.checkpoint'] = checkpoint_module
spec_checkpoint.loader.exec_module(checkpoint_module)

spec_state = importlib.util.spec_from_file_location("state_manager", V1_SRC / "state_manager.py")
state_module = importlib.util.module_from_spec(spec_state)
sys.modules['v1_implementation.src.state_manager'] = state_module
spec_state.loader.exec_module(state_module)

spec_circuits = importlib.util.spec_from_file_location("circuits", V1_SRC / "circuits.py")
circuits_module = importlib.util.module_from_spec(spec_circuits)
sys.modules['v1_implementation.src.circuits'] = circuits_module
spec_circuits.loader.exec_module(circuits_module)

spec_sim = importlib.util.spec_from_file_location("simulator", V1_SRC / "simulator.py")
sim_module = importlib.util.module_from_spec(spec_sim)
sys.modules['v1_implementation.src.simulator'] = sim_module
spec_sim.loader.exec_module(sim_module)

# Now import functions
run_circuit = sim_module.run_circuit
fetch_state = state_module.fetch_state
generate_ghz_circuit = circuits_module.generate_ghz_circuit
generate_qft_circuit = circuits_module.generate_qft_circuit
generate_w_circuit = circuits_module.generate_w_circuit

try:
    
    # Get circuit name and params from command line
    circuit_name = sys.argv[1]
    n_qubits = int(sys.argv[2])
    
    # Generate circuit
    if circuit_name == "GHZ":
        circuit = generate_ghz_circuit(n_qubits)
    elif circuit_name == "QFT":
        circuit = generate_qft_circuit(n_qubits)
    elif circuit_name == "W":
        circuit = generate_w_circuit(n_qubits)
    else:
        raise ValueError(f"Unknown circuit: {circuit_name}")
    
    # Run simulation
    conn = sqlite3.connect(":memory:")
    schema_path = V1_IMPL / "sql" / "schema.sql"
    db.initialize_schema(conn, str(schema_path))
    version = run_circuit(conn, circuit)
    
    # Get state
    n_qubits = circuit["number_of_qubits"]
    state = np.zeros(2**n_qubits, dtype=complex)
    rows = fetch_state(conn, version)
    for row in rows:
        idx, real, imag = row[0], row[1], row[2]
        state[idx] = complex(real, imag)
    
    conn.close()
    
    # Output as JSON
    result = {
        "n_qubits": n_qubits,
        "state": [{"idx": int(idx), "real": float(state[idx].real), "imag": float(state[idx].imag)} 
                  for idx in range(len(state)) if abs(state[idx]) > 1e-10]
    }
    print(json.dumps(result))
    
except Exception as e:
    import traceback
    print(json.dumps({"error": str(e), "traceback": traceback.format_exc()}))
    sys.exit(1)
