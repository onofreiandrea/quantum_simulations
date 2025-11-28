# Phase 1 SQL simulator

## Quick start

```bash
pip install -r v1_implementation/requirements.txt
BENCHMARK_PARITY_MAX_QUBITS=12 pytest v1_implementation/tests   # fast run
pytest v1_implementation/tests                                # full sweep
```

For the original benchmarking workflow open `v1_implementation/benchmark.ipynb`, run all cells, and it will reproduce the GHZ/W/QFT/QPE sweeps (with optional comparisons against the legacy implementation).

---

## How a circuit runs

1. Circuit JSON → `(n_qubits, list[Gate])` via `frontend.circuit_dict_to_gates`.
2. `simulator.run_circuit` initializes version `0` (`|0…0⟩`) and registers each unique gate tensor in `gate_matrix`.
3. For **every gate**:
   - `state_manager.apply_gate_atomic` inserts a `PENDING` WAL row.
   - `gate_translator.translate_gate(gate, version)` emits SQL that reads `state@version`, joins with `gate_matrix`, and writes `state@version+1`. This SQL only touches the current gate.
   - SQL runs inside a transaction; once it succeeds we mark the WAL entry `COMMITTED`.
4. After the final gate we can `create_checkpoint` to snapshot the last version.

---

## Modules

- `src/gates.py` – numpy tensors for every 1-/2-qubit gate (H, X, CZ, CRk, CU, …).
- `src/circuits.py` – GHZ, QFT, QPE, W, GHZ-QFT, etc., identical to the original repo.
- `src/frontend.py` – dispatcher that converts the JSON format into our `Gate` objects.
- `src/gate_loader.py` – writes flattened gate tensors into the `gate_matrix` table.
- `src/gate_translator.py` – SQL templates for arbitrary 1- or 2-qubit gates.
- `src/state_manager.py` – initializes the state and performs the WAL + per-gate SQL dance.
- `src/wal.py`, `src/checkpoint.py`, `src/recovery.py` – bookkeeping, exporting, replay.
- `src/simulator.py` – glue layer that ties the whole flow together.
- `sql/schema.sql` – schemas for `state`, `gate_matrix`, `wal`, `checkpoint`.
- `tests/` – basic GHZ/QFT sanity tests plus a parity suite that mirrors every circuit size from `Quantum/benchmark.ipynb`.
- `benchmark.ipynb` – notebook clone of the original benchmark notebook, now driving this implementation.

---

## WAL, checkpoints, recovery

### WAL

- Before applying a gate we insert `version`, `gate_name`, `qubits`, `params`, `status='PENDING'` into `wal`.
- After the gate’s SQL succeeds we flip that row to `COMMITTED`.
- There is exactly one WAL row per gate, in order, so we always know which gate produced which state version.

### Checkpoints

- At the end of a run we export `state` rows for the final version to `checkpoints/<run>/state_vX.csv` and record `(version, path, checksum, timestamp)` in the `checkpoint` table.
- CSV keeps Phase 1 simple, but for Phase 2 I would change to Parquet/DuckDB exports.

### Recovery

1. Load the latest checkpoint entry; if none exists assume version `0` still exists.
2. Restore the checkpoint file into the `state` table (delete existing rows, bulk insert the snapshot).
3. Replay the remaining gates by re-running `apply_gate_atomic` for every gate whose index is greater than the checkpoint version.

For now the replay step still reads from the original circuit JSON (or generator). That satisfies my idea for Phase 1, but it means recovery requires the original circuit description to be available. A future enhancement is to serialize enough info into WAL rows so we can reconstruct gates directly from the database.

---

Future phases (batched SQL, alternate storage engines, extra gates) can hook in at the translator/state manager layer without touching the frontend.

