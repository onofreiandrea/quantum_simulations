"""End-to-end tests for the single-node experiment harness + instrumented runner."""
from __future__ import annotations

import csv
import json

import numpy as np

from wenbo_engine.experiments.config import ExperimentConfig
from wenbo_engine.experiments.run_experiment import run_experiment, compile_plan
from wenbo_engine.experiments import instrumented_runner as ir
from wenbo_engine.kernel.ref_dense import simulate
from wenbo_engine.storage.manifest import read_manifest
from wenbo_engine.storage.block_store import read_chunk
from wenbo_engine.profiling import STAGE_COLUMNS
from wenbo_engine.tests.fixtures.circuits import ghz, qft, bell_2q


REQUIRED_ARTIFACTS = [
    "config.json", "circuit.json", "plan.json", "cost_model.json",
    "stage_profile.csv", "io_profile.csv", "mpi_profile.csv",
    "recovery_events.json", "final_summary.json", "final_norm.txt",
    "git_commit.txt",
]


def _collect(final_dir):
    man = read_manifest(final_dir)
    parts = [read_chunk(final_dir / "chunks" / c) for c in man.chunks]
    return np.concatenate(parts).astype(np.complex128)


def test_experiment_creates_all_artifacts(tmp_path):
    cfg = ExperimentConfig.from_dict({
        "run_id": "t_local",
        "runner": "single_node",
        "circuit": {"source": "builtin", "name": "ghz", "n_qubits": 5},
        "chunk_bits": 3,
        "checksum": True,
        "calib_chunks": 2,
        "output_dir": str(tmp_path / "experiments"),
    })
    run_dir = run_experiment(cfg)

    for name in REQUIRED_ARTIFACTS:
        assert (run_dir / name).exists(), f"missing artifact: {name}"

    # final norm ~ 1
    norm = float((run_dir / "final_norm.txt").read_text().strip())
    assert abs(norm - 1.0) < 1e-6

    # stage csv schema
    with open(run_dir / "stage_profile.csv", newline="") as f:
        reader = csv.DictReader(f)
        assert reader.fieldnames == STAGE_COLUMNS
        rows = list(reader)
    assert len(rows) >= 1
    # checksum column exercised (checksum=True)
    assert any(float(r["checksum_sec"]) >= 0 for r in rows)

    # summary sane
    summary = json.loads((run_dir / "final_summary.json").read_text())
    assert summary["n_qubits"] == 5
    assert summary["runner"] == "single_node"
    assert summary["n_stages"] == len(rows)


def test_plan_matches_levels(tmp_path):
    cd = qft(5)
    plan = compile_plan(cd, chunk_size=8, n_ranks=1)
    assert plan["n_qubits"] == 5
    assert plan["n_chunks"] == (1 << 5) // 8
    # every gate accounted for exactly once
    tot = plan["totals"]
    assert (tot["local_ops"] + tot["rank_nonlocal_ops"]
            + tot["mpi_nonlocal_ops"]) == len(cd["gates"])


def test_instrumented_runner_matches_reference(tmp_path):
    for circ in (bell_2q(), ghz(5), qft(4)):
        work = tmp_path / f"w{circ['number_of_qubits']}_{len(circ['gates'])}"
        res = ir.run(circ, work, chunk_size=4, use_wal=True)
        got = _collect(res.final_dir)
        want = simulate(circ)
        assert np.allclose(got, want, atol=1e-6), \
            f"state mismatch for {circ['number_of_qubits']}q"


def test_instrumented_runner_wal_resume(tmp_path):
    cd = ghz(4)
    work = tmp_path / "w"
    res1 = ir.run(cd, work, chunk_size=4, use_wal=True)
    assert res1.start_step == 0
    assert not res1.resumed
    # second run on the same work dir: WAL says all steps done -> resume, no-op
    res2 = ir.run(cd, work, chunk_size=4, use_wal=True)
    assert res2.resumed
    assert res2.start_step == res2.n_steps
    got = _collect(res2.final_dir)
    assert np.allclose(got, simulate(cd), atol=1e-6)


def test_experiment_from_json_circuit(tmp_path):
    cd = ghz(4)
    circ_path = tmp_path / "circ.json"
    circ_path.write_text(json.dumps(cd))
    cfg = ExperimentConfig.from_dict({
        "run_id": "t_json",
        "runner": "single_node",
        "circuit": {"source": "json", "path": str(circ_path)},
        "chunk_bits": 2,
        "calibrate": False,
        "output_dir": str(tmp_path / "experiments"),
    })
    run_dir = run_experiment(cfg)
    norm = float((run_dir / "final_norm.txt").read_text().strip())
    assert abs(norm - 1.0) < 1e-6
    # calibrate=False still writes a cost_model.json marker
    cm = json.loads((run_dir / "cost_model.json").read_text())
    assert cm.get("calibrated") is False
