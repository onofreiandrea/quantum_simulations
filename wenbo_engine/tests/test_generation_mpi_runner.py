"""Integration tests for --recovery=generation through the MPI runner.

These run in a single MPI process (COMM_WORLD size 1), which exercises the
full path: _run_generation -> _apply_step kernels -> commit protocol ->
RecoveryScanner resume.  Multi-rank behaviour is covered by the pure-Python
recovery unit tests (test_generation_recovery.py); here we verify the runner
wiring + numerical correctness + crash/resume.
"""
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

mpi4py = pytest.importorskip("mpi4py")
from mpi4py import MPI  # noqa: E402

from wenbo_engine.kernel.ref_dense import simulate  # noqa: E402
from wenbo_engine.mpi import mpi_runner  # noqa: E402
from wenbo_engine.recovery import RecoveryScanner, commits_dir  # noqa: E402

pytestmark = pytest.mark.skipif(
    MPI.COMM_WORLD.Get_size() != 1,
    reason="in-process generation tests assume a single MPI rank",
)


def _circuit():
    # 3 qubits, several levels so there is >1 generation to commit.
    return {"number_of_qubits": 3, "gates": [
        {"qubits": [0], "gate": "H"},
        {"qubits": [1], "gate": "H"},
        {"qubits": [0, 1], "gate": "CNOT"},
        {"qubits": [2], "gate": "X"},
        {"qubits": [1, 2], "gate": "CNOT"},
    ]}


def test_generation_run_matches_reference():
    cd = _circuit()
    ref = simulate(cd)
    cs = 1 << cd["number_of_qubits"]   # whole state in one chunk (1 rank)
    with tempfile.TemporaryDirectory() as td:
        final = mpi_runner.run(cd, td, chunk_size=cs, recovery="generation")
        assert "gen_" in final.name
        state = mpi_runner.collect_state(td)
        np.testing.assert_allclose(state, ref, atol=1e-6)
        # run.json records the generation mode
        meta = json.loads((Path(td) / "run.json").read_text())
        assert meta["recovery_mode"] == "generation"
        # at least one global commit record exists
        assert list((Path(td) / "commits").glob("commit_*.json"))


def test_generation_run_is_resumable_idempotent():
    cd = _circuit()
    ref = simulate(cd)
    cs = 1 << cd["number_of_qubits"]
    with tempfile.TemporaryDirectory() as td:
        mpi_runner.run(cd, td, chunk_size=cs, recovery="generation")
        gen_after_first = RecoveryScanner(td).scan(quarantine=False).generation
        # Re-running resumes from the committed final generation (no-op steps).
        mpi_runner.run(cd, td, chunk_size=cs, recovery="generation")
        gen_after_second = RecoveryScanner(td).scan(quarantine=False).generation
        assert gen_after_second == gen_after_first
        np.testing.assert_allclose(mpi_runner.collect_state(td), ref, atol=1e-6)


_CRASH_SCRIPT = '''
import sys, json
sys.path.insert(0, "{repo_root}")
from wenbo_engine.mpi import mpi_runner
cd = json.loads('{cd_json}')
mpi_runner.run(cd, "{work_dir}", chunk_size={cs}, recovery="generation")
'''


def test_generation_crash_after_commit_then_resume():
    """Crash after a committed step, then resume → correct final state."""
    cd = _circuit()
    ref = simulate(cd)
    cs = 1 << cd["number_of_qubits"]
    repo_root = str(Path(__file__).resolve().parent.parent.parent)

    with tempfile.TemporaryDirectory() as td:
        script = _CRASH_SCRIPT.format(
            repo_root=repo_root, cd_json=json.dumps(cd), work_dir=td, cs=cs)
        env = os.environ.copy()
        env["WE_CRASH_AFTER_STEP"] = "1"   # exit after committing generation 1
        result = subprocess.run([sys.executable, "-c", script],
                                env=env, capture_output=True, timeout=60)
        assert result.returncode != 0, result.stderr.decode()[-2000:]

        # A commit record for generation 1 must exist (crash was post-commit).
        assert (commits_dir(td) / "commit_000001.json").exists()
        scanned = RecoveryScanner(td).scan(quarantine=False)
        assert scanned.generation == 1

        # Resume to completion (no crash env) → correct result.
        final = mpi_runner.run(cd, td, chunk_size=cs, recovery="generation")
        np.testing.assert_allclose(mpi_runner.collect_state(td), ref, atol=1e-6)
        assert "gen_" in final.name


# ── Check 2: all-rank agreement after each committed generation (real MPI) ──

import shutil  # noqa: E402

_AGREEMENT_SCRIPT = r'''
import sys, json
sys.path.insert(0, "{repo_root}")
from mpi4py import MPI
from wenbo_engine.mpi import mpi_runner
from wenbo_engine.recovery import RecoveryScanner, gen_dir
from wenbo_engine.recovery.rank_manifest import RankManifest

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
cd = json.loads('{cd_json}')
work = "{work_dir}"
mpi_runner.run(cd, work, chunk_size={cs}, recovery="generation", comm=comm)
comm.Barrier()

# Each rank independently determines the committed generation + lineage it sees.
res = RecoveryScanner(work).scan(quarantine=False)
gen = res.generation
man = RankManifest.read(gen_dir(work, rank, gen))
view = {{
    "committed_generation": gen,
    "parent_generation": man.parent_generation,
    "stage_id": man.stage_id,
    "commit_hash": res.record.commit_hash,
}}
views = comm.gather(view, root=0)
if rank == 0:
    ok = all(v == views[0] for v in views)
    print("VIEWS", json.dumps(views))
    print("AGREE" if ok else "DISAGREE")
    sys.exit(0 if ok else 2)
'''


def test_all_rank_agreement_under_mpi(tmp_path):
    """After committing, all ranks agree on gen id, parent, stage, commit hash."""
    if not shutil.which("mpirun"):
        pytest.skip("mpirun not available")
    cd = _circuit()
    cs = 1 << (cd["number_of_qubits"] - 1)   # 2 chunks total -> 1 chunk/rank @ 2 ranks
    repo_root = str(Path(__file__).resolve().parent.parent.parent)
    work = str(tmp_path / "agree")
    script = tmp_path / "agree_check.py"
    script.write_text(_AGREEMENT_SCRIPT.format(
        repo_root=repo_root, cd_json=json.dumps(cd), work_dir=work, cs=cs))

    result = subprocess.run(
        ["mpirun", "-np", "2", sys.executable, str(script)],
        capture_output=True, text=True, timeout=120)
    out = result.stdout + result.stderr
    assert result.returncode == 0, out
    assert "AGREE" in result.stdout, out
    assert "DISAGREE" not in result.stdout, out
