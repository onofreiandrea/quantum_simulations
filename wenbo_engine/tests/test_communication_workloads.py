"""Tests for the MPI-nonlocal communication benchmark suite.

Pure (no-MPI) tests cover the generators, the static classifier, and the
*real runner* classifier (mpi_runner._compile_steps).  The end-to-end
tests launch real ``mpirun`` jobs and check state against ``ref_dense``,
that MPI traffic is actually measured, that the artifact bundle is written,
and that metrics do not leak between sequential in-process runs.
"""
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from wenbo_engine.bench import communication_workloads as cw
from wenbo_engine.bench.communication_workloads import (
    classify_circuit,
    runner_classification,
    communication_light,
    mixed_staged,
    mpi_nonlocal_heavy,
    rank_nonlocal_heavy,
    build_circuit,
)
from wenbo_engine.circuit.io import validate_circuit_dict

REPO_ROOT = Path(__file__).resolve().parents[2]

HAS_MPI = shutil.which("mpirun") is not None
try:
    import mpi4py  # noqa: F401
    HAS_MPI4PY = True
except ImportError:
    HAS_MPI4PY = False

mpi_required = pytest.mark.skipif(
    not (HAS_MPI and HAS_MPI4PY), reason="mpirun / mpi4py not available")


def _all_qubits(cd):
    return [q for g in cd["gates"] for q in g["qubits"]]


def _env():
    env = dict(os.environ)
    env["PYTHONPATH"] = str(REPO_ROOT) + os.pathsep + env.get("PYTHONPATH", "")
    return env


# ── 1. determinism ──────────────────────────────────────────────────────

@pytest.mark.parametrize("kind", list(cw.GENERATORS))
def test_generator_deterministic(kind):
    a = build_circuit(kind, n=16, depth=30, chunk_bits=8, num_ranks=4, seed=7)
    b = build_circuit(kind, n=16, depth=30, chunk_bits=8, num_ranks=4, seed=7)
    c = build_circuit(kind, n=16, depth=30, chunk_bits=8, num_ranks=4, seed=8)
    dump = lambda cd: json.dumps(cd, sort_keys=True)
    assert dump(a) == dump(b), "same seed must produce identical circuit"
    assert dump(a) != dump(c), "different seed must produce different circuit"


@pytest.mark.parametrize("kind", list(cw.GENERATORS))
def test_generated_circuits_valid(kind):
    cd = build_circuit(kind, n=16, depth=30, chunk_bits=8, num_ranks=4, seed=1)
    validate_circuit_dict(cd)
    assert len(cd["gates"]) >= 1


# ── 2. workload meaning (static) ────────────────────────────────────────

def test_communication_light_is_low_bit():
    n, depth = 16, 60
    cd = communication_light(n, depth, seed=3)
    low = max(1, n // 4)
    assert all(q < low for q in _all_qubits(cd))
    info = classify_circuit(cd, chunk_bits=8, num_ranks=4)
    assert info["mpi_nonlocal_gate_count"] == 0
    assert info["rank_nonlocal_gate_count"] == 0
    assert info["local_gate_count"] == depth


def test_mpi_nonlocal_heavy_has_rank_bit_gates():
    n, depth, chunk_bits, num_ranks = 12, 40, 8, 4
    cd = mpi_nonlocal_heavy(n, depth, chunk_bits, num_ranks, seed=5)
    info = classify_circuit(cd, chunk_bits, num_ranks)
    assert info["mpi_nonlocal_gate_count"] == depth
    p = 2
    assert any(q >= n - p for q in _all_qubits(cd))
    assert info["partner_rank_pairs"] > 0


# ── 3. classification under multiple num_ranks + chunk_bits ─────────────
#     (verified against BOTH the static classifier and the real runner)

# (n, chunk_bits) layouts that stay valid for num_ranks up to 8.
_LAYOUTS = [(16, 8), (16, 10), (18, 12)]
_RANKS = [2, 4, 8]


@pytest.mark.parametrize("num_ranks", _RANKS)
@pytest.mark.parametrize("n,chunk_bits", _LAYOUTS)
def test_classification_matches_real_runner(n, chunk_bits, num_ranks):
    """Static classify_circuit must agree with the real runner compiler."""
    for kind in cw.GENERATORS:
        cd = build_circuit(kind, n, depth=30, chunk_bits=chunk_bits,
                           num_ranks=num_ranks, seed=11)
        stat = classify_circuit(cd, chunk_bits, num_ranks)
        run = runner_classification(cd, chunk_bits, num_ranks)
        assert run["local_ops"] == stat["local_gate_count"], kind
        assert run["rank_nonlocal_ops"] == stat["rank_nonlocal_gate_count"], kind
        assert run["mpi_nonlocal_ops"] == stat["mpi_nonlocal_gate_count"], kind


@pytest.mark.parametrize("num_ranks", _RANKS)
@pytest.mark.parametrize("n,chunk_bits", _LAYOUTS)
def test_intended_locality_holds_real_runner(n, chunk_bits, num_ranks):
    """Each workload hits its intended class under the real runner compiler."""
    depth = 30
    # communication_light: zero MPI
    cl = runner_classification(
        communication_light(n, depth, 1), chunk_bits, num_ranks)
    assert cl["mpi_nonlocal_ops"] == 0

    # rank_nonlocal_heavy: rank-nonlocal > 0, MPI == 0 for the intended ranks
    rn = runner_classification(
        build_circuit("rank_nonlocal_heavy", n, depth, chunk_bits, num_ranks, 1),
        chunk_bits, num_ranks)
    assert rn["rank_nonlocal_ops"] > 0
    assert rn["mpi_nonlocal_ops"] == 0

    # mpi_nonlocal_heavy: MPI > 0
    mp = runner_classification(
        mpi_nonlocal_heavy(n, depth, chunk_bits, num_ranks, 1),
        chunk_bits, num_ranks)
    assert mp["mpi_nonlocal_ops"] > 0

    # mixed_staged: all three classes present
    mx = runner_classification(
        mixed_staged(n, depth, chunk_bits, num_ranks, 1), chunk_bits, num_ranks)
    assert mx["local_ops"] > 0
    assert mx["rank_nonlocal_ops"] > 0
    assert mx["mpi_nonlocal_ops"] > 0


def test_rank_nonlocal_generalizes_beyond_4_ranks():
    """rank_nonlocal_heavy must stay MPI-free even at num_ranks=8."""
    n, depth, chunk_bits = 16, 40, 8
    for num_ranks in (2, 4, 8):
        cd = rank_nonlocal_heavy(n, depth, chunk_bits, seed=2, num_ranks=num_ranks)
        info = classify_circuit(cd, chunk_bits, num_ranks)
        assert info["rank_nonlocal_gate_count"] == depth, num_ranks
        assert info["mpi_nonlocal_gate_count"] == 0, num_ranks


# ── 6. monkeypatch/proxy safety (in-process, no MPI) ────────────────────

def test_instrument_restores_even_on_failure():
    from wenbo_engine.mpi import mpi_runner as mr
    names = ("read_chunk", "write_chunk_atomic", "apply_1q", "apply_2q",
             "apply_1q_pair", "apply_2q_pair_qa_local",
             "apply_2q_pair_qb_local", "apply_2q_quad")
    before = {nm: getattr(mr, nm) for nm in names}

    # normal exit restores
    with cw._instrument_runner(cw.Metrics()):
        assert getattr(mr, "read_chunk") is not before["read_chunk"]
    for nm in names:
        assert getattr(mr, nm) is before[nm], f"{nm} not restored after normal exit"

    # exception inside the context still restores
    with pytest.raises(RuntimeError):
        with cw._instrument_runner(cw.Metrics()):
            raise RuntimeError("boom")
    for nm in names:
        assert getattr(mr, nm) is before[nm], f"{nm} not restored after exception"


def test_metrics_instances_are_independent():
    """Two Metrics never share counters (fresh per run -> no leak)."""
    m1, m2 = cw.Metrics(), cw.Metrics()
    m1.mpi_bytes_sent += 100
    m1.observed_partner_pairs.add(frozenset((0, 1)))
    assert m2.mpi_bytes_sent == 0
    assert m2.observed_partner_pairs == set()


# ── 4/5/8: real MPI execution ───────────────────────────────────────────

def _run_cli(tmp, kind, n, depth, chunk_bits, np_ranks, output_dir=None,
             verify=True, extra=None):
    out = Path(tmp) / "profile.json"
    cmd = [
        "mpirun", "--oversubscribe", "-np", str(np_ranks),
        sys.executable, "-m", "wenbo_engine.bench.communication_workloads",
        "--kind", kind, "--n", str(n), "--depth", str(depth),
        "--chunk-bits", str(chunk_bits),
        "--work-dir", str(Path(tmp) / "work"),
        "--output", str(out),
    ]
    if verify:
        cmd.append("--verify")
    if output_dir:
        cmd += ["--output-dir", str(output_dir)]
    if extra:
        cmd += extra
    proc = subprocess.run(cmd, env=_env(), capture_output=True, text=True,
                          timeout=300)
    assert proc.returncode == 0, (
        f"mpirun failed (code {proc.returncode}):\n{proc.stdout}\n{proc.stderr}")
    assert out.exists(), f"profile not written\n{proc.stderr}"
    return json.loads(out.read_text()), proc


@mpi_required
@pytest.mark.parametrize("np_ranks", [2, 4])
@pytest.mark.parametrize("kind", list(cw.GENERATORS))
def test_workload_correct_vs_ref(kind, np_ranks):
    """Small-n correctness for every workload at np=2 and np=4."""
    with tempfile.TemporaryDirectory() as tmp:
        res, _ = _run_cli(tmp, kind, n=6, depth=12, chunk_bits=2,
                          np_ranks=np_ranks)
    assert res["correct"] is True, f"{kind} np={np_ranks} state mismatch"
    assert abs(res["final_norm"] - 1.0) < 1e-5


@mpi_required
def test_mpi_nonlocal_heavy_metrics_are_measured():
    with tempfile.TemporaryDirectory() as tmp:
        res, _ = _run_cli(tmp, "mpi_nonlocal_heavy", n=5, depth=8,
                          chunk_bits=2, np_ranks=2)
    agg = res["aggregate"]
    assert res["measured_mpi_nonlocal_ops"] > 0
    assert agg["mpi_bytes_sent"] > 0, "expected measured nonzero MPI bytes"
    assert agg["sendrecv_count"] > 0
    assert res["partner_rank_pairs"] > 0, "expected nonempty observed partners"
    assert res["correct"] is True


@mpi_required
def test_artifact_bundle_written():
    with tempfile.TemporaryDirectory() as tmp:
        adir = Path(tmp) / "artifacts"
        res, _ = _run_cli(tmp, "mpi_nonlocal_heavy", n=6, depth=12,
                          chunk_bits=3, np_ranks=4, output_dir=adir)
        required_files = [
            "config.json", "circuit.json", "plan.json", "cost_model.json",
            "stage_profile.csv", "mpi_profile.csv", "io_profile.csv",
            "recovery_events.json", "final_summary.json", "final_norm.txt",
            "git_commit.txt",
        ]
        for fn in required_files:
            assert (adir / fn).exists(), f"missing artifact {fn}"
        summary = json.loads((adir / "final_summary.json").read_text())
        for key in ("workload_kind", "seed", "n", "depth", "chunk_bits",
                    "num_ranks", "intended_locality", "measured_local_ops",
                    "measured_rank_nonlocal_ops", "measured_mpi_nonlocal_ops",
                    "mpi_bytes_sent", "sendrecv_count", "partner_rank_pairs"):
            assert key in summary, f"final_summary missing {key}"
        assert summary["measured_mpi_nonlocal_ops"] > 0
        assert summary["mpi_bytes_sent"] > 0
        # final_norm.txt parses and is ~1
        assert abs(float((adir / "final_norm.txt").read_text()) - 1.0) < 1e-5


@mpi_required
def test_metrics_do_not_leak_between_workloads():
    """Two workloads in ONE process: light after heavy must report 0 MPI."""
    driver = (
        "import json,os,tempfile\n"
        "from mpi4py import MPI\n"
        "from wenbo_engine.bench import communication_workloads as cw\n"
        "comm=MPI.COMM_WORLD\n"
        "td=tempfile.mkdtemp()\n"
        "heavy=cw.run_workload('mpi_nonlocal_heavy',6,12,3,os.path.join(td,'h'),"
        "comm=comm,seed=1)\n"
        "light=cw.run_workload('communication_light',6,12,3,os.path.join(td,'l'),"
        "comm=comm,seed=1)\n"
        "if comm.Get_rank()==0:\n"
        "    print('RESULT '+json.dumps({'heavy':heavy['aggregate']['mpi_bytes_sent'],"
        "'light':light['aggregate']['mpi_bytes_sent'],"
        "'light_pairs':light['partner_rank_pairs']}))\n"
    )
    cmd = ["mpirun", "--oversubscribe", "-np", "2", sys.executable, "-c", driver]
    proc = subprocess.run(cmd, env=_env(), capture_output=True, text=True,
                          timeout=300)
    assert proc.returncode == 0, f"{proc.stdout}\n{proc.stderr}"
    line = [l for l in proc.stdout.splitlines() if l.startswith("RESULT ")]
    assert line, f"no result line:\n{proc.stdout}\n{proc.stderr}"
    data = json.loads(line[0][len("RESULT "):])
    assert data["heavy"] > 0, "heavy run should have MPI bytes"
    assert data["light"] == 0, f"stale MPI bytes leaked into light run: {data}"
    assert data["light_pairs"] == 0, "stale partner pairs leaked into light run"
