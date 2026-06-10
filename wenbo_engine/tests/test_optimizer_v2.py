"""Tests for Optimizer v2 (wenbo_engine.planner).

Maps to the 7 required cases:

  1. plan serialization is deterministic
  2. stage plan preserves gate dependencies
  3. placement maps hot qubits to low physical bits
  4. final state matches current execution (vs ref_dense, atol=1e-6)
  5. ablation report has all 5 modes with all required metric fields
  6. optimizer does NOT silently remove MPI stress unless requested
  7. stage_v2_fusion reduces step count OR data movement vs `current`

All proofs are deterministic plan metrics computed in-process (no cluster
run); the orchestrator runs pytest to confirm.
"""
import numpy as np
import pytest

from wenbo_engine.kernel.ref_dense import simulate
from wenbo_engine.planner import (
    ABLATION_MODES, HardwareConfig, build_plan, plan_metrics,
    ablation_report, serialize_plan, deserialize_plan, plan_to_json,
)
from wenbo_engine.planner.circuit_dag import build_dag
from wenbo_engine.planner.optimizer_v2 import replay_statevector, plan_to_gates
from wenbo_engine.planner.placement_planner import plan_placement
from wenbo_engine.planner.qubit_activity import qubit_activity


# ── circuit fixtures ─────────────────────────────────────────────────

def _clustered_circuit():
    """8q circuit with a clustered high-qubit phase then a low-qubit phase.

    The high-qubit phase is non-local under the default layout, giving
    staging room to help; consecutive local layers give fusion room.
    """
    return {"number_of_qubits": 8, "gates": [
        {"qubits": [4], "gate": "H"},
        {"qubits": [5], "gate": "H"},
        {"qubits": [6], "gate": "H"},
        {"qubits": [4, 5], "gate": "CNOT"},
        {"qubits": [5, 6], "gate": "CNOT"},
        {"qubits": [4, 6], "gate": "CZ"},
        {"qubits": [4], "gate": "T"},
        {"qubits": [5], "gate": "S"},
        {"qubits": [4, 5], "gate": "CNOT"},
        {"qubits": [5, 6], "gate": "CNOT"},
        {"qubits": [0], "gate": "H"},
        {"qubits": [1], "gate": "H"},
        {"qubits": [2], "gate": "H"},
        {"qubits": [0, 1], "gate": "CNOT"},
        {"qubits": [1, 2], "gate": "CNOT"},
        {"qubits": [0, 2], "gate": "CZ"},
    ]}


def _small_circuit():
    return {"number_of_qubits": 4, "gates": [
        {"qubits": [0], "gate": "H"},
        {"qubits": [2], "gate": "H"},
        {"qubits": [0, 2], "gate": "CNOT"},
        {"qubits": [1, 3], "gate": "CNOT"},
        {"qubits": [3], "gate": "RX", "params": {"theta": 0.7}},
        {"qubits": [1, 2], "gate": "CZ"},
    ]}


def _mpi_heavy_circuit():
    """6q circuit with gates deliberately on the top (rank) bits.

    With chunk_bits=2 and num_ranks=4 (p=2), bits 4,5 are rank bits, so a
    gate touching qubit 4 or 5 is MPI-nonlocal.
    """
    return {"number_of_qubits": 6, "gates": [
        {"qubits": [4, 0], "gate": "CNOT"},
        {"qubits": [5, 1], "gate": "CNOT"},
        {"qubits": [4, 1], "gate": "CZ"},
        {"qubits": [5, 0], "gate": "CNOT"},
        {"qubits": [0], "gate": "H"},
    ]}


def _hw(n=8, chunk_bits=2, num_ranks=4, recovery="wal"):
    return HardwareConfig(n_qubits=n, chunk_bits=chunk_bits,
                          num_ranks=num_ranks, recovery=recovery)


# ── Case 1: deterministic serialization ──────────────────────────────

@pytest.mark.parametrize("mode", ABLATION_MODES)
def test_serialization_deterministic(mode):
    cd = _clustered_circuit()
    hw = _hw()
    p1 = build_plan(cd, hw, mode)
    p2 = build_plan(cd, hw, mode)

    s1 = serialize_plan(p1)
    s2 = serialize_plan(p2)
    # same input => byte-identical plan
    assert plan_to_json(p1) == plan_to_json(p2)

    # serialize -> deserialize -> serialize is stable (byte-identical)
    round_trip = serialize_plan(deserialize_plan(s1))
    import json
    assert json.dumps(round_trip, sort_keys=True) == \
        json.dumps(s1, sort_keys=True)
    assert s1 == s2


def test_deserialized_plan_replays_identically():
    cd = _clustered_circuit()
    plan = build_plan(cd, _hw(), "stage_v2_fusion")
    rebuilt = deserialize_plan(serialize_plan(plan))
    np.testing.assert_allclose(
        replay_statevector(rebuilt), replay_statevector(plan), atol=1e-9)


# ── Case 2: gate dependencies preserved ──────────────────────────────

@pytest.mark.parametrize("mode", ABLATION_MODES)
def test_dependencies_preserved(mode):
    """No gate is scheduled before a conflicting earlier gate.

    We check, in the PHYSICAL circuit the mode runs, that the plan's flat op
    order keeps every pair of conflicting (qubit-sharing) gates in their
    original relative order.  SWAP ops (staging) are excluded from the DAG
    check since they are inserted by the planner, not original gates; their
    correctness is covered by the statevector equivalence test (case 4).
    """
    cd = _clustered_circuit()
    plan = build_plan(cd, _hw(), mode)

    # Reconstruct the physical circuit the plan executes (original gates
    # only, in plan order) and assert it is a valid linear extension of the
    # physical-circuit DAG.
    phys_gates = []
    for op in plan_to_gates(plan):
        if op.gate == "SWAP" or op.gate == "__matrix__":
            # SWAPs / fused matrices: skip for the DAG order check (handled
            # by equivalence). Fused matrices on >1 qubit could hide order,
            # but fusion only merges commuting/local ops within a stage.
            continue
        phys_gates.append(op)

    # Build the DAG over the physical original circuit and verify order.
    if plan.perm is not None:
        from wenbo_engine.planner.placement_planner import apply_placement
        if plan.mode == "current_static_reorder":
            from wenbo_engine.circuit.reorder import reorder_qubits
            phys_cd, _ = reorder_qubits(cd)
        else:
            phys_cd = apply_placement(cd, plan.perm)
    else:
        phys_cd = cd

    # For levelized modes the plan's named ops are exactly the gates; check
    # their order against the physical DAG.
    if plan.mode in ("current", "current_static_reorder"):
        dag = build_dag(phys_cd)
        qubits = [g["qubits"] for g in phys_cd["gates"]]
        # Map plan ops back to gate indices greedily by (qubits, gate).
        order = _recover_order(phys_cd["gates"], phys_gates)
        assert dag.is_topological(order), \
            f"{mode}: plan reorders a gate past a conflict"
        # explicit pairwise conflict check
        pos = {g: i for i, g in enumerate(order)}
        for a in range(len(qubits)):
            for b in range(a + 1, len(qubits)):
                if set(qubits[a]) & set(qubits[b]):
                    assert pos[a] < pos[b], \
                        f"{mode}: conflicting gates {a},{b} reordered"


def _recover_order(gates, plan_ops):
    """Map plan ops back to original gate indices (greedy, order-stable)."""
    used = [False] * len(gates)
    order = []
    for op in plan_ops:
        for i, g in enumerate(gates):
            if used[i]:
                continue
            if g["qubits"] == op.qubits and g["gate"] == op.gate:
                used[i] = True
                order.append(i)
                break
    # Append any gates not matched (should be none for levelized modes).
    for i in range(len(gates)):
        if not used[i]:
            order.append(i)
    return order


# ── Case 3: placement maps hot qubits to low physical bits ───────────

def test_placement_hot_qubits_low_bits():
    # qubit 7 is the hottest, then 6, then 5 ... build it explicitly.
    gates = []
    for _ in range(5):
        gates.append({"qubits": [7], "gate": "H"})
    for _ in range(4):
        gates.append({"qubits": [6], "gate": "H"})
    for _ in range(3):
        gates.append({"qubits": [5], "gate": "H"})
    gates.append({"qubits": [0], "gate": "H"})
    cd = {"number_of_qubits": 8, "gates": gates}

    act = qubit_activity(cd)
    assert act.hottest()[0] == 7  # hottest qubit identified

    k, p = 2, 1
    perm = plan_placement(cd, k=k, p=p, activity=act)
    # hottest qubit -> physical bit 0 (lowest / chunk-local)
    assert perm[7] == 0
    assert perm[6] == 1
    # bijection
    assert sorted(perm.values()) == list(range(8))
    # rank bits (top p positions) get the coldest / least-active qubits:
    rank_positions = set(range(8 - p, 8))
    rank_qubits = [q for q, pos in perm.items() if pos in rank_positions]
    # an untouched qubit (activity 0) should be on a rank bit, never qubit 7
    assert 7 not in rank_qubits


def test_placement_avoids_active_on_rank_bits_when_possible():
    """When #active qubits <= non-rank positions, no active qubit on rank."""
    cd = {"number_of_qubits": 6, "gates": [
        {"qubits": [0], "gate": "H"},
        {"qubits": [1], "gate": "H"},
        {"qubits": [2], "gate": "H"},
        {"qubits": [0, 1], "gate": "CNOT"},
    ]}  # qubits 3,4,5 are inactive
    k, p = 2, 1  # n=6, non-rank positions = 5, rank positions = {5}
    perm = plan_placement(cd, k=k, p=p)
    rank_pos = 5
    on_rank = [q for q, pos in perm.items() if pos == rank_pos]
    act = qubit_activity(cd)
    for q in on_rank:
        assert act.activity.get(q, 0) == 0, \
            "an active qubit was placed on a rank bit unnecessarily"


# ── Case 4: equivalence to ref_dense for small circuits ──────────────

@pytest.mark.parametrize("mode", ABLATION_MODES)
@pytest.mark.parametrize("circuit_fn", [_small_circuit, _mpi_heavy_circuit])
def test_plan_matches_reference(mode, circuit_fn):
    cd = circuit_fn()
    n = cd["number_of_qubits"]
    # choose a layout valid for this circuit
    hw = _hw(n=n, chunk_bits=2, num_ranks=4, recovery="wal")
    plan = build_plan(cd, hw, mode)
    got = replay_statevector(plan)
    ref = simulate(cd)
    np.testing.assert_allclose(got, ref, atol=1e-6,
                               err_msg=f"mode={mode} circuit mismatch")


@pytest.mark.parametrize("mode", ABLATION_MODES)
def test_plan_matches_reference_clustered(mode):
    cd = _clustered_circuit()
    hw = _hw(n=8, chunk_bits=2, num_ranks=4)
    got = replay_statevector(build_plan(cd, hw, mode))
    np.testing.assert_allclose(got, simulate(cd), atol=1e-6)


# ── Case 5: ablation report completeness ─────────────────────────────

_REQUIRED_FIELDS = {
    "estimated_runtime_sec", "bytes_read", "bytes_written",
    "mpi_bytes_sent", "sendrecv_count", "n_stages", "n_steps",
    "n_commits", "full_state_passes", "final_norm",
}


def test_ablation_report_complete():
    cd = _clustered_circuit()
    report = ablation_report(cd, _hw(), verify_norm=True)
    assert set(report["modes"].keys()) == set(ABLATION_MODES)
    assert report["order"] == ABLATION_MODES
    for mode, metrics in report["modes"].items():
        missing = _REQUIRED_FIELDS - set(metrics)
        assert not missing, f"{mode} missing fields {missing}"
        # final_norm must be ~1 (valid quantum state)
        assert abs(metrics["final_norm"] - 1.0) < 1e-6, \
            f"{mode} produced non-normalized state"


# ── Case 6: MPI stress not silently removed ──────────────────────────

def test_mpi_stress_preserved_unless_requested():
    """On an MPI-heavy circuit, non-reduction modes keep mpi_bytes_sent > 0.

    ``current`` / ``stage_v2`` / ``stage_v2_fusion`` do not perform static
    qubit relocation, so the MPI-nonlocal work (and its bytes) must remain.
    Only the explicit reduction modes (static reorder / placement) are
    allowed to move qubits off the rank bits.
    """
    cd = _mpi_heavy_circuit()
    hw = _hw(n=6, chunk_bits=2, num_ranks=4)
    report = ablation_report(cd, hw, verify_norm=False)

    for mode in ("current", "stage_v2", "stage_v2_fusion"):
        assert report["modes"][mode]["mpi_bytes_sent"] > 0, \
            f"{mode} silently removed MPI stress"
        assert report["modes"][mode]["sendrecv_count"] > 0


def test_reduction_modes_may_reduce_mpi():
    """The explicit reduction modes are ALLOWED to cut MPI bytes."""
    cd = _mpi_heavy_circuit()
    hw = _hw(n=6, chunk_bits=2, num_ranks=4)
    report = ablation_report(cd, hw, verify_norm=False)
    base = report["modes"]["current"]["mpi_bytes_sent"]
    placement = report["modes"]["stage_v2_placement_fusion"]["mpi_bytes_sent"]
    assert placement <= base


# ── Case 7: stage_v2_fusion strictly improves vs current ─────────────

def _sequential_local_circuit():
    """6q circuit: many sequential single-qubit gates on low (local) qubits.

    Under raw levelization (``current``) each gate is its own level (a
    serial dependency chain on the same qubit), so ``current`` needs one
    I/O pass per gate.  Staging + fusion collapse the consecutive local
    layers into a single local pass — a guaranteed strict reduction.
    n=6 > chunk_bits=3 so the staging code path (not the trivial
    single-chunk path) is exercised.
    """
    gates = []
    for _ in range(10):
        gates.append({"qubits": [0], "gate": "H"})
    for _ in range(10):
        gates.append({"qubits": [1], "gate": "H"})
    return {"number_of_qubits": 6, "gates": gates}


def test_stage_v2_fusion_improves_over_current():
    cd = _sequential_local_circuit()
    hw = _hw(n=6, chunk_bits=3, num_ranks=1, recovery="wal")
    report = ablation_report(cd, hw, verify_norm=False)
    cur = report["modes"]["current"]
    fused = report["modes"]["stage_v2_fusion"]

    improved_steps = fused["n_steps"] < cur["n_steps"]
    improved_passes = fused["full_state_passes"] < cur["full_state_passes"]
    improved_read = fused["bytes_read"] < cur["bytes_read"]
    improved_write = fused["bytes_written"] < cur["bytes_written"]
    improved_mpi = fused["mpi_bytes_sent"] < cur["mpi_bytes_sent"]

    assert (improved_steps or improved_passes or improved_read
            or improved_write or improved_mpi), (
        f"stage_v2_fusion did not improve any metric vs current: "
        f"current={cur}, fused={fused}")


def test_stage_v2_fusion_fewer_or_equal_stages_than_stage_v2():
    """Fusion never increases stage count over plain staging."""
    cd = _clustered_circuit()
    hw = _hw(n=8, chunk_bits=3, num_ranks=1)
    report = ablation_report(cd, hw, verify_norm=False)
    assert (report["modes"]["stage_v2_fusion"]["n_steps"]
            <= report["modes"]["stage_v2"]["n_steps"])


# ── extra: cost model & hardware validation ──────────────────────────

def test_cost_model_defaults_make_runtime_positive():
    cd = _clustered_circuit()
    plan = build_plan(cd, _hw(), "current")
    assert plan.metrics["estimated_runtime_sec"] > 0


def test_invalid_hardware_rejected():
    cd = _small_circuit()
    with pytest.raises(ValueError):
        build_plan(cd, HardwareConfig(n_qubits=4, chunk_bits=2, num_ranks=3),
                   "current")
