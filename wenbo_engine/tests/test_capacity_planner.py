"""Tests for the hardware-bound capacity planner.

Covers the seven required scenarios:
  1. complex64 / complex128 state sizes are exact
  2. impossible configs are rejected
  3. feasible configs are accepted
  4. max_feasible_qubits is computed correctly
  5. recovery mode changes feasibility
  6. durable checkpoint requirement is reported separately
  7. 45q is a *scenario*, never a hardcoded target
"""
from __future__ import annotations

from dataclasses import replace

import pytest

from wenbo_engine.planner.capacity_planner import (
    BYTES_PER_AMP,
    RECOVERY_MODES,
    PlannerConfig,
    evaluate_qubits,
    max_feasible_qubits,
    plan,
    recommend_recovery_mode,
    recommend_chunk_bits,
    estimate_peak_ram,
    state_size_bytes,
)

TIB = 1 << 40
GIB = 1 << 30


# ── 1. exact state sizes ─────────────────────────────────────────────────
@pytest.mark.parametrize("n", [0, 1, 10, 20, 30, 45])
def test_complex64_state_size(n):
    assert state_size_bytes(n, "complex64") == (1 << n) * 8


@pytest.mark.parametrize("n", [0, 1, 10, 20, 30, 45])
def test_complex128_state_size(n):
    assert state_size_bytes(n, "complex128") == (1 << n) * 16


def test_complex128_is_double_complex64():
    for n in range(0, 20):
        assert state_size_bytes(n, "complex128") == 2 * state_size_bytes(n, "complex64")


def test_known_45q_complex64_is_256_tib():
    # 2^45 * 8 bytes = 2^48 bytes = 256 TiB exactly.
    assert state_size_bytes(45, "complex64") == 256 * TIB


def test_bad_precision_rejected():
    with pytest.raises(ValueError):
        state_size_bytes(10, "float32")


# ── 2. impossible configs are rejected ───────────────────────────────────
def test_zero_storage_is_infeasible():
    cfg = PlannerConfig(
        precision="complex64", num_ranks=4,
        local_storage_per_rank_tib=0.0, ram_per_rank_gib=64.0,
        recovery_mode="wal",
    )
    assert evaluate_qubits(cfg, 10).feasible is False
    assert max_feasible_qubits(cfg) is None


def test_state_too_big_for_storage_is_infeasible():
    # 50q complex64 = 2^53 bytes = 8192 TiB; tiny cluster cannot hold it.
    cfg = PlannerConfig(
        precision="complex64", num_ranks=4,
        local_storage_per_rank_tib=1.0, ram_per_rank_gib=64.0,
        recovery_mode="wal",
    )
    f = evaluate_qubits(cfg, 50)
    assert f.feasible is False
    assert f.local_feasible is False
    assert f.reasons  # a human-readable reason is attached


def test_no_ram_is_infeasible():
    cfg = PlannerConfig(
        precision="complex64", num_ranks=4,
        local_storage_per_rank_tib=100.0, ram_per_rank_gib=0.0,
        recovery_mode="wal",
    )
    f = evaluate_qubits(cfg, 10)
    assert f.ram_feasible is False
    assert f.feasible is False


# ── 3. feasible configs are accepted ─────────────────────────────────────
def test_small_circuit_on_big_cluster_is_feasible():
    cfg = PlannerConfig(
        precision="complex64", num_ranks=64,
        local_storage_per_rank_tib=8.0, ram_per_rank_gib=256.0,
        recovery_mode="wal",
    )
    f = evaluate_qubits(cfg, 30)
    assert f.feasible is True
    assert f.local_feasible and f.ram_feasible and f.durable_feasible
    assert f.storage_margin_per_rank_bytes > 0


# ── 4. max_feasible_qubits is computed correctly ─────────────────────────
def test_max_feasible_matches_hand_computation():
    # num_ranks=1, no reserve, no temp headroom, abundant RAM, "none" (2*S,
    # zero metadata overhead so the boundary is exact).
    # 2 * 2^q * 8 <= storage.  storage = 64 GiB  ->  2^q <= 2^32  ->  q = 32.
    cfg = PlannerConfig(
        precision="complex64", num_ranks=1,
        local_storage_per_rank_tib=64 * GIB / TIB,  # 64 GiB
        ram_per_rank_gib=64.0,
        reserved_storage_fraction=0.0, max_temp_storage_fraction=0.0,
        recovery_mode="none",
    )
    assert max_feasible_qubits(cfg) == 32
    assert evaluate_qubits(cfg, 32).feasible is True
    assert evaluate_qubits(cfg, 33).feasible is False


def test_max_feasible_is_monotone_and_boundary_exact():
    cfg = PlannerConfig(
        precision="complex64", num_ranks=64,
        local_storage_per_rank_tib=8.0, ram_per_rank_gib=256.0,
        reserved_storage_fraction=0.15, max_temp_storage_fraction=0.10,
        recovery_mode="generation",
    )
    mq = max_feasible_qubits(cfg)
    assert mq is not None
    assert evaluate_qubits(cfg, mq).feasible is True
    assert evaluate_qubits(cfg, mq + 1).feasible is False


def test_max_candidate_caps_the_search():
    cfg = PlannerConfig(
        precision="complex64", num_ranks=1,
        local_storage_per_rank_tib=1e9,  # effectively unlimited
        ram_per_rank_gib=64.0, recovery_mode="wal",
        max_candidate_qubits=12,
    )
    assert max_feasible_qubits(cfg) == 12


# ── 5. recovery mode changes feasibility ─────────────────────────────────
def test_recovery_mode_shifts_the_frontier():
    # num_ranks=1, no reserve/temp, RAM abundant.  S = 2^q * 8.
    #   wal:        hot+dest                 = 2*S <= storage
    #   generation: hot+dest+2 retained      = 4*S <= storage   (3 committed)
    # storage = 20 GiB:
    #   wal: q=30 -> 2*8=16<=20 ok, q=31 -> 32>20  => wal_max = 30
    #   gen: q=29 -> 4*4=16<=20 ok, q=30 -> 4*8=32>20 => gen_max = 29
    base = PlannerConfig(
        precision="complex64", num_ranks=1,
        local_storage_per_rank_tib=20 * GIB / TIB,
        ram_per_rank_gib=64.0,
        reserved_storage_fraction=0.0, max_temp_storage_fraction=0.0,
        recovery_mode="wal",
    )
    wal_max = max_feasible_qubits(replace(base, recovery_mode="wal"))
    gen_max = max_feasible_qubits(replace(base, recovery_mode="generation"))
    dur_max = max_feasible_qubits(replace(base, recovery_mode="generation+durable"))

    assert wal_max == 30
    assert gen_max == 29
    # stronger protection never allows more qubits
    assert wal_max >= gen_max >= dur_max
    # and here it strictly costs qubits
    assert wal_max > gen_max


def test_same_qubit_count_can_flip_feasibility_by_mode():
    cfg = PlannerConfig(
        precision="complex64", num_ranks=1,
        local_storage_per_rank_tib=20 * GIB / TIB,
        ram_per_rank_gib=64.0,
        reserved_storage_fraction=0.0, max_temp_storage_fraction=0.0,
        recovery_mode="wal",
    )
    assert evaluate_qubits(replace(cfg, recovery_mode="wal"), 30).feasible is True
    assert evaluate_qubits(replace(cfg, recovery_mode="generation"), 30).feasible is False


# ── 6. durable checkpoint requirement is reported separately ─────────────
def test_durable_checkpoint_reported_as_separate_budget():
    cfg = PlannerConfig(
        precision="complex64", num_ranks=64,
        local_storage_per_rank_tib=8.0, ram_per_rank_gib=256.0,
        recovery_mode="generation+durable",
        durable_storage_available_tib=400.0,
    )
    f = evaluate_qubits(cfg, 40)
    # durable requirement == one full state snapshot, tracked apart from local.
    assert f.durable_checkpoint_required_bytes == state_size_bytes(40, "complex64")
    assert f.durable_storage_bytes == 400.0 * TIB
    assert f.durable_feasible is True
    # the durable snapshot is NOT folded into local storage here
    assert f.durable_local_bytes == 0.0


def test_durable_folds_into_local_when_no_separate_store():
    cfg = PlannerConfig(
        precision="complex64", num_ranks=64,
        local_storage_per_rank_tib=8.0, ram_per_rank_gib=256.0,
        recovery_mode="generation+durable",
        durable_storage_available_tib=None,
    )
    f = evaluate_qubits(cfg, 40)
    # with no separate durable store, the rank's slice lands on local NVMe
    assert f.durable_local_bytes == f.per_rank_state_bytes
    # and a separate durable requirement is still reported for visibility
    assert f.durable_checkpoint_required_bytes == state_size_bytes(40, "complex64")


def test_durable_can_be_the_binding_constraint():
    cfg = PlannerConfig(
        precision="complex64", num_ranks=64,
        local_storage_per_rank_tib=100.0, ram_per_rank_gib=256.0,
        recovery_mode="generation+durable",
        durable_storage_available_tib=1.0,  # far too small for any real snapshot
    )
    f = evaluate_qubits(cfg, 40)  # 8 TiB snapshot vs 1 TiB durable
    assert f.local_feasible is True
    assert f.durable_feasible is False
    assert f.feasible is False


def test_plan_reports_overheads_separately_in_json():
    cfg = PlannerConfig(
        precision="complex64", num_ranks=64,
        local_storage_per_rank_tib=8.0, ram_per_rank_gib=256.0,
        recovery_mode="generation",
    )
    out = plan(cfg, num_qubits=40)
    req = out["requested"]
    # each storage category has its own reported line item
    for key in (
        "hot_source_per_rank_tib",
        "destination_temp_per_rank_tib",
        "retained_local_recovery_per_rank_tib",
        "durable_checkpoint_total_tib",
        "total_local_required_per_rank_tib",
        "total_local_required_tib",
        "total_durable_required_tib",
        "wal_recovery_overhead_per_rank_tib",
    ):
        assert key in req
    # generation default (3 committed) keeps 2 retained recovery copies beyond
    # the hot source, and a destination generation.
    f = evaluate_qubits(cfg, 40)
    assert f.committed_generations_retained == 3
    assert f.needs_destination_generation is True
    assert f.hot_source_bytes == f.per_rank_state_bytes
    assert f.retained_local_recovery_bytes == 2 * f.per_rank_state_bytes


# ── 7. 45q is a scenario, not a hardcoded target ─────────────────────────
def test_45q_is_just_one_scenario_scarce_hardware():
    # Hardware too small for 45q -> max feasible is BELOW 45 and 45 is rejected.
    cfg = PlannerConfig(
        precision="complex64", num_ranks=64,
        local_storage_per_rank_tib=8.0, ram_per_rank_gib=256.0,
        recovery_mode="generation",
    )
    out = plan(cfg, num_qubits=45)
    assert out["requested_qubits"] == 45
    assert out["requested_feasible"] is False
    assert out["max_feasible_qubits"] < 45


def test_45q_is_just_one_scenario_ample_hardware():
    # Give it plenty: max feasible should exceed 45, proving 45 is not a ceiling.
    cfg = PlannerConfig(
        precision="complex64", num_ranks=1024,
        local_storage_per_rank_tib=64.0, ram_per_rank_gib=256.0,
        recovery_mode="wal",
    )
    out = plan(cfg)
    assert out["max_feasible_qubits"] > 45
    assert evaluate_qubits(cfg, 45).feasible is True


def test_max_feasible_tracks_precision():
    # complex128 amplitudes are 2x the bytes -> one fewer qubit fits.
    common = dict(
        num_ranks=8, local_storage_per_rank_tib=4.0,
        ram_per_rank_gib=128.0,
        reserved_storage_fraction=0.0, max_temp_storage_fraction=0.0,
        recovery_mode="wal",
    )
    q64 = max_feasible_qubits(PlannerConfig(precision="complex64", **common))
    q128 = max_feasible_qubits(PlannerConfig(precision="complex128", **common))
    assert q64 == q128 + 1


# ── recommendation + config validation ───────────────────────────────────
def test_recommend_prefers_strongest_feasible_mode():
    # Storage holds the durable variant comfortably -> recommend it.
    cfg = PlannerConfig(
        precision="complex64", num_ranks=64,
        local_storage_per_rank_tib=64.0, ram_per_rank_gib=256.0,
        recovery_mode="wal",
    )
    assert recommend_recovery_mode(cfg, 40) == "generation+durable"


def test_recommend_downgrades_when_storage_tight():
    # Tight storage: only the cheapest modes fit -> recommend weaker mode.
    cfg = PlannerConfig(
        precision="complex64", num_ranks=1,
        local_storage_per_rank_tib=20 * GIB / TIB,
        ram_per_rank_gib=64.0,
        reserved_storage_fraction=0.0, max_temp_storage_fraction=0.0,
        recovery_mode="generation",
    )
    # at 30q only wal/none fit (generation needs 3*S)
    rec = recommend_recovery_mode(cfg, 30)
    assert rec in ("wal", "none")


def test_invalid_config_rejected():
    with pytest.raises(ValueError):
        PlannerConfig(precision="bad", num_ranks=4,
                      local_storage_per_rank_tib=1.0)
    with pytest.raises(ValueError):
        PlannerConfig(recovery_mode="bad", num_ranks=4,
                      local_storage_per_rank_tib=1.0)
    with pytest.raises(ValueError):
        PlannerConfig(num_ranks=0, local_storage_per_rank_tib=1.0)


def test_all_recovery_modes_are_evaluable():
    for mode in RECOVERY_MODES:
        cfg = PlannerConfig(
            precision="complex64", num_ranks=64,
            local_storage_per_rank_tib=32.0, ram_per_rank_gib=256.0,
            recovery_mode=mode,
            durable_storage_available_tib=1000.0,
        )
        assert max_feasible_qubits(cfg) is not None


# ── follow-up: power-of-two rank enforcement ─────────────────────────────
@pytest.mark.parametrize("ranks", [1, 2, 8, 64, 1024])
def test_power_of_two_ranks_accepted(ranks):
    cfg = PlannerConfig(
        precision="complex64", num_ranks=ranks,
        local_storage_per_rank_tib=8.0, ram_per_rank_gib=256.0,
        recovery_mode="wal",
    )
    assert cfg.num_ranks == ranks


@pytest.mark.parametrize("ranks", [3, 5, 6, 10, 48, 100])
def test_non_power_of_two_ranks_rejected(ranks):
    with pytest.raises(ValueError):
        PlannerConfig(
            precision="complex64", num_ranks=ranks,
            local_storage_per_rank_tib=8.0, ram_per_rank_gib=256.0,
            recovery_mode="wal",
        )


def test_non_power_of_two_allowed_with_override():
    cfg = PlannerConfig(
        precision="complex64", num_ranks=10,
        local_storage_per_rank_tib=8.0, ram_per_rank_gib=256.0,
        recovery_mode="wal", allow_non_power_of_two=True,
    )
    assert cfg.num_ranks == 10
    assert max_feasible_qubits(cfg) is not None


# ── follow-up: retention is parameterized, not hardcoded ─────────────────
def test_generation_storage_scales_with_committed_generations():
    base = dict(
        precision="complex64", num_ranks=64,
        local_storage_per_rank_tib=8.0, ram_per_rank_gib=256.0,
        recovery_mode="generation",
    )
    f3 = evaluate_qubits(PlannerConfig(committed_generations_retained=3, **base), 40)
    f5 = evaluate_qubits(PlannerConfig(committed_generations_retained=5, **base), 40)
    # 5 committed keeps 2 more retained rollback copies than 3 committed.
    assert f5.retained_local_recovery_bytes == \
        f3.retained_local_recovery_bytes + 2 * f3.per_rank_state_bytes
    assert f5.total_local_required_per_rank_bytes > \
        f3.total_local_required_per_rank_bytes


def test_generation_default_matches_agent2_three_generations():
    # Agent 2's generation recovery prunes to 3 committed generations.
    cfg = PlannerConfig(
        precision="complex64", num_ranks=64,
        local_storage_per_rank_tib=8.0, ram_per_rank_gib=256.0,
        recovery_mode="generation",
    )
    f = evaluate_qubits(cfg, 40)
    assert f.committed_generations_retained == 3
    # peak local = hot(1) + destination(1) + retained(2) = 4 * S, plus the
    # atomic-write temp headroom and tiny WAL/commit metadata.
    S = f.per_rank_state_bytes
    temp = cfg.max_temp_storage_fraction * S
    expected = 4 * S + temp + f.wal_recovery_overhead_bytes
    assert f.total_local_required_per_rank_bytes == pytest.approx(expected)


def test_destination_generation_toggle_changes_storage():
    base = dict(
        precision="complex64", num_ranks=64,
        local_storage_per_rank_tib=8.0, ram_per_rank_gib=256.0,
        recovery_mode="generation",
    )
    with_dest = evaluate_qubits(
        PlannerConfig(needs_destination_generation=True, **base), 40)
    no_dest = evaluate_qubits(
        PlannerConfig(needs_destination_generation=False, **base), 40)
    assert with_dest.destination_temp_bytes > no_dest.destination_temp_bytes


def test_durable_snapshots_retained_scales_durable_requirement():
    base = dict(
        precision="complex64", num_ranks=64,
        local_storage_per_rank_tib=8.0, ram_per_rank_gib=256.0,
        recovery_mode="generation+durable",
        durable_storage_available_tib=10000.0,
    )
    f1 = evaluate_qubits(PlannerConfig(durable_snapshots_retained=1, **base), 40)
    f2 = evaluate_qubits(PlannerConfig(durable_snapshots_retained=2, **base), 40)
    assert f2.durable_checkpoint_required_bytes == \
        2 * f1.durable_checkpoint_required_bytes


# ── follow-up: recommended runnable config is internally consistent ──────
def test_recommended_config_is_internally_consistent():
    cfg = PlannerConfig(
        precision="complex64", num_ranks=64,
        local_storage_per_rank_tib=8.0, ram_per_rank_gib=256.0,
        recovery_mode="generation",
    )
    out = plan(cfg)
    rc = out["recommended_config"]
    assert rc is not None
    rq = rc["recommended_num_qubits"]
    # the recommended count is exactly the reported frontier
    assert rq == out["max_feasible_qubits"]
    # re-evaluating the recommended config reproduces a feasible plan
    rebuilt = PlannerConfig(
        precision=rc["precision"], num_ranks=rc["num_ranks"],
        local_storage_per_rank_tib=8.0, ram_per_rank_gib=256.0,
        recovery_mode=rc["recovery_mode"],
        committed_generations_retained=rc["committed_generations_retained"],
        needs_destination_generation=rc["needs_destination_generation"],
        durable_snapshots_retained=rc["durable_snapshots_retained"],
    )
    f = evaluate_qubits(rebuilt, rq)
    assert f.feasible is True
    assert rc["feasible"] is True
    # the next qubit up must NOT be feasible (it really is the frontier)
    assert evaluate_qubits(rebuilt, rq + 1).feasible is False
    # reported storage requirement matches an independent recomputation
    assert rc["total_local_required_tib"] == pytest.approx(
        round(f.total_local_required_bytes / TIB, 6))
    # power-of-two ranks (runnable under the MPI runner)
    assert (rc["num_ranks"] & (rc["num_ranks"] - 1)) == 0


def test_recommended_config_carries_durable_warning():
    cfg = PlannerConfig(
        precision="complex64", num_ranks=64,
        local_storage_per_rank_tib=64.0, ram_per_rank_gib=256.0,
        recovery_mode="generation+durable",
        durable_storage_available_tib=None,
    )
    out = plan(cfg)
    rc = out["recommended_config"]
    assert rc["durable_snapshots_retained"] >= 1
    assert any("durable" in w for w in rc["warnings"])


# ── RAM working-set model (ram-aware execution) ──────────────────────────

def _cluster_cfg(**kw):
    # 8 x i3en.xlarge: 2.2 TiB NVMe / 30 GiB RAM per rank, generation recovery.
    base = dict(precision="complex64", num_ranks=8,
                local_storage_per_rank_tib=2.2, ram_per_rank_gib=30.0,
                recovery_mode="generation")
    base.update(kw)
    return PlannerConfig(**base)


def test_storage_and_ram_feasibility_reported_separately():
    # case 1: both verdicts present and independent
    f = evaluate_qubits(_cluster_cfg(), 36)
    assert f.storage_feasible is True            # NVMe has ample room
    assert f.ram_feasible is False               # but RAM working set does not
    assert f.storage_feasible != f.ram_feasible
    out = plan(_cluster_cfg(), num_qubits=36)["requested"]
    assert "storage_feasible" in out and "ram_feasible" in out
    assert out["storage_feasible"] is True and out["ram_feasible"] is False


def test_n38_storage_feasible_but_ram_infeasible():
    # case 2: n=38 fits storage (8x2.2TiB) but not RAM (30GiB) at default chunk_bits
    f = evaluate_qubits(_cluster_cfg(), 38)
    assert f.storage_feasible is True
    assert f.ram_feasible is False
    assert f.estimated_peak_ram_bytes > f.ram_working_budget_bytes


def test_auto_chunk_bits_recommends_smaller_chunk_bits():
    # case 3: auto-chunk-bits recommends a smaller chunk_bits that fits RAM
    default_cb = evaluate_qubits(_cluster_cfg(), 36).chunk_bits
    f = evaluate_qubits(_cluster_cfg(auto_chunk_bits=True, ram_budget_gib=21.0,
                                     has_mpi=False), 36)
    assert f.recommended_chunk_bits is not None
    assert f.recommended_chunk_bits < default_cb
    assert f.ram_feasible is True                 # at the recommended chunk_bits
    assert f.estimated_peak_ram_bytes <= f.ram_working_budget_bytes
    # and n=34 mpi (step) likewise gets a feasible recommendation
    g = evaluate_qubits(_cluster_cfg(auto_chunk_bits=True, ram_budget_gib=21.0,
                                     execution_mode="step", has_mpi=True), 34)
    assert g.recommended_chunk_bits is not None and g.ram_feasible is True


def test_recommend_chunk_bits_none_when_budget_too_small():
    # a 0.4 GiB budget cannot even hold metadata + 1 chunk -> None
    rec = recommend_chunk_bits(num_qubits=30, num_ranks=8,
                               ram_budget_bytes=int(0.4 * GIB),
                               execution_mode="compute_unit", has_mpi=False)
    assert rec is None


def test_estimate_peak_ram_unbounded_vs_bounded():
    # unbounded overlay holds the whole partition; bounded streams (much less)
    common = dict(num_qubits=34, num_ranks=8, chunk_bits=29,
                  execution_mode="compute_unit", has_mpi=False)
    unb = estimate_peak_ram(bounded_overlay=False, **common)
    bnd = estimate_peak_ram(bounded_overlay=True, **common)
    assert unb["estimated_peak_ram_bytes"] > bnd["estimated_peak_ram_bytes"]
