"""Pure cost model for MPI-exchange windows (analysis-only, deterministic)."""
from wenbo_engine.mpi import window_cost_model as cm


def test_chunk_bytes():
    assert cm.chunk_bytes(19) == (1 << 19) * 8
    assert cm.chunk_bytes(10, itemsize=16) == (1 << 10) * 16


def test_window_vs_baseline_fetches():
    # 10 steps each fetch 8 distinct chunks; baseline = 80 fetches.
    assert cm.baseline_fetches(80) == 80
    # window gathers the 8 distinct once + scatters once = 16 transfers.
    assert cm.window_fetches(8) == 16
    assert cm.repeated_fetches_avoided(80, 8) == 72
    # never negative
    assert cm.repeated_fetches_avoided(4, 8) == 0


def test_bytes_for_and_byte_reduction():
    cb = cm.chunk_bytes(19)
    assert cm.bytes_for(80, cb) == 80 * cb
    assert cm.bytes_for(16, cb) == 16 * cb
    assert cm.bytes_for(80, cb) - cm.bytes_for(16, cb) == 64 * cb


def test_sendrecv_reduction():
    assert cm.sendrecv_reduction(20, 4) == 16
    assert cm.sendrecv_reduction(3, 4) == -1     # honest: window can cost more


def test_ram_estimate_scales_with_chunks():
    cb = cm.chunk_bytes(19)
    r1 = cm.estimate_window_ram_gib(8, cb)
    r2 = cm.estimate_window_ram_gib(16, cb)
    assert r2 == 2 * r1
    assert cm.estimate_window_ram_gib(16, cb, overhead_factor=1.5) == 1.5 * r2


def test_ram_feasible_rule():
    assert cm.ram_feasible(0.1, 21) is True
    assert cm.ram_feasible(30.0, 21) is False
    assert cm.ram_feasible(0.1, None) is False   # unknown budget → not feasible


def test_expected_recomputation_more_with_fewer_commits():
    # 20 gates: committing once leaves a bigger segment than committing 10×.
    few = cm.expected_recomputation_units(20, 1)
    many = cm.expected_recomputation_units(20, 10)
    assert few > many
    assert cm.expected_recomputation_units(0, 5) == 0.0
    assert cm.expected_recomputation_units(20, 0) == 0.0


def test_recomputation_cost_increase_positive_when_commits_drop():
    inc = cm.recomputation_cost_increase(20, commit_count_baseline=10,
                                         commit_count_window=1)
    assert inc > 0
    # equal cadence → no increase
    assert cm.recomputation_cost_increase(20, 5, 5) == 0.0


def test_blended_costs_deterministic():
    w = cm.CostWeights()
    kw = dict(mpi_bytes=10 << 30, sendrecv_count=20, commit_count=11,
              recompute_units=1.0, weights=w)
    assert cm.baseline_cost(**kw) == cm.baseline_cost(**kw)
    kw2 = dict(gather_bytes=1 << 30, scatter_bytes=1 << 30, sendrecv_count=4,
               commit_count=1, recompute_units=10.0, extra_ram_gib=0.2,
               weights=w)
    assert cm.window_cost(**kw2) == cm.window_cost(**kw2)


def test_weights_to_dict_roundtrip():
    d = cm.weights_to_dict(cm.CostWeights())
    assert d["byte_cost_per_gib"] == 1.0
    assert "recompute_cost_per_gate" in d
