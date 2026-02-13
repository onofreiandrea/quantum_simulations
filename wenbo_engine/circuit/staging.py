"""Circuit staging with qubit remapping — Atlas ILP, heuristic, and greedy.

Key idea: instead of doing a butterfly exchange for every non-local gate,
rearrange the physical qubit layout (via SWAP operations) so that most
gates become chunk-local.  Trade a few SWAP I/O passes for many avoided
non-local gate passes.

Three methods are available (selected via the ``method`` parameter of
:func:`atlas_stages`):

  ``"ilp"``
    Formulate an Integer Linear Program (via PuLP) that finds the
    optimal assignment of gates to stages and which k qubits should
    be local in each stage, minimising the total number of shuffles.
    Faithful reimplementation of the staging ILP from Atlas
    (Xu et al., SC'24, CMU; https://github.com/quantum-compiler/atlas).

  ``"heuristic"``  *(default)*
    Dependency-aware greedy, ported from Atlas's
    ``num_iterations_by_heuristics`` (circuit.cc).  No external deps.
    Priority for next local set: first-unexecuted-gate qubits >
    global-gate count > local-gate count > qubit index.

  ``"greedy"``
    Simple frequency-based lookahead.  Kept for backward compatibility.

All three methods also use the *insular qubit* optimisation from Atlas:
diagonal / sparse gates (Z, S, T, CZ, CR) have qubits that do not need
to be in the local set, relaxing constraints and allowing more gates per
stage.

Reference:
  Xu et al., "Atlas: Hierarchical Partitioning for Quantum Circuit
  Simulation on GPUs", SC'24.  arXiv:2408.09055
  https://github.com/quantum-compiler/atlas
"""
from __future__ import annotations

from collections import Counter


import numpy as np

from wenbo_engine.circuit.io import validate_circuit_dict, levelize
from wenbo_engine.circuit.fusion import fuse_1q_ops
from wenbo_engine.kernel import gates as gmod

# Optional PuLP import — only needed for method="ilp".
try:
    import pulp  # type: ignore
    HAS_PULP = True
except ImportError:
    HAS_PULP = False

_SWAP_U = None


def _swap_matrix() -> np.ndarray:
    global _SWAP_U
    if _SWAP_U is None:
        _SWAP_U = gmod.SWAP()
    return _SWAP_U


# ── insular qubit detection (Atlas §3.1) ────────────────────────────

# "Insular" qubits are those that appear only in diagonal positions of
# the gate matrix.  They can remain global (non-local) without requiring
# a shuffle — the gate can be decomposed into a per-chunk conditional
# phase.  For our purposes we use this information when *deciding* which
# qubits to make local (ILP/heuristic), but still execute global
# diagonal gates via the existing butterfly exchange at runtime.

_SPARSE_GATES = frozenset({"Z", "S", "T", "CZ", "CR"})


def non_insular_qubits(gate: dict) -> list[int]:
    """Return the subset of a gate's qubits that MUST be local.

    Qubits not in this list are "insular" — they can remain global
    and the gate can still be executed (via conditional phase per chunk
    in Atlas's ``getMat_per_device``, or via butterfly exchange in our
    out-of-core runner).

    Follows Atlas's ``is_sparse()`` check in ``circuit.cc``:
    when a gate is sparse (diagonal), the locality constraint is
    **entirely skipped** for ALL its qubits — not just some.

    Sparse gates in our gate set:
      - Z, S, T: diagonal 1Q gates.
      - CZ, CR: diagonal 2Q gates (controlled phase).
    """
    name = gate["gate"]
    # Sparse/diagonal gates: no qubit needs to be local.
    if name in _SPARSE_GATES:
        return []
    # All other gates: every qubit must be local.
    return list(gate["qubits"])


# ── qubit mapping ────────────────────────────────────────────────────

class QubitMap:
    """Bidirectional logical <-> physical qubit mapping."""

    def __init__(self, n: int):
        self.n = n
        self._l2p = list(range(n))
        self._p2l = list(range(n))

    def phys(self, logical: int) -> int:
        return self._l2p[logical]

    def logical(self, physical: int) -> int:
        return self._p2l[physical]

    def local_set(self, k: int) -> set[int]:
        """Logical qubits currently at physical positions < k."""
        return {self._p2l[p] for p in range(min(k, self.n))}

    def swap_phys(self, pa: int, pb: int) -> None:
        """Swap two physical positions in the mapping."""
        la, lb = self._p2l[pa], self._p2l[pb]
        self._l2p[la], self._l2p[lb] = pb, pa
        self._p2l[pa], self._p2l[pb] = lb, la

    def to_list(self) -> list[int]:
        return list(self._l2p)

    def is_identity(self) -> bool:
        return all(self._l2p[i] == i for i in range(self.n))


# ── shared helpers ───────────────────────────────────────────────────

def _gen_swap_ops(
    qmap: QubitMap, current_local: set[int], desired_local: set[int],
) -> list[tuple[list[int], np.ndarray]]:
    """Generate SWAP gate ops to transition between local sets.

    Modifies *qmap* in-place.
    """
    need_in = sorted(desired_local - current_local)
    need_out = sorted(current_local - desired_local)
    U = _swap_matrix()
    ops: list[tuple[list[int], np.ndarray]] = []
    for lq_in, lq_out in zip(need_in, need_out):
        p_in = qmap.phys(lq_in)
        p_out = qmap.phys(lq_out)
        ops.append(([p_out, p_in], U))
        qmap.swap_phys(p_out, p_in)
    return ops


def _flush_stage(
    stage_gates: list[dict], qmap: QubitMap, k: int,
) -> list[dict]:
    """Convert accumulated all-local gates into a fused local-only step."""
    if not stage_gates:
        return []
    ops: list[tuple[list[int], np.ndarray]] = []
    for g in stage_gates:
        phys_qs = [qmap.phys(q) for q in g["qubits"]]
        U = gmod.gate_matrix(g["gate"], g["params"])
        ops.append((phys_qs, U))
    return [{"local_ops": fuse_1q_ops(ops), "nonlocal_ops": []}]


def _is_local(gate: dict, qmap: QubitMap, k: int) -> bool:
    return all(qmap.phys(q) < k for q in gate["qubits"])



# ── method 1: ILP staging (Atlas §3.2) ──────────────────────────────

def _compute_local_qubits_ilp(
    gates: list[dict],
    n: int,
    k: int,
) -> list[set[int]]:
    """Use an ILP (PuLP) to find optimal local-qubit sets per stage.

    Binary-searches on the number of stages S.  For each candidate S,
    solves the ILP.  Returns the first feasible solution.

    ILP formulation (following Atlas ``compute_local_qubits_with_ilp``):

    Variables:
      x[s][q]  in {0,1}  — qubit q is local in stage s
      y[g][s]  in {0,1}  — gate g is assigned to stage s

    Constraints:
      1) Each gate assigned to exactly one stage.
      2) Dependency ordering (topological).
      3) Non-insular qubits of each gate must be local in its stage.
      4) Exactly k local qubits per stage.

    Objective: minimise total transition cost between consecutive stages
      (measured as number of qubits that change local/global status).
    """
    if not HAS_PULP:
        raise ImportError("PuLP is required for method='ilp'. pip install pulp")

    n_gates = len(gates)
    if n_gates == 0:
        return [set(range(min(k, n)))]

    # Pre-compute non-insular qubit sets and dependency predecessors.
    ni_qubits = [non_insular_qubits(g) for g in gates]

    # Predecessor map: for each gate, the set of gates it depends on
    # (through shared qubits, respecting circuit order).
    qubit_last: dict[int, int] = {}
    predecessors: list[list[int]] = [[] for _ in range(n_gates)]
    for gi, g in enumerate(gates):
        for q in g["qubits"]:
            if q in qubit_last:
                predecessors[gi].append(qubit_last[q])
            qubit_last[q] = gi

    # Binary search on number of stages.
    lo_s, hi_s = 1, n_gates  # worst case: each gate is its own stage
    # Quick upper bound: the heuristic solution.
    heur_stages = _compute_local_qubits_heuristic(gates, n, k)
    hi_s = min(hi_s, len(heur_stages))
    best_result: list[set[int]] | None = None

    while lo_s <= hi_s:
        mid_s = (lo_s + hi_s) // 2
        result = _try_ilp(gates, n, k, mid_s, ni_qubits, predecessors)
        if result is not None:
            best_result = result
            hi_s = mid_s - 1
        else:
            lo_s = mid_s + 1

    if best_result is None:
        # Fallback to heuristic if ILP failed (shouldn't happen).
        return heur_stages
    return best_result


def _try_ilp(
    gates: list[dict],
    n: int,
    k: int,
    num_stages: int,
    ni_qubits: list[list[int]],
    predecessors: list[list[int]],
) -> list[set[int]] | None:
    """Try to solve the staging ILP for a given number of stages.

    Returns list of local-qubit sets if feasible, else None.
    """
    S = num_stages
    G = len(gates)
    Q = n

    prob = pulp.LpProblem("atlas_staging", pulp.LpMinimize)

    # Variables
    x = [[pulp.LpVariable(f"x_{s}_{q}", cat="Binary")
          for q in range(Q)] for s in range(S)]
    y = [[pulp.LpVariable(f"y_{g}_{s}", cat="Binary")
          for s in range(S)] for g in range(G)]

    # Transition cost variables (for objective)
    if S > 1:
        d = [[pulp.LpVariable(f"d_{s}_{q}", lowBound=0, cat="Continuous")
              for q in range(Q)] for s in range(S - 1)]

    # Constraint 1: each gate assigned to exactly one stage.
    for g in range(G):
        prob += pulp.lpSum(y[g][s] for s in range(S)) == 1

    # Constraint 2: dependency ordering.
    # If gate p precedes gate g, then p must be in an earlier-or-equal stage.
    # Encoded as: cumulative assignment of p by stage s >= y[g][s].
    for g in range(G):
        for p in predecessors[g]:
            for s in range(S):
                prob += pulp.lpSum(y[p][sp] for sp in range(s + 1)) >= y[g][s]

    # Constraint 3: non-insular qubits must be local.
    for g in range(G):
        for q in ni_qubits[g]:
            for s in range(S):
                prob += y[g][s] <= x[s][q]

    # Constraint 4: exactly k local qubits per stage.
    for s in range(S):
        prob += pulp.lpSum(x[s][q] for q in range(Q)) == k

    # Objective: minimise total transition cost.
    if S > 1:
        for s in range(S - 1):
            for q in range(Q):
                prob += d[s][q] >= x[s][q] - x[s + 1][q]
                prob += d[s][q] >= x[s + 1][q] - x[s][q]
        prob += pulp.lpSum(d[s][q] for s in range(S - 1) for q in range(Q))
    else:
        prob += 0  # trivial objective for single stage

    # Solve (suppress output).
    status = prob.solve(pulp.PULP_CBC_CMD(msg=0, timeLimit=30))

    if pulp.LpStatus[status] != "Optimal":
        return None

    # Extract local qubit sets.
    result: list[set[int]] = []
    for s in range(S):
        local = {q for q in range(Q) if pulp.value(x[s][q]) > 0.5}
        result.append(local)
    return result


# ── method 2: Atlas heuristic (circuit.cc num_iterations_by_heuristics)

def _compute_local_qubits_heuristic(
    gates: list[dict],
    n: int,
    k: int,
) -> list[set[int]]:
    """Atlas's dependency-aware greedy heuristic for stage computation.

    Ported from ``num_iterations_by_heuristics`` in Atlas's circuit.cc:
      1. Start with no qubits local.
      2. Try to execute all gates whose non-insular qubits are local,
         respecting dependency order (a gate is executable only if no
         prior unexecuted gate blocks its qubits).
      3. When blocked, count for each qubit how many unexecuted gates
         it participates in (global vs local).
      4. Prioritise: first-unexecuted-gate's qubits > global count
         (descending) > local count (descending) > qubit index.
      5. Select top-k as local for the next stage.
    """
    n_gates = len(gates)
    if n_gates == 0:
        return [set(range(min(k, n)))]

    executed = [False] * n_gates
    local_qubit = [False] * n
    stages: list[set[int]] = []

    while True:
        # ── execute all gates that can be executed with current layout ──
        all_done = True
        executable = [True] * n  # qubit not blocked by prior gate
        for gi in range(n_gates):
            g = gates[gi]
            if executed[gi]:
                continue
            qs = g["qubits"]
            # Gate is executable if:
            #   (a) all its qubits are unblocked (dependency ordering), AND
            #   (b) all its non-insular qubits are in the local set.
            ok = all(executable[q] for q in qs)
            ni = non_insular_qubits(g)
            if ok and all(local_qubit[q] for q in ni):
                executed[gi] = True
            else:
                all_done = False
                # Block all qubits touched by this unexecutable gate.
                for q in qs:
                    executable[q] = False

        if all_done:
            break

        # ── pick next local set ──────────────────────────────────────
        # Count per-qubit participation in unexecuted gates.
        first_unexecuted_gate = [False] * n
        local_gates_count = [0] * n
        global_gates_count = [0] * n
        first = True

        for gi in range(n_gates):
            g = gates[gi]
            if executed[gi]:
                continue
            qs = g["qubits"]
            ni = non_insular_qubits(g)
            is_local_gate = all(local_qubit[q] for q in ni)

            for q in qs:
                if is_local_gate:
                    local_gates_count[q] += 1
                else:
                    global_gates_count[q] += 1
                if first:
                    first_unexecuted_gate[q] = True
            first = False

        # Sort qubits by Atlas priority:
        #   1. Qubit is in first unexecuted gate (True > False)
        #   2. More global (non-local) gates (descending)
        #   3. More local gates (descending)
        #   4. Smaller qubit index (ascending, for determinism)
        candidates = list(range(n))
        candidates.sort(key=lambda q: (
            -int(first_unexecuted_gate[q]),
            -global_gates_count[q],
            -local_gates_count[q],
            q,
        ))

        # Select top-k.
        new_local: set[int] = set()
        for q in candidates:
            new_local.add(q)
            if len(new_local) >= k:
                break

        stages.append(new_local)

        # Update local_qubit mask.
        for q in range(n):
            local_qubit[q] = q in new_local

    return stages


# ── method 3: greedy frequency lookahead (original) ──────────────────

def _compute_local_qubits_greedy(
    gates: list[dict],
    n: int,
    k: int,
    lookahead: int = 200,
) -> list[set[int]]:
    """Simple frequency-based lookahead (original implementation).

    Not dependency-aware.  Kept for backward compatibility.
    This wraps the old algorithm into the local-qubit-sets interface
    so it can share the common stage-to-steps conversion.
    """
    # The greedy method works differently — it uses a running QubitMap
    # and makes decisions gate-by-gate.  To fit the local-qubit-sets
    # interface we fall through to the old code path in atlas_stages.
    # Return None to signal "use legacy code path".
    return None  # type: ignore[return-value]


# ── convert local-qubit sets to steps + SWAPs ───────────────────────

def _local_sets_to_steps(
    gates: list[dict],
    n: int,
    k: int,
    local_sets: list[set[int]],
) -> tuple[list[dict], list[int]]:
    """Convert a sequence of local-qubit sets into runner steps.

    For each stage:
      1. Set up a QubitMap so that the stage's local qubits are at
         physical positions 0..k-1 (via SWAPs from the previous stage).
      2. Execute all gates whose non-insular qubits are in this stage's
         local set, respecting dependency order.
      3. Emit fused local-only steps + any remaining non-local ops.
    """
    qmap = QubitMap(n)
    steps: list[dict] = []
    executed = [False] * len(gates)

    for stage_idx, desired_local in enumerate(local_sets):
        # ── emit SWAPs to transition to this stage's local set ──
        current_local = qmap.local_set(k)
        swap_ops = _gen_swap_ops(qmap, current_local, desired_local)
        if swap_ops:
            steps.append({"local_ops": [], "nonlocal_ops": swap_ops})

        # ── collect gates executable in this stage ──
        local_qubit_mask = [False] * n
        for q in desired_local:
            local_qubit_mask[q] = True

        executable_q = [True] * n
        stage_local_gates: list[dict] = []
        stage_nonlocal_ops: list[tuple[list[int], np.ndarray]] = []

        for gi in range(len(gates)):
            if executed[gi]:
                continue
            g = gates[gi]
            qs = g["qubits"]

            # Check dependency: all qubits must be unblocked.
            if not all(executable_q[q] for q in qs):
                for q in qs:
                    executable_q[q] = False
                continue

            ni = non_insular_qubits(g)
            if all(local_qubit_mask[q] for q in ni):
                # Gate can execute in this stage.
                executed[gi] = True

                # Check if ALL qubits are physically local (for kernel).
                if all(qmap.phys(q) < k for q in qs):
                    stage_local_gates.append(g)
                else:
                    # Non-insular qubits are local but some insular qubits
                    # are global — execute via butterfly exchange.
                    phys_qs = [qmap.phys(q) for q in qs]
                    U = gmod.gate_matrix(g["gate"], g["params"])
                    stage_nonlocal_ops.append((phys_qs, U))
            else:
                # Gate not executable in this stage — block its qubits.
                for q in qs:
                    executable_q[q] = False

        # ── emit steps for this stage ──
        if stage_local_gates:
            steps.extend(_flush_stage(stage_local_gates, qmap, k))
        if stage_nonlocal_ops:
            steps.append({"local_ops": [], "nonlocal_ops": stage_nonlocal_ops})

    return steps, qmap.to_list()


# ── greedy (legacy) code path ────────────────────────────────────────

def _greedy_stages(
    cd: dict,
    all_gates: list[dict],
    n: int,
    k: int,
    lookahead: int,
) -> tuple[list[dict], list[int]]:
    """Original greedy gate-by-gate staging (frequency lookahead)."""
    qmap = QubitMap(n)
    steps: list[dict] = []
    stage_gates: list[dict] = []

    gi = 0
    while gi < len(all_gates):
        g = all_gates[gi]
        if _is_local(g, qmap, k):
            stage_gates.append(g)
            gi += 1
            continue

        # Non-local gate: flush + remap.
        steps.extend(_flush_stage(stage_gates, qmap, k))
        stage_gates = []

        # Frequency-based lookahead for desired local set.
        freq: Counter = Counter()
        for gg in all_gates[gi:gi + lookahead]:
            for q in gg["qubits"]:
                freq[q] += 1
        desired: set[int] = set()
        for q, _ in freq.most_common():
            desired.add(q)
            if len(desired) >= k:
                break
        for q in range(n):
            if len(desired) >= k:
                break
            if q not in desired:
                desired.add(q)

        current = qmap.local_set(k)
        swap_ops = _gen_swap_ops(qmap, current, desired)
        if swap_ops:
            steps.append({"local_ops": [], "nonlocal_ops": swap_ops})

        if _is_local(g, qmap, k):
            stage_gates.append(g)
            gi += 1
        else:
            phys_qs = [qmap.phys(q) for q in g["qubits"]]
            U = gmod.gate_matrix(g["gate"], g["params"])
            steps.append({
                "local_ops": [],
                "nonlocal_ops": [(phys_qs, U)],
            })
            gi += 1

    steps.extend(_flush_stage(stage_gates, qmap, k))
    return steps, qmap.to_list()


# ── main entry point ─────────────────────────────────────────────────

def atlas_stages(
    circuit_dict: dict,
    k: int,
    method: str = "heuristic",
    lookahead: int = 200,
) -> tuple[list[dict], list[int]]:
    """Convert a circuit into steps using circuit staging.

    Parameters
    ----------
    circuit_dict : dict
        Validated circuit dictionary.
    k : int
        log2(chunk_size) — number of local qubit positions.
    method : str
        ``"ilp"`` — Atlas ILP (requires PuLP).
        ``"heuristic"`` — Atlas dependency-aware greedy (default).
        ``"greedy"`` — simple frequency lookahead (legacy).
    lookahead : int
        Look-ahead window size (only used by ``"greedy"``).

    Returns
    -------
    steps : list[dict]
        Step dicts ``{"local_ops": [...], "nonlocal_ops": [...]}``.
    log_to_phys : list[int]
        Final logical-to-physical qubit mapping.
    """
    cd = validate_circuit_dict(circuit_dict)
    n = cd["number_of_qubits"]
    all_gates = cd["gates"]

    # Trivial case: all qubits fit in one chunk.
    if n <= k:
        from wenbo_engine.circuit.fusion import batch_levels
        return batch_levels(levelize(cd), k), list(range(n))

    if method == "greedy":
        return _greedy_stages(cd, all_gates, n, k, lookahead)

    if method == "ilp":
        local_sets = _compute_local_qubits_ilp(all_gates, n, k)
    elif method == "heuristic":
        local_sets = _compute_local_qubits_heuristic(all_gates, n, k)
    else:
        raise ValueError(f"unknown staging method: {method!r}")

    return _local_sets_to_steps(all_gates, n, k, local_sets)


# ── state permutation utility ─────────────────────────────────────────

def permute_state(state: np.ndarray, log_to_phys: list[int]) -> np.ndarray:
    """Reorder an in-memory state vector from permuted physical layout
    to the standard logical qubit order.

    log_to_phys[i] = physical bit position of logical qubit i.

    Uses numpy tensor transpose -- O(2^n), suitable for n <= ~30.
    """
    n = len(log_to_phys)
    if all(log_to_phys[i] == i for i in range(n)):
        return state

    # Reshape to n-dim tensor.  C-order: axis j = bit (n-1-j).
    # We want result axis (n-1-q) to come from tensor axis (n-1-log_to_phys[q]).
    perm = [0] * n
    for q in range(n):
        perm[n - 1 - q] = n - 1 - log_to_phys[q]

    tensor = state.reshape([2] * n)
    return tensor.transpose(perm).reshape(-1).copy()


# ── stats ─────────────────────────────────────────────────────────────

def staging_stats(
    circuit_dict: dict,
    k: int,
    method: str = "heuristic",
) -> dict:
    """Compare step counts: staging vs baseline (batch_levels)."""
    from wenbo_engine.circuit.fusion import batch_levels

    cd = validate_circuit_dict(circuit_dict)
    levels = levelize(cd)

    baseline = batch_levels(levels, k)
    staged, _ = atlas_stages(circuit_dict, k, method=method)

    bl_nl = sum(1 for s in baseline if s.get("nonlocal_ops"))
    st_nl = sum(1 for s in staged if s.get("nonlocal_ops"))

    return {
        "baseline_steps": len(baseline),
        "staged_steps": len(staged),
        "baseline_nonlocal_steps": bl_nl,
        "staged_nonlocal_steps": st_nl,
        "reduction": (
            f"{len(baseline)}->{len(staged)} "
            f"({(1 - len(staged) / max(len(baseline), 1)) * 100:.0f}% fewer I/O passes)"
        ),
    }
