"""StagePlan: one execution stage under a chosen recovery-aware strategy.

A :class:`StagePlan` is the per-stage unit of the recovery-aware planner v1.
It records *what* a stage does (gates, active qubits, locality summary), *how*
it will be executed under the selected strategy (storage layout, execution
mode, extent-I/O mode, MPI exchange mode, compute-unit threshold, commit /
durable policy), and the *predicted* per-stage cost quantities.

It is a pure data record — no I/O, no kernels, no recovery logic.  The planner
builds a list of these from the real runner's compiled steps so the prediction
is grounded in the same levelization the runner actually executes.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class StagePlan:
    # ── identity / content ──────────────────────────────────────────────
    stage_id: int
    gates: int                              # gate count in this stage
    active_qubits: list[int] = field(default_factory=list)
    locality_summary: dict = field(default_factory=dict)  # local/rank/mpi counts
    n_steps: int = 1                        # circuit steps fused into this stage
    kind: str = "step"                      # "local" (fused) | "step"

    # ── strategy this stage is planned under ────────────────────────────
    storage_layout: str = "chunks"
    execution_mode: str = "step"
    extent_io_mode: str = "materialize"
    mpi_exchange_mode: str = "naive"
    compute_unit_min_gates: int = 4
    commit_policy: str = "per_stage_global_commit_record"
    durable_policy: str = "none"

    # ── predicted per-stage quantities ──────────────────────────────────
    estimated_nvme_read_bytes: int = 0
    estimated_nvme_write_bytes: int = 0
    estimated_mpi_bytes: int = 0
    estimated_sendrecv_count: int = 0
    estimated_kernel_time: float = 0.0
    estimated_layout_materialization_cost: float = 0.0
    estimated_commit_cost: float = 0.0
    estimated_durable_checkpoint_cost: float = 0.0
    expected_recomputation_cost: float = 0.0

    def to_dict(self) -> dict:
        return {
            "stage_id": self.stage_id,
            "gates": self.gates,
            "active_qubits": list(self.active_qubits),
            "locality_summary": dict(self.locality_summary),
            "n_steps": self.n_steps,
            "kind": self.kind,
            "storage_layout": self.storage_layout,
            "execution_mode": self.execution_mode,
            "extent_io_mode": self.extent_io_mode,
            "mpi_exchange_mode": self.mpi_exchange_mode,
            "compute_unit_min_gates": self.compute_unit_min_gates,
            "commit_policy": self.commit_policy,
            "durable_policy": self.durable_policy,
            "estimated_nvme_read_bytes": self.estimated_nvme_read_bytes,
            "estimated_nvme_write_bytes": self.estimated_nvme_write_bytes,
            "estimated_mpi_bytes": self.estimated_mpi_bytes,
            "estimated_sendrecv_count": self.estimated_sendrecv_count,
            "estimated_kernel_time": self.estimated_kernel_time,
            "estimated_layout_materialization_cost":
                self.estimated_layout_materialization_cost,
            "estimated_commit_cost": self.estimated_commit_cost,
            "estimated_durable_checkpoint_cost":
                self.estimated_durable_checkpoint_cost,
            "expected_recomputation_cost": self.expected_recomputation_cost,
        }
