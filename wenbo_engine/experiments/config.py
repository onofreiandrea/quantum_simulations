"""Experiment configuration: load + validate YAML/JSON into dataclasses.

A config fully describes a reproducible run: the circuit, the runner and its
knobs, the chunk size, and which artifacts to produce.  Loading is tolerant
of either YAML or JSON and fills in sensible defaults, but validates the
fields that would otherwise fail deep inside a run.
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

VALID_RUNNERS = ("single_node", "mpi")
VALID_SOURCES = ("builtin", "json", "qasm")


@dataclass
class CircuitConfig:
    """How to obtain the circuit to simulate."""
    source: str = "builtin"          # builtin | json | qasm
    name: str = "ghz"                # builtin function name (fixtures)
    n_qubits: int = 4
    params: dict[str, Any] = field(default_factory=dict)
    path: str | None = None          # for source in {json, qasm}

    def validate(self) -> None:
        if self.source not in VALID_SOURCES:
            raise ValueError(
                f"circuit.source must be one of {VALID_SOURCES}, got {self.source!r}")
        if self.source == "builtin":
            if not self.name:
                raise ValueError("circuit.name required for source=builtin")
            if not isinstance(self.n_qubits, int) or self.n_qubits < 1:
                raise ValueError(
                    f"circuit.n_qubits must be positive int, got {self.n_qubits!r}")
        else:
            if not self.path:
                raise ValueError(
                    f"circuit.path required for source={self.source}")


@dataclass
class ExperimentConfig:
    """A complete, reproducible experiment description."""
    run_id: str | None = None
    description: str = ""
    runner: str = "single_node"      # single_node | mpi
    circuit: CircuitConfig = field(default_factory=CircuitConfig)

    # chunking: chunk_size wins if set, else 1 << chunk_bits
    chunk_bits: int | None = None
    chunk_size: int | None = None

    # runner knobs
    buffer_depth: int = 4
    kernel: str = "scalar"
    use_wal: bool = True
    use_fusion: bool = False
    # crash-recovery mode: none | wal | generation.  None derives from use_wal
    # (back-compat: True -> wal, False -> none).  "generation" requires the MPI
    # runner (wenbo_engine.recovery global commit protocol).
    recovery: str | None = None

    # observability knobs
    checksum: bool = False           # checksum each chunk per stage
    calibrate: bool = True
    calib_chunks: int = 8

    # paths
    output_dir: str = "experiments"
    work_dir: str | None = None      # scratch for chunk files; default: tmp

    seed: int = 42

    # ── validation / derived ────────────────────────────────────────────
    def validate(self) -> None:
        if self.runner not in VALID_RUNNERS:
            raise ValueError(
                f"runner must be one of {VALID_RUNNERS}, got {self.runner!r}")
        self.circuit.validate()
        if self.chunk_size is not None:
            if self.chunk_size < 1 or (self.chunk_size & (self.chunk_size - 1)) != 0:
                raise ValueError(
                    f"chunk_size must be a power of two, got {self.chunk_size}")
        if self.chunk_bits is not None and self.chunk_bits < 0:
            raise ValueError(f"chunk_bits must be >= 0, got {self.chunk_bits}")
        if self.buffer_depth < 1:
            raise ValueError(f"buffer_depth must be >= 1, got {self.buffer_depth}")
        if self.recovery is not None and self.recovery not in (
                "none", "wal", "generation"):
            raise ValueError(
                f"recovery must be one of none|wal|generation, got {self.recovery!r}")
        if self.recovery == "generation" and self.runner != "mpi":
            raise ValueError(
                "recovery=generation requires runner=mpi "
                "(generation recovery is implemented in the MPI runner)")

    def resolved_recovery(self) -> str:
        """Effective recovery mode (derives from use_wal when unset)."""
        if self.recovery is not None:
            return self.recovery
        return "wal" if self.use_wal else "none"

    def resolved_chunk_size(self, n_qubits: int) -> int:
        """Final chunk_size, clamped to the full state if it is too large."""
        if self.chunk_size is not None:
            cs = self.chunk_size
        elif self.chunk_bits is not None:
            cs = 1 << self.chunk_bits
        else:
            cs = 1 << min(20, n_qubits)   # default: 2^20 or whole state
        N = 1 << n_qubits
        if cs > N:
            cs = N
        if N % cs != 0:
            raise ValueError(
                f"2^{n_qubits} not divisible by chunk_size={cs}")
        return cs

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    # ── construction ────────────────────────────────────────────────────
    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentConfig":
        d = dict(d or {})
        circ = d.pop("circuit", {}) or {}
        if isinstance(circ, CircuitConfig):
            circuit = circ
        else:
            known = {f for f in CircuitConfig().__dict__}
            unknown = set(circ) - known
            if unknown:
                raise ValueError(f"unknown circuit keys: {sorted(unknown)}")
            circuit = CircuitConfig(**circ)

        known = {f for f in cls().__dict__} - {"circuit"}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        cfg = cls(circuit=circuit, **d)
        cfg.validate()
        return cfg


def load_config(path: str | Path) -> ExperimentConfig:
    """Load an experiment config from a .yaml/.yml/.json file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"config not found: {path}")
    text = path.read_text()
    if path.suffix.lower() in (".yaml", ".yml"):
        import yaml
        data = yaml.safe_load(text)
    elif path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        # Try YAML first (a superset of JSON), then JSON.
        try:
            import yaml
            data = yaml.safe_load(text)
        except Exception:
            data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"config root must be a mapping, got {type(data).__name__}")
    return ExperimentConfig.from_dict(data)
