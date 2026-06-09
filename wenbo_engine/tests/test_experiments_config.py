"""Tests for experiment config loading and validation."""
from __future__ import annotations

import json

import pytest

from wenbo_engine.experiments.config import ExperimentConfig, load_config


def test_load_yaml(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text(
        "runner: single_node\n"
        "circuit:\n"
        "  source: builtin\n"
        "  name: ghz\n"
        "  n_qubits: 5\n"
        "chunk_bits: 3\n"
        "buffer_depth: 2\n"
    )
    cfg = load_config(p)
    assert cfg.runner == "single_node"
    assert cfg.circuit.name == "ghz"
    assert cfg.circuit.n_qubits == 5
    assert cfg.resolved_chunk_size(5) == 8


def test_load_json(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({
        "runner": "mpi",
        "circuit": {"source": "builtin", "name": "qft", "n_qubits": 6},
        "chunk_size": 16,
    }))
    cfg = load_config(p)
    assert cfg.runner == "mpi"
    assert cfg.chunk_size == 16
    assert cfg.resolved_chunk_size(6) == 16


def test_resolved_chunk_size_clamps_to_state():
    cfg = ExperimentConfig.from_dict({"chunk_size": 1 << 20,
                                      "circuit": {"n_qubits": 4}})
    # 2^20 > 2^4, clamp to full state 16
    assert cfg.resolved_chunk_size(4) == 16


def test_invalid_runner_rejected():
    with pytest.raises(ValueError):
        ExperimentConfig.from_dict({"runner": "spark_xyz"})


def test_unknown_key_rejected():
    with pytest.raises(ValueError):
        ExperimentConfig.from_dict({"nonsense": 1})


def test_unknown_circuit_key_rejected():
    with pytest.raises(ValueError):
        ExperimentConfig.from_dict({"circuit": {"bogus": 1}})


def test_non_power_of_two_chunk_rejected():
    with pytest.raises(ValueError):
        ExperimentConfig.from_dict({"chunk_size": 17})


def test_missing_file():
    with pytest.raises(FileNotFoundError):
        load_config("/no/such/config.yaml")
