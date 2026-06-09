"""Tests for the machine calibration runner."""
from __future__ import annotations

import json

from wenbo_engine.profiling import CalibrationRunner


def test_calibration_run_non_mpi(tmp_path):
    runner = CalibrationRunner(tmp_path, chunk_size=1 << 12, n_chunks=4)
    model = runner.run(include_mpi=False)

    for key in ("nvme", "fsync", "rename", "checksum", "mpi"):
        assert key in model

    nvme = model["nvme"]
    assert nvme["read_bandwidth_MBps"] > 0
    assert nvme["write_bandwidth_MBps"] > 0
    assert model["fsync"]["fsync_sec_per_call"] >= 0
    assert model["rename"]["rename_sec_per_call"] >= 0
    assert model["checksum"]["checksum_throughput_MBps"] > 0
    assert model["mpi"]["available"] is False


def test_calibration_write(tmp_path):
    runner = CalibrationRunner(tmp_path, chunk_size=1 << 12, n_chunks=2)
    out = tmp_path / "cost_model.json"
    model = runner.write(out, include_mpi=False)
    assert out.exists()
    loaded = json.loads(out.read_text())
    assert loaded["chunk_size"] == model["chunk_size"]
    assert "nvme" in loaded
