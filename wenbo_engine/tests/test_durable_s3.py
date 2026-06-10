"""S3 durable backend tests (mocked S3 via moto).

Exercises the OPTIONAL :class:`~wenbo_engine.durable.s3_backend.S3Backend`
against an in-process mock S3 (no real AWS, no cost).  Skipped automatically
when ``boto3`` or ``moto`` are not installed, so the suite still passes in a
minimal environment.

Reuses the real-state generation helpers from the restore tests, so the
promote -> restore round-trip is numerically meaningful (norm == 1) — the
only difference here is the durable backend is S3 instead of a local path.
"""
import shutil

import numpy as np
import pytest

boto3 = pytest.importorskip("boto3")
moto = pytest.importorskip("moto")
from moto import mock_aws  # noqa: E402

from wenbo_engine.durable import (  # noqa: E402
    DurableCheckpointManager, DurableRestoreManager,
)
from wenbo_engine.durable.s3_backend import S3Backend  # noqa: E402
from wenbo_engine.recovery import LocalCoordinator, RecoveryScanner  # noqa: E402
from wenbo_engine.tests.test_durable_restore import (  # noqa: E402
    _commit_real_run, _real_state, _read_state, _norm, CIRCUIT_HASH,
)

BUCKET = "wenbo-durable-test"
REGION = "us-east-1"
RUN_ID = "durable_s3_run"


def _s3_backend(prefix=""):
    """A mocked-S3-backed S3Backend with the bucket already created."""
    client = boto3.client("s3", region_name=REGION)
    client.create_bucket(Bucket=BUCKET)
    return S3Backend(bucket=BUCKET, prefix=prefix, client=client)


# ── backend primitives ──────────────────────────────────────────────────

@mock_aws
def test_s3_backend_primitives_roundtrip():
    be = _s3_backend(prefix="run42")
    assert be.exists("a/b.bin") is False
    res = be.put("a/b.bin", b"hello-durable")
    assert res.size_bytes == len(b"hello-durable")
    assert be.exists("a/b.bin") is True
    assert be.get("a/b.bin") == b"hello-durable"
    # checksum verify helper round-trips
    assert be.checksum("a/b.bin") == res.checksum
    be.put("a/c.bin", b"x")
    assert be.list("a/") == ["a/b.bin", "a/c.bin"]
    be.delete("a/b.bin")
    assert be.exists("a/b.bin") is False
    assert be.list("a/") == ["a/c.bin"]


@mock_aws
def test_s3_backend_root_parses_bucket_and_prefix():
    boto3.client("s3", region_name=REGION).create_bucket(Bucket=BUCKET)
    client = boto3.client("s3", region_name=REGION)
    be = S3Backend(root=f"{BUCKET}/some/prefix", client=client)
    assert be.bucket == BUCKET
    assert be.prefix == "some/prefix"
    be.put("k.bin", b"v")
    # object lands under the parsed prefix
    raw = client.get_object(Bucket=BUCKET, Key="some/prefix/k.bin")["Body"].read()
    assert raw == b"v"


# ── promote -> restore round-trip over S3 ────────────────────────────────

@mock_aws
def test_s3_promote_then_restore_roundtrip(tmp_path):
    chunk_size = 2
    psi = _real_state(n_qubits=3)               # length 8, normalized
    n_chunks = len(psi) // chunk_size
    work = tmp_path / "work"

    gm, coord = _commit_real_run(work, psi, chunk_size)
    backend = _s3_backend()

    cm = DurableCheckpointManager(work, RUN_ID, backend, coord)
    cm.upload_run_metadata()
    assert cm.promote(1) is not None

    original = _read_state(work, 0, 1, n_chunks)
    assert abs(_norm(original) - 1.0) < 1e-6

    # Lose the local state entirely; restore from S3.
    shutil.rmtree(work)
    assert not work.exists()

    rm = DurableRestoreManager(work, RUN_ID, backend, coord)
    result = rm.restore_latest(check_checksums=True)
    assert result.restored
    assert result.generation == 1

    restored = _read_state(work, 0, 1, n_chunks)
    assert np.allclose(restored, original, atol=0)
    assert abs(_norm(restored) - 1.0) < 1e-6

    # The standard local recovery scanner picks up the restored generation.
    res = RecoveryScanner(work).scan(quarantine=False, check_checksums=True)
    assert res.recovered and res.generation == 1
    assert res.record.circuit_hash == CIRCUIT_HASH


@mock_aws
def test_s3_restore_no_durable_commit_is_noop(tmp_path):
    work = tmp_path / "work"
    backend = _s3_backend()
    rm = DurableRestoreManager(work, RUN_ID, backend, LocalCoordinator())
    result = rm.restore_latest(check_checksums=True)
    assert not result.restored
    assert result.generation is None
