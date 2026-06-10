"""Durable checkpoint + restore for committed generations (R4 recovery).

Generation recovery (:mod:`wenbo_engine.recovery`) protects against a crash
that leaves the *local* work_dir intact: the newest valid committed generation
is found again from the on-disk global commit record.  It does **not** protect
against loss of the local NVMe / work_dir itself.

This package adds R4: a committed generation can be *promoted* to a separate
durable store (a plain filesystem path — local disk, NFS, or a JuiceFS mount —
or, optionally, S3) and later *restored* into a freshly rebuilt local work_dir
so execution can resume from it.

Hot-path rule (rule 4): durable storage is **never** touched during normal
gate execution.  The numerical kernels and the per-step commit protocol stay
entirely on local NVMe.  Promotion is a separate, explicit step run only
*between* committed generations, at the configured interval.  No kernel / I/O /
MPI source imports this package.

Layout under ``durable_root`` (mirrors the local layout, plus durable commits)::

    durable_root/<run_id>/
      run.json
      plan.json
      durable_commits/
        durable_commit_000010.json        # the durability point
      generations/
        gen_000010/
          rank_0000/ { manifest.json, chunks/ }
          rank_0001/ { manifest.json, chunks/ }

Durability invariant (mirrors the local one): **no durable commit record ⇒ no
durable generation.**  Restore ignores any durable generation that is not named
by a valid ``durable_commit_*.json``.
"""
from __future__ import annotations

from wenbo_engine.durable.backend import DurableBackend, PutResult
from wenbo_engine.durable.local_path_backend import LocalPathBackend
from wenbo_engine.durable.durable_commit import (
    DurableCommitRecord, durable_commit_filename, list_durable_commit_keys,
)
from wenbo_engine.durable.durable_checkpoint_manager import (
    DurableCheckpointManager, DurableConfig,
)
from wenbo_engine.durable.durable_restore_manager import (
    DurableRestoreManager, RestoreResult,
)

__all__ = [
    "DurableBackend", "PutResult",
    "LocalPathBackend",
    "DurableCommitRecord", "durable_commit_filename", "list_durable_commit_keys",
    "DurableCheckpointManager", "DurableConfig",
    "DurableRestoreManager", "RestoreResult",
    "make_backend",
]


def make_backend(kind: str, **kwargs) -> DurableBackend:
    """Construct a durable backend by name.

    ``kind="local_path"`` builds a :class:`LocalPathBackend` (requires
    ``root=``).  ``kind="s3"`` builds an :class:`S3Backend` (optional; imported
    lazily so a missing ``boto3`` never breaks importing this package).
    """
    if kind == "local_path":
        return LocalPathBackend(**kwargs)
    if kind == "s3":
        # Lazy import so boto3 is only required when S3 is actually selected.
        from wenbo_engine.durable.s3_backend import S3Backend
        return S3Backend(**kwargs)
    raise ValueError(f"unknown durable backend {kind!r} (expected local_path|s3)")
