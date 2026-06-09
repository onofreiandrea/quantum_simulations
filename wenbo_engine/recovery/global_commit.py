"""Global commit record — the cluster-wide durability point.

A generation ``g`` is *committed* if and only if a ``commits/commit_XXXXXX.json``
record exists, is internally consistent (its stored ``commit_hash`` matches its
content), and lists, for every rank, the manifest hash that rank prepared.

Hard invariant: **no global commit record ⇒ no committed generation.**
A rank may have written all its chunks and its manifest for generation g+1,
but until the coordinator has written the commit record, g+1 does not exist as
a recoverable state and recovery must roll back to g.

Pure Python — no MPI, no numpy.
"""
from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, field
from pathlib import Path

COMMITS_DIRNAME = "commits"


def commit_filename(generation: int) -> str:
    return f"commit_{generation:06d}.json"


def _stable_hash(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                     default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


@dataclass
class GlobalCommitRecord:
    """Records that all ranks prepared generation ``generation``."""

    generation: int
    n_ranks: int
    circuit_hash: str
    step_index: int               # circuit step that produced this gen (stage id)
    # rank -> that rank's RankManifest.manifest_hash for this generation
    rank_manifest_hashes: dict[int, str] = field(default_factory=dict)
    parent_generation: int = -1   # generation this was derived from (-1 = init)
    created: float = 0.0
    commit_hash: str = ""

    # ── hashing ───────────────────────────────────────────────────────

    def _hash_payload(self) -> dict:
        """Content identity (excludes ``created`` and ``commit_hash``)."""
        return {
            "generation": self.generation,
            "n_ranks": self.n_ranks,
            "circuit_hash": self.circuit_hash,
            "step_index": self.step_index,
            "parent_generation": self.parent_generation,
            # normalize keys to sorted ints for a deterministic payload
            "rank_manifest_hashes": {
                str(r): self.rank_manifest_hashes[r]
                for r in sorted(self.rank_manifest_hashes)
            },
        }

    def compute_hash(self) -> str:
        return _stable_hash(self._hash_payload())

    def seal(self) -> "GlobalCommitRecord":
        self.commit_hash = self.compute_hash()
        return self

    def verify_self_hash(self) -> bool:
        return bool(self.commit_hash) and \
            self.commit_hash == self.compute_hash()

    # ── structural validation ─────────────────────────────────────────

    def validate(self) -> None:
        if self.n_ranks < 1:
            raise ValueError(f"n_ranks must be >= 1, got {self.n_ranks}")
        missing = set(range(self.n_ranks)) - set(self.rank_manifest_hashes)
        if missing:
            raise ValueError(
                f"commit gen {self.generation} missing manifest hashes for "
                f"ranks {sorted(missing)}")

    # ── (de)serialization ─────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "generation": self.generation,
            "n_ranks": self.n_ranks,
            "circuit_hash": self.circuit_hash,
            "step_index": self.step_index,
            "parent_generation": self.parent_generation,
            "rank_manifest_hashes": {
                str(r): h for r, h in sorted(self.rank_manifest_hashes.items())
            },
            "created": self.created,
            "commit_hash": self.commit_hash,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GlobalCommitRecord":
        return cls(
            generation=int(d["generation"]),
            n_ranks=int(d["n_ranks"]),
            circuit_hash=str(d["circuit_hash"]),
            step_index=int(d["step_index"]),
            rank_manifest_hashes={
                int(r): str(h)
                for r, h in d.get("rank_manifest_hashes", {}).items()
            },
            parent_generation=int(d.get("parent_generation", -1)),
            created=float(d.get("created", 0.0)),
            commit_hash=str(d.get("commit_hash", "")),
        )

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_json(cls, raw: str | bytes) -> "GlobalCommitRecord":
        if isinstance(raw, bytes):
            raw = raw.decode()
        return cls.from_dict(json.loads(raw))

    # ── atomic file I/O (commit-protocol steps 9–11) ──────────────────

    def write_atomic(self, commits_dir: str | Path) -> Path:
        """Write commit_XXXXXX.json atomically (tmp + fsync + rename).

        This is the single point at which a generation becomes committed.
        """
        self.validate()
        self.seal()
        d = Path(commits_dir)
        d.mkdir(parents=True, exist_ok=True)
        final = d / commit_filename(self.generation)
        tmp = d / (commit_filename(self.generation) + ".tmp")
        raw = self.to_json()
        with open(tmp, "w") as f:
            f.write(raw)
            f.flush()
            os.fsync(f.fileno())
        # fsync the directory so the rename is durable across a power loss
        os.replace(str(tmp), str(final))
        _fsync_dir(d)
        return final

    @classmethod
    def read_file(cls, path: str | Path) -> "GlobalCommitRecord":
        with open(path) as f:
            return cls.from_json(f.read())


def list_commit_files(commits_dir: str | Path) -> list[Path]:
    """All commit_XXXXXX.json files, sorted oldest→newest by generation."""
    d = Path(commits_dir)
    if not d.exists():
        return []
    files = [p for p in d.glob("commit_*.json") if p.is_file()]
    return sorted(files, key=lambda p: p.name)


def _fsync_dir(d: Path) -> None:
    """Best-effort directory fsync (no-op on platforms that disallow it)."""
    try:
        fd = os.open(str(d), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except (OSError, PermissionError):
        pass


def _fsync_file(path: Path) -> None:
    """fsync a file's data to stable storage (best-effort)."""
    try:
        fd = os.open(str(path), os.O_RDONLY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
    except (OSError, PermissionError):
        pass
