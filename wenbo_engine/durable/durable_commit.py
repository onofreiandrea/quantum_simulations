"""Durable commit record + durable layout helpers.

A durable generation ``g`` is *durable* iff a ``durable_commits/
durable_commit_XXXXXX.json`` record exists in the durable store, is internally
consistent (its stored ``commit_hash`` matches its content), and names, for
every rank, the durable manifest hash and the per-rank chunk checksums that
were uploaded.

Hard invariant (mirrors the local one): **no durable commit record ⇒ no
durable generation.**  Restore ignores any durable generation directory not
covered by a valid durable commit record — written LAST by the coordinator, so
it is the single durability point of the promotion protocol.

Pure Python — no MPI, no numpy, no backend coupling (records are bytes).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field

DURABLE_COMMITS_DIRNAME = "durable_commits"
DURABLE_GENERATIONS_DIRNAME = "generations"
DURABLE_RUN_JSON = "run.json"
DURABLE_PLAN_JSON = "plan.json"


# ── key layout (POSIX-style backend keys) ──────────────────────────────

def run_root(run_id: str) -> str:
    return run_id.strip("/")


def durable_run_json_key(run_id: str) -> str:
    return f"{run_root(run_id)}/{DURABLE_RUN_JSON}"


def durable_plan_json_key(run_id: str) -> str:
    return f"{run_root(run_id)}/{DURABLE_PLAN_JSON}"


def durable_commits_prefix(run_id: str) -> str:
    return f"{run_root(run_id)}/{DURABLE_COMMITS_DIRNAME}"


def durable_commit_filename(generation: int) -> str:
    return f"durable_commit_{generation:06d}.json"


def durable_commit_key(run_id: str, generation: int) -> str:
    return f"{durable_commits_prefix(run_id)}/{durable_commit_filename(generation)}"


def durable_gen_prefix(run_id: str, generation: int) -> str:
    return (f"{run_root(run_id)}/{DURABLE_GENERATIONS_DIRNAME}/"
            f"gen_{generation:06d}")


def durable_rank_prefix(run_id: str, generation: int, rank: int) -> str:
    return f"{durable_gen_prefix(run_id, generation)}/rank_{rank:04d}"


def durable_manifest_key(run_id: str, generation: int, rank: int) -> str:
    return f"{durable_rank_prefix(run_id, generation, rank)}/manifest.json"


def durable_chunk_key(run_id: str, generation: int, rank: int,
                      filename: str) -> str:
    return f"{durable_rank_prefix(run_id, generation, rank)}/chunks/{filename}"


def _stable_hash(payload: dict) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"),
                     default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


@dataclass
class DurableFileRecord:
    """One uploaded durable file (chunk or manifest): key, size, checksum."""
    key: str
    size_bytes: int
    checksum: str

    def to_dict(self) -> dict:
        return {"key": self.key, "size_bytes": self.size_bytes,
                "checksum": self.checksum}

    @classmethod
    def from_dict(cls, d: dict) -> "DurableFileRecord":
        return cls(key=str(d["key"]), size_bytes=int(d["size_bytes"]),
                   checksum=str(d["checksum"]))


@dataclass
class DurableRankEntry:
    """A rank's durable upload for a generation: manifest + its chunk files."""
    rank: int
    manifest: DurableFileRecord
    manifest_hash: str                  # the RankManifest content hash
    chunks: list[DurableFileRecord] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "rank": self.rank,
            "manifest": self.manifest.to_dict(),
            "manifest_hash": self.manifest_hash,
            "chunks": [c.to_dict() for c in self.chunks],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DurableRankEntry":
        return cls(
            rank=int(d["rank"]),
            manifest=DurableFileRecord.from_dict(d["manifest"]),
            manifest_hash=str(d["manifest_hash"]),
            chunks=[DurableFileRecord.from_dict(c) for c in d.get("chunks", [])],
        )


@dataclass
class DurableCommitRecord:
    """Records that generation ``generation`` was fully promoted to durable.

    Written LAST in the promotion protocol; its presence (and validity) is the
    durability point.  It carries, per rank, the durable manifest record plus
    every uploaded chunk's key/size/checksum so restore can verify each file
    independently of any backend metadata.
    """

    generation: int
    n_ranks: int
    circuit_hash: str
    step_index: int               # circuit step that produced this gen
    parent_generation: int = -1
    # the local GlobalCommitRecord.commit_hash this was promoted from (provenance)
    source_commit_hash: str = ""
    ranks: dict[int, DurableRankEntry] = field(default_factory=dict)
    created: float = 0.0
    commit_hash: str = ""

    # ── hashing ───────────────────────────────────────────────────────

    def _hash_payload(self) -> dict:
        return {
            "generation": self.generation,
            "n_ranks": self.n_ranks,
            "circuit_hash": self.circuit_hash,
            "step_index": self.step_index,
            "parent_generation": self.parent_generation,
            "source_commit_hash": self.source_commit_hash,
            "ranks": {
                str(r): self.ranks[r].to_dict() for r in sorted(self.ranks)
            },
        }

    def compute_hash(self) -> str:
        return _stable_hash(self._hash_payload())

    def seal(self) -> "DurableCommitRecord":
        self.commit_hash = self.compute_hash()
        return self

    def verify_self_hash(self) -> bool:
        return bool(self.commit_hash) and \
            self.commit_hash == self.compute_hash()

    # ── structural validation ─────────────────────────────────────────

    def validate(self) -> None:
        if self.n_ranks < 1:
            raise ValueError(f"n_ranks must be >= 1, got {self.n_ranks}")
        missing = set(range(self.n_ranks)) - set(self.ranks)
        if missing:
            raise ValueError(
                f"durable commit gen {self.generation} missing ranks "
                f"{sorted(missing)}")

    # ── (de)serialization ─────────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "generation": self.generation,
            "n_ranks": self.n_ranks,
            "circuit_hash": self.circuit_hash,
            "step_index": self.step_index,
            "parent_generation": self.parent_generation,
            "source_commit_hash": self.source_commit_hash,
            "ranks": {
                str(r): e.to_dict() for r, e in sorted(self.ranks.items())
            },
            "created": self.created,
            "commit_hash": self.commit_hash,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "DurableCommitRecord":
        return cls(
            generation=int(d["generation"]),
            n_ranks=int(d["n_ranks"]),
            circuit_hash=str(d["circuit_hash"]),
            step_index=int(d["step_index"]),
            parent_generation=int(d.get("parent_generation", -1)),
            source_commit_hash=str(d.get("source_commit_hash", "")),
            ranks={
                int(r): DurableRankEntry.from_dict(e)
                for r, e in d.get("ranks", {}).items()
            },
            created=float(d.get("created", 0.0)),
            commit_hash=str(d.get("commit_hash", "")),
        )

    def to_json(self, *, indent: int | None = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)

    @classmethod
    def from_json(cls, raw: str | bytes) -> "DurableCommitRecord":
        if isinstance(raw, bytes):
            raw = raw.decode()
        return cls.from_dict(json.loads(raw))

    def to_bytes(self) -> bytes:
        return self.to_json().encode()


def list_durable_commit_keys(backend, run_id: str) -> list[str]:
    """All durable_commit_XXXXXX.json keys, sorted oldest→newest."""
    prefix = durable_commits_prefix(run_id)
    keys = [k for k in backend.list(prefix)
            if k.rsplit("/", 1)[-1].startswith("durable_commit_")
            and k.endswith(".json")]
    return sorted(keys)
