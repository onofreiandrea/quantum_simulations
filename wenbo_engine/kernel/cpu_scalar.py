"""CPU scalar kernel: vectorised numpy, one gate at a time.

Operates in-place on a chunk (1-D complex64/128 array).
Qubit indices are chunk-local (qubit q ↔ bit q of the local index).
"""
from __future__ import annotations

import math

import numpy as np


def check_local(qubit: int, chunk_len: int) -> None:
    k = int(math.log2(chunk_len))
    if qubit >= k:
        raise NotImplementedError(
            f"qubit {qubit} >= log2(chunk_size)={k}: non-local gate requires layout/collect step"
        )


def apply_1q(chunk: np.ndarray, qubit: int, U: np.ndarray) -> None:
    check_local(qubit, len(chunk))
    N = len(chunk)
    step = 1 << qubit
    block = step << 1
    base = np.arange(0, N, block)
    off = np.arange(step)
    idx0 = (base[:, None] + off[None, :]).ravel()
    idx1 = idx0 + step
    a, b = chunk[idx0].copy(), chunk[idx1].copy()
    chunk[idx0] = U[0, 0] * a + U[0, 1] * b
    chunk[idx1] = U[1, 0] * a + U[1, 1] * b


def apply_2q(chunk: np.ndarray, qa: int, qb: int, U: np.ndarray) -> None:
    check_local(qa, len(chunk))
    check_local(qb, len(chunk))
    N = len(chunk)
    idx = np.arange(N)
    bases = idx[((idx >> qa) & 1 == 0) & ((idx >> qb) & 1 == 0)]
    i00 = bases
    i01 = bases | (1 << qb)
    i10 = bases | (1 << qa)
    i11 = bases | (1 << qa) | (1 << qb)
    v = np.stack([chunk[i00], chunk[i01], chunk[i10], chunk[i11]])  # (4, M)
    r = U @ v
    chunk[i00], chunk[i01], chunk[i10], chunk[i11] = r[0], r[1], r[2], r[3]
