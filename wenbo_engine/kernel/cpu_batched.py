"""CPU batched kernel: gather → GEMM → scatter.

Same interface as cpu_scalar but batches all pair/quad updates into a
single matrix multiply to maximise BLAS utilisation.
"""
from __future__ import annotations

import numpy as np
from wenbo_engine.kernel.cpu_scalar import check_local


def apply_1q(chunk: np.ndarray, qubit: int, U: np.ndarray) -> None:
    check_local(qubit, len(chunk))
    N = len(chunk)
    step = 1 << qubit
    block = step << 1
    base = np.arange(0, N, block)
    off = np.arange(step)
    idx0 = (base[:, None] + off[None, :]).ravel()
    idx1 = idx0 + step
    # gather into (2, M) matrix
    V = np.stack([chunk[idx0], chunk[idx1]])
    R = U @ V  # single GEMM
    chunk[idx0] = R[0]
    chunk[idx1] = R[1]


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
    V = np.stack([chunk[i00], chunk[i01], chunk[i10], chunk[i11]])  # (4, M)
    R = U @ V  # single GEMM
    chunk[i00], chunk[i01], chunk[i10], chunk[i11] = R[0], R[1], R[2], R[3]
