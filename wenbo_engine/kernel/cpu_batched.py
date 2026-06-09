"""CPU batched kernel: gather → GEMM → scatter.

Uses numba JIT compilation for zero-copy, single-pass gate application.
Falls back to numpy if numba is not installed.
"""
from __future__ import annotations

import numpy as np
from wenbo_engine.kernel.cpu_scalar import check_local

try:
    import numba

    @numba.njit(cache=True, parallel=True)
    def _apply_1q_numba(chunk, qubit, u00, u01, u10, u11):
        step = 1 << qubit
        block = step << 1
        N = len(chunk)
        n_blocks = N >> (qubit + 1)
        for blk in numba.prange(n_blocks):
            base = blk * block
            for off in range(step):
                i = base + off
                j = i + step
                a = chunk[i]
                b = chunk[j]
                chunk[i] = u00 * a + u01 * b
                chunk[j] = u10 * a + u11 * b

    @numba.njit(cache=True, parallel=True)
    def _apply_2q_numba(chunk, qa, qb, U):
        # qa < qb guaranteed by caller
        C = 1 << qa
        B = 1 << (qb - qa - 1)
        N = len(chunk)
        A = N >> (qb + 1)
        total = A * B * C
        for idx in numba.prange(total):
            c = idx % C
            ab = idx // C
            b = ab % B
            a = ab // B
            i00 = (a << (qb + 1)) | (0 << qb) | (b << (qa + 1)) | (0 << qa) | c
            i01 = i00 | (1 << qb)
            i10 = i00 | (1 << qa)
            i11 = i00 | (1 << qa) | (1 << qb)
            s0 = chunk[i00]
            s1 = chunk[i01]
            s2 = chunk[i10]
            s3 = chunk[i11]
            chunk[i00] = U[0, 0]*s0 + U[0, 1]*s1 + U[0, 2]*s2 + U[0, 3]*s3
            chunk[i01] = U[1, 0]*s0 + U[1, 1]*s1 + U[1, 2]*s2 + U[1, 3]*s3
            chunk[i10] = U[2, 0]*s0 + U[2, 1]*s1 + U[2, 2]*s2 + U[2, 3]*s3
            chunk[i11] = U[3, 0]*s0 + U[3, 1]*s1 + U[3, 2]*s2 + U[3, 3]*s3

    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


def apply_1q(chunk: np.ndarray, qubit: int, U: np.ndarray) -> None:
    check_local(qubit, len(chunk))
    if _HAS_NUMBA:
        _apply_1q_numba(chunk, qubit, U[0, 0], U[0, 1], U[1, 0], U[1, 1])
        return
    step = 1 << qubit
    block = step << 1
    v = chunk.reshape(-1, block)
    lo = v[:, :step].copy()
    hi = v[:, step:]
    v[:, :step] = U[0, 0] * lo + U[0, 1] * hi
    v[:, step:] = U[1, 0] * lo + U[1, 1] * hi


def apply_2q(chunk: np.ndarray, qa: int, qb: int, U: np.ndarray) -> None:
    check_local(qa, len(chunk))
    check_local(qb, len(chunk))
    if qa > qb:
        qa, qb = qb, qa
        perm = [0, 2, 1, 3]
        U = U[np.ix_(perm, perm)]
    if _HAS_NUMBA:
        _apply_2q_numba(chunk, qa, qb, np.ascontiguousarray(U))
        return
    N = len(chunk)
    C = 1 << qa
    B = 1 << (qb - qa - 1)
    A = N >> (qb + 1)
    x = chunk.reshape(A, 2, B, 2, C)
    s00 = x[:, 0, :, 0, :].ravel()
    s01 = x[:, 1, :, 0, :].ravel()
    s10 = x[:, 0, :, 1, :].ravel()
    s11 = x[:, 1, :, 1, :].ravel()
    V = np.stack([s00, s01, s10, s11])
    R = U @ V
    shape = (A, B, C)
    x[:, 0, :, 0, :] = R[0].reshape(shape)
    x[:, 1, :, 0, :] = R[1].reshape(shape)
    x[:, 0, :, 1, :] = R[2].reshape(shape)
    x[:, 1, :, 1, :] = R[3].reshape(shape)
