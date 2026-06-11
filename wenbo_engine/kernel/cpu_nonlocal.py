"""Butterfly-exchange kernels for non-local gates (pure array ops, no I/O).

When a gate touches qubit q >= log2(chunk_size), the paired amplitudes
live in different chunks.  These functions operate on in-memory arrays
that have already been loaded by the caller.

Terminology:
    k = log2(chunk_size)          local qubits: 0 .. k-1
    partner_bit = q - k           bit position *within the chunk index*
    chunk c  pairs with  c XOR (1 << partner_bit)

Three cases for 2-qubit gates (qa = qubits[0], qb = qubits[1]):
    A) qa local,  qb non-local  →  pair on qb, local pairs on qa
    B) qa non-local, qb local   →  pair on qa, local pairs on qb
    C) both non-local            →  quad of chunks, element-wise 4×4
"""
from __future__ import annotations

import numpy as np

from wenbo_engine.kernel import backend

try:
    import numba

    @numba.njit(cache=True, parallel=True)
    def _apply_1q_pair_numba(c0, c1, u00, u01, u10, u11):
        N = len(c0)
        for i in numba.prange(N):
            a = c0[i]
            b = c1[i]
            c0[i] = u00 * a + u01 * b
            c1[i] = u10 * a + u11 * b

    @numba.njit(cache=True, parallel=True)
    def _apply_2q_pair_qa_local_numba(c0, c1, qa, U):
        step = 1 << qa
        block = step << 1
        N = len(c0)
        n_blocks = N // block
        for blk in numba.prange(n_blocks):
            base = blk * block
            for off in range(step):
                i = base + off          # lo index within block
                j = i + step            # hi index within block
                # s00=c0[i], s01=c1[i], s10=c0[j], s11=c1[j]
                s0 = c0[i]
                s1 = c1[i]
                s2 = c0[j]
                s3 = c1[j]
                c0[i] = U[0,0]*s0 + U[0,1]*s1 + U[0,2]*s2 + U[0,3]*s3
                c1[i] = U[1,0]*s0 + U[1,1]*s1 + U[1,2]*s2 + U[1,3]*s3
                c0[j] = U[2,0]*s0 + U[2,1]*s1 + U[2,2]*s2 + U[2,3]*s3
                c1[j] = U[3,0]*s0 + U[3,1]*s1 + U[3,2]*s2 + U[3,3]*s3

    @numba.njit(cache=True, parallel=True)
    def _apply_2q_pair_qb_local_numba(c0, c1, qb, U):
        step = 1 << qb
        block = step << 1
        N = len(c0)
        n_blocks = N // block
        for blk in numba.prange(n_blocks):
            base = blk * block
            for off in range(step):
                i = base + off
                j = i + step
                # s00=c0[i], s01=c0[j], s10=c1[i], s11=c1[j]
                s0 = c0[i]
                s1 = c0[j]
                s2 = c1[i]
                s3 = c1[j]
                c0[i] = U[0,0]*s0 + U[0,1]*s1 + U[0,2]*s2 + U[0,3]*s3
                c0[j] = U[1,0]*s0 + U[1,1]*s1 + U[1,2]*s2 + U[1,3]*s3
                c1[i] = U[2,0]*s0 + U[2,1]*s1 + U[2,2]*s2 + U[2,3]*s3
                c1[j] = U[3,0]*s0 + U[3,1]*s1 + U[3,2]*s2 + U[3,3]*s3

    @numba.njit(cache=True, parallel=True)
    def _apply_2q_quad_numba(c00, c01, c10, c11, U):
        N = len(c00)
        for i in numba.prange(N):
            s0 = c00[i]
            s1 = c01[i]
            s2 = c10[i]
            s3 = c11[i]
            c00[i] = U[0,0]*s0 + U[0,1]*s1 + U[0,2]*s2 + U[0,3]*s3
            c01[i] = U[1,0]*s0 + U[1,1]*s1 + U[1,2]*s2 + U[1,3]*s3
            c10[i] = U[2,0]*s0 + U[2,1]*s1 + U[2,2]*s2 + U[2,3]*s3
            c11[i] = U[3,0]*s0 + U[3,1]*s1 + U[3,2]*s2 + U[3,3]*s3

    _HAS_NUMBA = True
except ImportError:
    _HAS_NUMBA = False


def apply_1q_pair(c0: np.ndarray, c1: np.ndarray, U: np.ndarray) -> None:
    """1-qubit gate across two partner chunks.  Modifies both in-place."""
    if backend.use_numba():
        _apply_1q_pair_numba(c0, c1, U[0, 0], U[0, 1], U[1, 0], U[1, 1])
        return
    tmp = U[0, 0] * c0 + U[0, 1] * c1
    c1[:] = U[1, 0] * c0 + U[1, 1] * c1
    c0[:] = tmp


def apply_2q_pair_qa_local(c0: np.ndarray, c1: np.ndarray,
                           qa: int, U: np.ndarray) -> None:
    """2-qubit gate: qa local, qb non-local."""
    if backend.use_numba():
        _apply_2q_pair_qa_local_numba(c0, c1, qa, np.ascontiguousarray(U))
        return
    step = 1 << qa
    block = step << 1
    v0 = c0.reshape(-1, block)
    v1 = c1.reshape(-1, block)
    s00 = v0[:, :step].ravel()
    s01 = v1[:, :step].ravel()
    s10 = v0[:, step:].ravel()
    s11 = v1[:, step:].ravel()
    V = np.stack([s00, s01, s10, s11])
    R = U @ V
    shape = v0[:, :step].shape
    v0[:, :step] = R[0].reshape(shape)
    v1[:, :step] = R[1].reshape(shape)
    v0[:, step:] = R[2].reshape(shape)
    v1[:, step:] = R[3].reshape(shape)


def apply_2q_pair_qb_local(c0: np.ndarray, c1: np.ndarray,
                           qb: int, U: np.ndarray) -> None:
    """2-qubit gate: qa non-local, qb local."""
    if backend.use_numba():
        _apply_2q_pair_qb_local_numba(c0, c1, qb, np.ascontiguousarray(U))
        return
    step = 1 << qb
    block = step << 1
    v0 = c0.reshape(-1, block)
    v1 = c1.reshape(-1, block)
    s00 = v0[:, :step].ravel()
    s01 = v0[:, step:].ravel()
    s10 = v1[:, :step].ravel()
    s11 = v1[:, step:].ravel()
    V = np.stack([s00, s01, s10, s11])
    R = U @ V
    shape = v0[:, :step].shape
    v0[:, :step] = R[0].reshape(shape)
    v0[:, step:] = R[1].reshape(shape)
    v1[:, :step] = R[2].reshape(shape)
    v1[:, step:] = R[3].reshape(shape)


def apply_2q_quad(c00: np.ndarray, c01: np.ndarray,
                  c10: np.ndarray, c11: np.ndarray,
                  U: np.ndarray) -> None:
    """2-qubit gate: both qubits non-local.  Element-wise across 4 chunks."""
    if backend.use_numba():
        _apply_2q_quad_numba(c00, c01, c10, c11, np.ascontiguousarray(U))
        return
    V = np.stack([c00, c01, c10, c11])
    R = U @ V
    c00[:], c01[:], c10[:], c11[:] = R[0], R[1], R[2], R[3]
