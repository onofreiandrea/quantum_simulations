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


def apply_1q_pair(c0: np.ndarray, c1: np.ndarray, U: np.ndarray) -> None:
    """1-qubit gate across two partner chunks.  Modifies both in-place."""
    a, b = c0.copy(), c1.copy()
    c0[:] = U[0, 0] * a + U[0, 1] * b
    c1[:] = U[1, 0] * a + U[1, 1] * b


def apply_2q_pair_qa_local(c0: np.ndarray, c1: np.ndarray,
                           qa: int, U: np.ndarray) -> None:
    """2-qubit gate: qa local, qb non-local."""
    N = len(c0)
    step = 1 << qa
    block = step << 1
    base = np.arange(0, N, block)
    off = np.arange(step)
    ia0 = (base[:, None] + off[None, :]).ravel()
    ia1 = ia0 + step
    V = np.stack([c0[ia0], c1[ia0], c0[ia1], c1[ia1]])
    R = U @ V
    c0[ia0], c1[ia0] = R[0], R[1]
    c0[ia1], c1[ia1] = R[2], R[3]


def apply_2q_pair_qb_local(c0: np.ndarray, c1: np.ndarray,
                           qb: int, U: np.ndarray) -> None:
    """2-qubit gate: qa non-local, qb local."""
    N = len(c0)
    step = 1 << qb
    block = step << 1
    base = np.arange(0, N, block)
    off = np.arange(step)
    ib0 = (base[:, None] + off[None, :]).ravel()
    ib1 = ib0 + step
    V = np.stack([c0[ib0], c0[ib1], c1[ib0], c1[ib1]])
    R = U @ V
    c0[ib0], c0[ib1] = R[0], R[1]
    c1[ib0], c1[ib1] = R[2], R[3]


def apply_2q_quad(c00: np.ndarray, c01: np.ndarray,
                  c10: np.ndarray, c11: np.ndarray,
                  U: np.ndarray) -> None:
    """2-qubit gate: both qubits non-local.  Element-wise across 4 chunks."""
    V = np.stack([c00, c01, c10, c11])
    R = U @ V
    c00[:], c01[:], c10[:], c11[:] = R[0], R[1], R[2], R[3]
