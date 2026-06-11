"""Diagonal MPI-nonlocal fast path + gate classification.

A gate that touches a rank/MPI bit normally forces an inter-rank ``Sendrecv``
(the paired amplitude lives on the partner rank).  But a **diagonal** gate only
multiplies each amplitude by a phase determined by its own basis index — it
never moves amplitudes between ranks.  So for diagonal MPI-nonlocal gates we can
apply the phase **locally**, using each amplitude's *global* basis index (rank
bits + chunk-index bits + local bits), with no exchange and no remote buffer.

Classification (pure, matrix-based — no circuit semantics changed):
  * ``diagonal``     — off-diagonal entries are ~0 (CZ, CR(k), RZ/T/S/Z/phase).
    ``requires_remote_amplitudes = False`` → fast path, no Sendrecv.
  * ``permutation``  — each row/col has exactly one non-zero (X, CNOT, SWAP).
    ``requires_remote_amplitudes = True`` (kept on the exchange path for now; a
    safe permutation path is a later optimization).
  * ``true_mixing``  — anything else (H, RX, RY on a rank bit). Mixes amplitudes
    across the rank boundary → ``requires_remote_amplitudes = True``.

Gate-matrix basis convention matches ``kernel.gates.gate_matrix``: for qubits
``qs`` (given order, MSB first) the basis index of an amplitude is
``sum(bit(qs[p]) << (len(qs)-1-p))``; a diagonal gate's phase is ``U[idx, idx]``.
"""
from __future__ import annotations

import numpy as np

_ATOL = 1e-9


def is_diagonal(U: np.ndarray) -> bool:
    U = np.asarray(U)
    if U.ndim != 2 or U.shape[0] != U.shape[1]:
        return False
    off = U - np.diag(np.diag(U))
    return bool(np.all(np.abs(off) <= _ATOL))


def is_permutation(U: np.ndarray) -> bool:
    U = np.asarray(U)
    if U.ndim != 2 or U.shape[0] != U.shape[1]:
        return False
    mag = np.abs(U)
    # exactly one entry of magnitude ~1 per row and per column, rest ~0
    rows_ok = np.all(np.sum(mag > 0.5, axis=1) == 1)
    cols_ok = np.all(np.sum(mag > 0.5, axis=0) == 1)
    one_hot = np.all((mag <= _ATOL) | (np.abs(mag - 1.0) <= 1e-6))
    return bool(rows_ok and cols_ok and one_hot)


def classify_nonlocal_gate(U: np.ndarray) -> tuple[str, bool]:
    """Return (kind, requires_remote_amplitudes) for an MPI-nonlocal gate.

    kind ∈ {"diagonal", "permutation", "true_mixing"}.  Only ``diagonal`` is
    safe to apply locally (requires_remote_amplitudes=False); everything else
    keeps the exchange path.
    """
    if is_diagonal(U):
        return "diagonal", False
    if is_permutation(U):
        return "permutation", True
    return "true_mixing", True


def _global_bit(q: int, k: int, n_local_bits: int, ci: int, rank: int,
                offsets: np.ndarray):
    """Bit value of qubit ``q`` for amplitudes at local offsets ``offsets``.

    Returns an int array (per element) for chunk-local qubits, or a constant
    0/1 for chunk-index / rank-bit qubits.
    """
    if q < k:                                   # chunk-local qubit: varies
        return (offsets >> q) & 1
    if q < k + n_local_bits:                     # chunk-index bit: constant/chunk
        return (ci >> (q - k)) & 1
    return (rank >> (q - k - n_local_bits)) & 1  # rank bit: constant/rank


def apply_diagonal_nonlocal_chunk(chunk: np.ndarray, ci: int, qs, U: np.ndarray,
                                  rank: int, k: int, n_local_bits: int) -> None:
    """Apply a diagonal gate to one local chunk in place, using global indices.

    No MPI, no remote buffer — multiplies each amplitude by ``U[idx, idx]`` where
    ``idx`` is built from the global bits of ``qs``.  Exact (same phase the dense
    gate would apply).
    """
    diagU = np.diag(np.asarray(U))
    nq = len(qs)
    N = len(chunk)
    # If no qubit is chunk-local, the phase is constant over the whole chunk.
    local_qs = [q for q in qs if q < k]
    if not local_qs:
        idx = 0
        for p, q in enumerate(qs):
            b = _global_bit(q, k, n_local_bits, ci, rank, np.empty(0))
            idx |= int(b) << (nq - 1 - p)
        chunk *= diagU[idx]
        return
    offs = np.arange(N, dtype=np.int64)
    idx = np.zeros(N, dtype=np.int64)
    for p, q in enumerate(qs):
        b = _global_bit(q, k, n_local_bits, ci, rank, offs)
        b = b if isinstance(b, np.ndarray) else np.int64(b)
        idx |= (np.asarray(b) << (nq - 1 - p))
    chunk *= diagU[idx]
