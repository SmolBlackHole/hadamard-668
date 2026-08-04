"""Non-periodic autocorrelation (NPAF) — the core math for Turyn search."""

from __future__ import annotations

import numpy as np
from numba import njit

from gpu import xp

TURYN_WEIGHTS = np.array((1.0, 1.0, 2.0, 2.0), dtype=np.float64)


def npa_f_residual(sequences, *, lengths, weights):
    """Weighted NPAF residual via FFT on the active GPU backend. O(n log n).

    Returns (batch, n-1) — correlations at shifts 1..n-1.
    """
    seqs = xp.asarray(sequences, dtype=xp.float32)
    if seqs.ndim not in (2, 3):
        raise ValueError("sequences must be (4, n) or (batch, 4, n)")
    had_batch = seqs.ndim == 3
    if not had_batch:
        seqs = seqs[None, ...]
    n = seqs.shape[2]
    fft_size = 2 * n - 1
    spectrum = xp.fft.fft(seqs, n=fft_size, axis=2)
    power = spectrum.real**2 + spectrum.imag**2
    w = xp.asarray(weights, dtype=xp.float32)
    total = xp.sum(w[None, :, None] * power, axis=1)
    result = xp.fft.ifft(total, axis=1).real[:, 1:n]
    return result[0] if not had_batch else result


@njit
def _npa_f_core(sequences, lengths, weights):
    """Weighted NPAF for ragged sequences. JIT-compiled O(n^2)."""
    m = sequences.astype(np.int64)
    n = int(lengths.max())
    total = np.zeros(n, dtype=np.int64)
    for si in range(len(sequences)):
        L = int(lengths[si])
        w = int(weights[si])
        for shift in range(1, L):
            acc = np.int64(0)
            for k in range(L - shift):
                acc += m[si, k] * m[si, k + shift]
            total[shift] += w * acc
    return total


def nonperiodic_autocorrelation_state(
    sequences: np.ndarray,
    *,
    lengths: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Weighted NPAF for ragged sequences. Exact integer arithmetic."""
    matrix = np.asarray(sequences, dtype=np.int8)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError("sequences must be a non-empty two-dimensional array")
    if lengths.shape != (matrix.shape[0],) or np.any(lengths < 1):
        raise ValueError("lengths must contain one positive value per sequence")
    if weights.shape != lengths.shape:
        raise ValueError("weights must contain one value per sequence")
    return _npa_f_core(matrix, lengths, weights)


def nonperiodic_correlation_energy(correlations: np.ndarray) -> int:
    """Squared NPAF energy without the zero shift."""
    values = np.asarray(correlations, dtype=np.int64)[1:]
    return int(np.dot(values, values))


def npa_f_gradient(
    sequences: np.ndarray, *, lengths: np.ndarray, weights: np.ndarray
) -> np.ndarray:
    """FFT gradient of the weighted NPAF energy. O(n log n).

    ``sequences``: (4, n) float64 array. Returns same shape.
    """
    n = int(lengths[0])
    fft_size = 2 * n - 1
    spectrum = np.fft.fft(sequences, n=fft_size, axis=1)
    autoco = np.fft.ifft(spectrum * spectrum.conj(), axis=1).real
    w = weights.astype(np.float64)
    total = w[0] * autoco[0] + w[1] * autoco[1] + w[2] * autoco[2] + w[3] * autoco[3]
    coeffs = np.zeros(fft_size, dtype=np.float64)
    coeffs[1:n] = total[1:n]
    coeffs[-(n - 1) :] = total[1:n][::-1]
    return (
        2.0 * w[:, None] * np.fft.ifft(np.fft.fft(coeffs)[None, :] * spectrum, axis=1).real[:, :n]
    )


def apply_nonperiodic_flip(
    sequences: np.ndarray,
    correlations: np.ndarray,
    sequence_index: int,
    value_index: int,
    *,
    lengths: np.ndarray,
    weight: int,
) -> int:
    """Flip one value and update NPAF state. O(L) vectorized."""
    if not 0 <= sequence_index < len(lengths):
        raise IndexError("sequence_index is out of range")
    L = int(lengths[sequence_index])
    if not 0 <= value_index < L:
        raise IndexError("value_index is out of range")
    seq = sequences[sequence_index]
    old = int(seq[value_index])
    j = value_index
    delta = 2 * weight * old
    nb = np.zeros(L, dtype=np.int64)
    if j + 1 < L:
        nb[1 : L - j] = seq[j + 1 : L].astype(np.int64)
    if j > 0:
        nb[1 : j + 1] += seq[:j][::-1].astype(np.int64)
    correlations[:L] -= delta * nb
    seq[j] = -old
    return nonperiodic_correlation_energy(correlations)
