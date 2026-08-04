"""Non-periodic autocorrelation (NPAF) — the core math for Turyn search."""

from __future__ import annotations

import numpy as np

TURYN_WEIGHTS_I = np.array((1, 1, 2, 2), dtype=np.int64)
TURYN_WEIGHTS_F = np.array((1.0, 1.0, 2.0, 2.0), dtype=np.float64)


def _npa_f_residual(sequences, *, lengths, weights, module=np):
    """Weighted NPAF residual via FFT. O(n log n).

    Returns (batch, n-1) — correlations at shifts 1..n-1.
    """
    seqs = module.asarray(sequences, dtype=module.float32)
    if seqs.ndim not in (2, 3):
        raise ValueError("sequences must be (4, n) or (batch, 4, n)")
    had_batch = seqs.ndim == 3
    if not had_batch:
        seqs = seqs[None, ...]
    n = seqs.shape[2]
    fft_size = 2 * n - 1
    spectrum = module.fft.fft(seqs, n=fft_size, axis=2)
    power = spectrum.real**2 + spectrum.imag**2
    w = module.asarray(weights, dtype=module.float32)
    total = module.sum(w[None, :, None] * power, axis=1)
    result = module.fft.ifft(total, axis=1).real[:, 1:n]
    return result[0] if not had_batch else result


def nonperiodic_autocorrelation_state(
    sequences: np.ndarray,
    *,
    lengths: np.ndarray | None = None,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Weighted NPAF for ragged sequences. Exact integer arithmetic."""
    matrix = np.asarray(sequences, dtype=np.int8)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError("sequences must be a non-empty two-dimensional array")
    actual_lengths = (
        np.full(matrix.shape[0], matrix.shape[1], dtype=np.int64)
        if lengths is None
        else np.asarray(lengths, dtype=np.int64)
    )
    if actual_lengths.shape != (matrix.shape[0],) or np.any(actual_lengths < 1):
        raise ValueError("lengths must contain one positive value per sequence")
    actual_weights = (
        np.ones(matrix.shape[0], dtype=np.int64)
        if weights is None
        else np.asarray(weights, dtype=np.int64)
    )
    if actual_weights.shape != actual_lengths.shape:
        raise ValueError("weights must contain one value per sequence")
    n = int(actual_lengths.max())
    total = np.zeros(n, dtype=np.int64)
    for seq, L, w in zip(matrix, actual_lengths, actual_weights, strict=False):
        s64 = seq.astype(np.int64)
        for s in range(1, int(L)):
            total[s] += int(w) * int(np.dot(s64[: L - s], s64[s:L]))
    return total


def nonperiodic_correlation_energy(correlations: np.ndarray) -> int:
    """Squared NPAF energy without the zero shift."""
    values = np.asarray(correlations, dtype=np.int64)[1:]
    return int(np.dot(values, values))


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
