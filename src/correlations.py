"""Non-periodic autocorrelation (NPAF) — the core math for Turyn search."""

from __future__ import annotations

import numpy as np


def nonperiodic_autocorrelation_state(
    sequences: np.ndarray,
    *,
    lengths: np.ndarray | None = None,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Weighted non-periodic autocorrelations for ragged sign sequences."""
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
    for seq, L, w in zip(matrix, actual_lengths, actual_weights):
        for s in range(1, int(L)):
            total[s] += int(w) * int(np.dot(seq[: L - s], seq[s:L]))
    return total


def nonperiodic_correlation_energy(correlations: np.ndarray) -> int:
    """Squared non-periodic correlation energy without the zero shift."""
    values = np.asarray(correlations, dtype=np.int64)[1:]
    return int(np.dot(values, values))


def nonperiodic_batch_energy(
    population: np.ndarray,
    *,
    lengths: np.ndarray,
    weights: np.ndarray,
    module=np,
) -> np.ndarray:
    """Weighted NPAF energies for a (batch, sequences, values) array."""
    values = module.asarray(population)
    if values.ndim != 3:
        raise ValueError("population must have shape (batch, sequences, values)")
    fft_size = 2 * values.shape[2] - 1
    spectrum = module.fft.fft(values, n=fft_size, axis=2)
    corr = module.fft.ifft(module.abs(spectrum) ** 2, axis=2).real
    w = module.asarray(weights, dtype=values.dtype)
    total = module.sum(w[None, :, None] * corr, axis=1)
    return module.sum(total[:, 1 : values.shape[2]] ** 2, axis=1)


def apply_nonperiodic_flip(
    sequences: np.ndarray,
    correlations: np.ndarray,
    sequence_index: int,
    value_index: int,
    *,
    lengths: np.ndarray,
    weight: int = 1,
) -> int:
    """Flip one ragged-sequence value and update NPAF state in O(max length)."""
    if not 0 <= sequence_index < len(lengths):
        raise IndexError("sequence_index is out of range")
    L = int(lengths[sequence_index])
    if not 0 <= value_index < L:
        raise IndexError("value_index is out of range")
    old = int(sequences[sequence_index, value_index])
    for s in range(1, L):
        neighbours = 0
        if value_index + s < L:
            neighbours += int(sequences[sequence_index, value_index + s])
        if value_index >= s:
            neighbours += int(sequences[sequence_index, value_index - s])
        correlations[s] -= 2 * weight * old * neighbours
    sequences[sequence_index, value_index] = -old
    return nonperiodic_correlation_energy(correlations)
