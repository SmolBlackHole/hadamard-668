"""Periodic correlation state and exact local flip deltas."""
from __future__ import annotations

import numpy as np

try:
    from numba import njit
except ImportError:
    NUMBA_AVAILABLE = False
else:
    NUMBA_AVAILABLE = True


def expand_symmetric_sequence(half: np.ndarray) -> np.ndarray:
    """Expand the independent half of an odd-length symmetric sequence."""
    return np.concatenate((half, half[:0:-1])).astype(np.int8)


def _autocorrelation_state_numpy(
    sequences: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    size = sequences.shape[1]
    total = np.zeros(size, dtype=np.int64)
    for sequence_index, sequence in enumerate(sequences):
        for displacement in range(size):
            correlation = 0
            for index in range(size):
                correlation += int(sequence[index]) * int(
                    sequence[(index + displacement) % size])
            total[displacement] += int(weights[sequence_index]) * correlation
    return total


def _apply_sequence_flip_numpy(
    sequences: np.ndarray,
    correlations: np.ndarray,
    sequence_index: int,
    value_index: int,
    weight: int,
) -> None:
    size = sequences.shape[1]
    old_value = int(sequences[sequence_index, value_index])
    for displacement in range(1, size):
        forward = int(sequences[sequence_index,
                      (value_index + displacement) % size])
        backward = int(
            sequences[sequence_index, (value_index - displacement) % size])
        correlations[displacement] -= 2 * weight * \
            old_value * (forward + backward)
    sequences[sequence_index, value_index] = -old_value


if NUMBA_AVAILABLE:
    @njit(cache=True)
    def _autocorrelation_state_numba(
        sequences: np.ndarray,
        weights: np.ndarray,
    ) -> np.ndarray:
        size = sequences.shape[1]
        total = np.zeros(size, dtype=np.int64)
        for sequence_index in range(sequences.shape[0]):
            for displacement in range(size):
                correlation = 0
                for index in range(size):
                    correlation += sequences[sequence_index, index] * sequences[
                        sequence_index, (index + displacement) % size]
                total[displacement] += weights[sequence_index] * correlation
        return total

    @njit(cache=True)
    def _apply_sequence_flip_numba(
        sequences: np.ndarray,
        correlations: np.ndarray,
        sequence_index: int,
        value_index: int,
        weight: int,
    ) -> None:
        size = sequences.shape[1]
        old_value = sequences[sequence_index, value_index]
        for displacement in range(1, size):
            forward = sequences[sequence_index,
                                (value_index + displacement) % size]
            backward = sequences[sequence_index,
                                 (value_index - displacement) % size]
            correlations[displacement] -= 2 * weight * \
                old_value * (forward + backward)
        sequences[sequence_index, value_index] = -old_value


def _sequence_matrix(sequences: tuple[np.ndarray, ...] | np.ndarray) -> np.ndarray:
    matrix = np.asarray(sequences, dtype=np.int8)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("sequences must be a non-empty two-dimensional array")
    return np.ascontiguousarray(matrix)


def autocorrelation_state(
    sequences: tuple[np.ndarray, ...] | np.ndarray,
    weights: np.ndarray | None = None,
) -> np.ndarray:
    """Return the weighted sum of all periodic autocorrelations."""
    matrix = _sequence_matrix(sequences)
    actual_weights = (np.ones(matrix.shape[0], dtype=np.int64) if weights is None
                      else np.asarray(weights, dtype=np.int64))
    if actual_weights.shape != (matrix.shape[0],):
        raise ValueError("weights must contain one value per sequence")
    if NUMBA_AVAILABLE:
        return _autocorrelation_state_numba(matrix, actual_weights)
    return _autocorrelation_state_numpy(matrix, actual_weights)


def correlation_energy(correlations: np.ndarray) -> int:
    """Return squared periodic-correlation energy without the zero shift."""
    values = np.asarray(correlations, dtype=np.int64)[1:]
    return int(np.dot(values, values))


def periodic_autocorrelation_energy(
    sequences: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> int:
    """Return full nonzero-shift periodic correlation energy."""
    return correlation_energy(autocorrelation_state(sequences))


def apply_sequence_flip(
    sequences: np.ndarray,
    correlations: np.ndarray,
    sequence_index: int,
    value_index: int,
    *,
    weight: int = 1,
) -> int:
    """Flip one sequence value and update its correlation state in O(K)."""
    if sequences.ndim != 2 or correlations.shape != (sequences.shape[1],):
        raise ValueError("sequences and correlations have incompatible shapes")
    if not 0 <= sequence_index < sequences.shape[0]:
        raise IndexError("sequence_index is out of range")
    if not 0 <= value_index < sequences.shape[1]:
        raise IndexError("value_index is out of range")
    if NUMBA_AVAILABLE:
        _apply_sequence_flip_numba(
            sequences, correlations, sequence_index, value_index, weight)
    else:
        _apply_sequence_flip_numpy(
            sequences, correlations, sequence_index, value_index, weight)
    return correlation_energy(correlations)


def apply_symmetric_flip(
    sequences: np.ndarray,
    correlations: np.ndarray,
    sequence_index: int,
    half_index: int,
    *,
    weight: int = 1,
) -> int:
    """Flip one independent value of an odd symmetric cyclic sequence."""
    size = sequences.shape[1]
    half = (size + 1) // 2
    if size != 2 * half - 1:
        raise ValueError("symmetric flips require odd-length sequences")
    if not 0 <= half_index < half:
        raise IndexError("half_index is out of range")
    apply_sequence_flip(sequences, correlations,
                        sequence_index, half_index, weight=weight)
    if half_index:
        apply_sequence_flip(
            sequences, correlations, sequence_index, size - half_index, weight=weight)
    return correlation_energy(correlations)
