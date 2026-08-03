"""Shared algebraic building blocks for circulant Hadamard constructions."""
from __future__ import annotations

import numpy as np

try:
    from numba import njit
except ImportError:
    NUMBA_AVAILABLE = False
else:
    NUMBA_AVAILABLE = True


def symmetric_circulant(half: np.ndarray) -> np.ndarray:
    """Expand the independent half of an odd-length symmetric sequence."""
    return np.concatenate((half, half[-2::-1])).astype(np.int8)


def _autocorrelation_energy_numpy(sequences: tuple[np.ndarray, ...]) -> int:
    size = len(sequences[0])
    total = np.zeros(size, dtype=np.int64)
    for sequence in sequences:
        for displacement in range(size):
            total[displacement] += int(np.dot(sequence,
                                       np.roll(sequence, -displacement)))
    return int(np.sum(total[1:(size + 1) // 2] ** 2))


if NUMBA_AVAILABLE:
    @njit(cache=True)
    def _autocorrelation_energy_numba(
        first: np.ndarray,
        second: np.ndarray,
        third: np.ndarray,
        fourth: np.ndarray,
    ) -> int:
        size = len(first)
        total = np.zeros(size, dtype=np.int64)
        for sequence in (first, second, third, fourth):
            for displacement in range(size):
                correlation = 0
                for index in range(size):
                    correlation += sequence[index] * \
                        sequence[(index + displacement) % size]
                total[displacement] += correlation
        energy = 0
        for displacement in range(1, (size + 1) // 2):
            energy += total[displacement] * total[displacement]
        return energy


def autocorrelation_energy(sequences: tuple[np.ndarray, ...]) -> int:
    if NUMBA_AVAILABLE and len(sequences) == 4:
        return int(_autocorrelation_energy_numba(*sequences))
    return _autocorrelation_energy_numpy(sequences)


def periodic_autocorrelation_energy(sequences: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> int:
    size = len(sequences[0])
    energy = 0
    for displacement in range(1, size):
        correlation = sum(
            int(np.dot(sequence, np.roll(sequence, -displacement)))
            for sequence in sequences)
        energy += correlation * correlation
    return energy


def circulant(values: np.ndarray) -> np.ndarray:
    return np.array([np.roll(values, index) for index in range(len(values))], dtype=np.int8)


def build_williamson(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    A, B, C, D = (circulant(sequence) for sequence in (a, b, c, d))
    return np.block([[A, B, C, D], [-B, A, -D, C], [-C, D, A, -B], [-D, -C, B, A]]).astype(np.int8)


def build_propus(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    A, B, C, D = (circulant(sequence) for sequence in (a, b, c, d))
    return np.block([[A, B, C, D], [B, D, -A, -C], [C, -A, -D, B], [D, -C, B, -A]]).astype(np.int8)


def build_goethals_seidel(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    A, B, C, D = (circulant(sequence) for sequence in (a, b, c, d))
    reverse = np.fliplr(np.eye(len(a), dtype=np.int8))
    BR, CR, DR = B @ reverse, C @ reverse, D @ reverse
    return np.block([[A, BR, CR, DR], [-BR, A, -DR.T, CR.T], [-CR, DR.T, A, -BR.T], [-DR, -CR.T, BR.T, A]]).astype(np.int8)
