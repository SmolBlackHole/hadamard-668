"""Shared Fourier projections for cyclic sequence strategies."""
from __future__ import annotations

import numpy as np


def project_power_complementarity(sequences: np.ndarray) -> np.ndarray:
    """Project equal-length sequences onto their common Fourier power shell."""
    state = np.asarray(sequences, dtype=np.float64)
    spectrum = np.fft.fft(state, axis=1)
    target = state.shape[0] * state.shape[1]
    power = np.sum(np.abs(spectrum) ** 2, axis=0)
    scale = np.sqrt(target / np.maximum(power, 1e-12))
    return np.fft.ifft(spectrum * scale[None, :], axis=1).real


def project_weighted_nonperiodic_power(
    sequences: np.ndarray,
    *,
    lengths: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Project padded sequences onto the weighted non-periodic power shell."""
    state = np.asarray(sequences, dtype=np.float64)
    fft_size = 2 * state.shape[1] - 1
    spectrum = np.fft.fft(state, n=fft_size, axis=1)
    target = float(np.dot(lengths, weights))
    power = np.sum(weights[:, None] * np.abs(spectrum) ** 2, axis=0)
    projected = np.fft.ifft(
        spectrum * np.sqrt(target / np.maximum(power, 1e-12))[None, :], axis=1).real[:, :state.shape[1]]
    for index, length in enumerate(lengths):
        projected[index, int(length):] = 0.0
    return projected
