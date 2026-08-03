"""Shared Fourier-space projections for complementary sequences."""
from __future__ import annotations

import numpy as np

from correlations import expand_symmetric_sequence


def project_power_complementarity(state: np.ndarray) -> np.ndarray:
    """Project four real sequences onto the Fourier power-complementarity set."""
    values = np.asarray(state)
    if values.ndim != 2 or values.shape[0] != 4:
        raise ValueError("state must have shape (4, K)")
    dtype = np.float32 if values.dtype == np.float32 else np.float64
    values = values.astype(dtype, copy=False)
    size = values.shape[1]
    spectrum = np.fft.fft(values, axis=1)
    norms = np.sqrt(np.sum(np.abs(spectrum) ** 2, axis=0))
    target = np.sqrt(4.0 * size)
    nonzero = norms > 1e-12
    spectrum[:, nonzero] *= target / norms[nonzero]
    spectrum[:, ~nonzero] = 0.0
    spectrum[0, ~nonzero] = target
    return np.fft.ifft(spectrum, axis=1).real.astype(dtype)


def _autocorrelation_gradient(sequences: np.ndarray) -> np.ndarray:
    values = np.asarray(sequences, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != 4:
        raise ValueError("sequences must have shape (4, K)")
    size = values.shape[1]
    spectrum = np.fft.fft(values, axis=1)
    violation = np.sum(np.abs(spectrum) ** 2, axis=0) - 4.0 * size
    return 2.0 / size * np.fft.ifft(
        violation[None, :] * spectrum, axis=1).real


def half_sequence_gradient(half_sequences: list[np.ndarray]) -> np.ndarray:
    """Return the power-complementarity gradient for symmetric half sequences."""
    full = np.stack([expand_symmetric_sequence(half)
                    for half in half_sequences])
    full_gradient = _autocorrelation_gradient(full)
    size = full_gradient.shape[1]
    half = (size + 1) // 2
    gradient = np.zeros((4, half), dtype=np.float64)
    gradient[:, 0] = full_gradient[:, 0]
    gradient[:, 1:] = full_gradient[:, 1:half] + \
        full_gradient[:, size - 1:size - half:-1]
    return gradient
