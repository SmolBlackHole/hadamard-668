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
