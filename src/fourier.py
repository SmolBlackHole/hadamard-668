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
    module=np,
) -> np.ndarray:
    """Project one state or a batch onto the weighted non-periodic power shell."""
    state = module.asarray(sequences, dtype=module.float32)
    if state.ndim not in (2, 3) or state.shape[-2] != len(lengths):
        raise ValueError(
            "sequences must have shape (sequences, values) or (batch, sequences, values)")
    fft_size = 2 * state.shape[-1] - 1
    spectrum = module.fft.fft(state, n=fft_size, axis=-1)
    actual_lengths = module.asarray(lengths)
    actual_weights = module.asarray(weights, dtype=state.dtype)
    target = module.sum(actual_lengths * actual_weights)
    shape = (1,) * (state.ndim - 2) + (len(weights), 1)
    power = module.sum(actual_weights.reshape(shape) *
                       module.abs(spectrum) ** 2, axis=-2)
    projected = module.fft.ifft(
        spectrum * module.sqrt(target / module.maximum(power, 1e-12))[..., None, :], axis=-1).real[..., :state.shape[-1]]
    for index, length in enumerate(lengths):
        projected[..., index, int(length):] = 0.0
    return projected
