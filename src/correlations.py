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


@njit
# type: ignore[reportReturnType]
def nonperiodic_correlation_energy(correlations: np.ndarray):
    """Squared NPAF energy without the zero shift."""
    values = correlations.astype(np.int64)[1:]
    acc = np.int64(0)
    for i in range(len(values)):
        acc += values[i] * values[i]
    return acc


# ── Douglas-Rachford projections ──────────────────────────────────────────────

_POWER_EPS = 1e-12


def project_fourier(batch, *, lengths, weights):
    """Project onto the weighted NPAF power shell (3D batch only)."""
    from gpu import xp as _xp

    state = _xp.asarray(batch, dtype=_xp.float32)
    n = state.shape[2]
    fft_size = 2 * n - 1
    spectrum = _xp.fft.fft(state, n=fft_size, axis=-1)
    target = float(
        (_xp.asarray(lengths, dtype=_xp.float32) * _xp.asarray(weights, dtype=_xp.float32)).sum()
    )
    power = _xp.sum(
        _xp.asarray(weights, dtype=_xp.float32).reshape(1, len(weights), 1)
        * (spectrum.real**2 + spectrum.imag**2),
        axis=-2,
    )
    projected = _xp.fft.ifft(
        spectrum * _xp.sqrt(target / _xp.maximum(power, _POWER_EPS))[:, None, :], axis=-1
    ).real[:, :, :n]
    for idx, L in enumerate(lengths):
        projected[:, idx, int(L) :] = 0.0
    return projected


def project_sign(state):
    """Project onto {+-1}, zeroing the W-padding position."""
    from gpu import xp as _xp

    signs = _xp.where(state >= 0, _xp.float32(1), _xp.float32(-1))
    signs[:, 3, -1] = 0.0
    return signs
