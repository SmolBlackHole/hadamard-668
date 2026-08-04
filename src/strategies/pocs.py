"""Douglas-Rachford projection for Turyn-type sequences."""

from __future__ import annotations

import time

import numpy as np

from correlations import _npa_f_residual
from gpu import check_orthogonality, to_numpy, xp

from .base import Result, TurynStrategy

_POWER_EPS = 1e-12  # numerical guard: avoid division by zero in the shell projection


def _project_fourier(
    batch: np.ndarray,
    *,
    lengths: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Project onto the weighted NPAF power shell (3D batch only)."""
    state = xp.asarray(batch, dtype=xp.float32)
    n = state.shape[2]
    fft_size = 2 * n - 1
    spectrum = xp.fft.fft(state, n=fft_size, axis=-1)
    target = float(
        (xp.asarray(lengths, dtype=xp.float32) * xp.asarray(weights, dtype=xp.float32)).sum()
    )
    power = xp.sum(
        xp.asarray(weights, dtype=xp.float32).reshape(1, len(weights), 1)
        * (spectrum.real**2 + spectrum.imag**2),
        axis=-2,
    )
    projected = xp.fft.ifft(
        spectrum * xp.sqrt(target / xp.maximum(power, _POWER_EPS))[:, None, :], axis=-1
    ).real[:, :, :n]
    for idx, L in enumerate(lengths):
        projected[:, idx, int(L) :] = 0.0
    return projected


def _project_sign(state: np.ndarray) -> np.ndarray:
    """Project onto {+-1}, zeroing the W-padding position."""
    signs = xp.where(state >= 0, xp.float32(1), xp.float32(-1))
    signs[:, 3, -1] = 0.0
    return signs


class PocsSearch(TurynStrategy):
    """Douglas-Rachford splitting between NPAF power shell and {+-1}."""

    def __init__(
        self,
        *,
        inner_steps: int = 50,
        n: int = TurynStrategy.DEFAULT_N,
        batch_size: int = 256,
        sieve: bool = True,
    ):
        super().__init__(n=n, sieve=sieve)
        self.inner_steps = inner_steps
        self.batch_size = batch_size

    @property
    def name(self) -> str:
        return "pocs"

    def search(self, steps: int, seed: int):
        started = time.perf_counter()
        rng = xp.random.default_rng(seed)
        batch_size = 1 if xp is np else self.batch_size  # single trajectory on CPU
        state = (
            self.seed_batch(batch_size, rng, module=xp).astype(xp.float32)
            if self.sieve
            else rng.standard_normal((batch_size, 4, self.N)).astype(xp.float32)
        )
        best = _project_sign(state)
        best_energy = xp.sum(
            _npa_f_residual(best, lengths=self.LENGTHS, weights=self.WEIGHTS, module=xp) ** 2,
            axis=1,
        )
        for _ in range(steps):
            for _ in range(self.inner_steps):
                # DR step: state += Ps(2*Pf(state) - state) - Pf(state), where Pf=power-shell, Ps=sign
                proj = _project_fourier(state, lengths=self.LENGTHS, weights=self.WEIGHTS)
                state = (state + _project_sign(2 * proj - state) - proj).astype(xp.float32)
            candidate = _project_sign(
                _project_fourier(state, lengths=self.LENGTHS, weights=self.WEIGHTS)
            )
            energy = xp.sum(
                _npa_f_residual(candidate, lengths=self.LENGTHS, weights=self.WEIGHTS, module=xp)
                ** 2,
                axis=1,
            )
            improved = energy < best_energy
            best = xp.where(improved[:, None, None], candidate, best)
            best_energy = xp.minimum(best_energy, energy)
        best_np = to_numpy(best[int(xp.argmin(best_energy).item())])
        matrix = self.build(best_np)[0]
        return Result(matrix, check_orthogonality(matrix), time.perf_counter() - started)
