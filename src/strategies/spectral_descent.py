"""FFT-guided local search via GPU batch — gradient-ranked multi-trajectory."""

from __future__ import annotations

import time

from correlations import TURYN_WEIGHTS, npa_f_residual
from gpu import to_numpy, xp

from .base import Result, TurynStrategy


def _apply_flips(batch, correlations, energies, seq_idx, col_idx, lengths):
    """Apply flips via vectorized indexing, then recompute from FFT."""
    weights = xp.asarray(TURYN_WEIGHTS, dtype=xp.float32)
    batch_idx = xp.arange(len(batch), dtype=xp.int64)
    batch[batch_idx, seq_idx.astype(xp.int64), col_idx.astype(xp.int64)] *= -1
    # Recompute via FFT — faster on GPU than per-element incremental update
    new_corr = npa_f_residual(batch, lengths=lengths, weights=weights)
    correlations[:] = new_corr
    energies[:] = xp.sum(correlations**2, axis=1)


class TurynSpectralDescentSearch(TurynStrategy):
    """FFT-gradient guided search over many parallel GPU trajectories."""

    gpu_exclusive = True

    def __init__(self, *, n: int, batch_size: int = 256, sieve: bool = True):
        super().__init__(n=n, sieve=sieve)
        self.batch_size = batch_size

    @classmethod
    def from_order(cls, order: int) -> TurynSpectralDescentSearch:
        n = (order // 4 + 1) // 3
        if order != 4 * (3 * n - 1) or n < 2:
            raise ValueError(f"order {order} has no TT(n) construction")
        return cls(n=n)

    @property
    def name(self) -> str:
        return "spectral_descent"

    def search(self, steps: int, seed: int) -> Result:
        started = time.perf_counter()
        rng = xp.random.default_rng(seed)
        batch = self.seed_batch(self.batch_size, rng, module=xp)
        correlations = npa_f_residual(batch, lengths=self.LENGTHS, weights=self.WEIGHTS)
        energies = xp.sum(correlations**2, axis=1)

        best_energy = float("inf")
        best_state = None

        for _ in range(steps):
            # FFT gradient: rank every position by energy-improvement proxy
            fft_size = 2 * self.N - 1
            spectrum = xp.fft.fft(batch, n=fft_size, axis=2)
            autoco = xp.fft.ifft(xp.abs(spectrum) ** 2, axis=2).real
            total = xp.sum(xp.asarray(self.WEIGHTS)[None, :, None] * autoco, axis=1)
            coeffs = xp.zeros_like(total)
            coeffs[:, 1 : self.N] = total[:, 1 : self.N]
            coeffs[:, -(self.N - 1) :] = total[:, 1 : self.N][:, ::-1]
            grad = (
                2
                * xp.asarray(self.WEIGHTS)[None, :, None]
                * xp.fft.ifft(xp.fft.fft(coeffs, axis=1)[:, None, :] * spectrum, axis=2).real[
                    :, :, : self.N
                ]
            )
            score = -2 * batch * grad
            score[:, 3, -1] = xp.inf  # exclude padding position of W sequence
            choice = xp.argmin(score.reshape(self.batch_size, -1), axis=1)
            sequence = (choice // self.N).astype(xp.int64)
            column = (choice % self.N).astype(xp.int64)
            _apply_flips(batch, correlations, energies, sequence, column, self.LENGTHS)

            batch_idx = int(energies.argmin())
            if float(energies[batch_idx]) < best_energy:
                best_energy = float(energies[batch_idx])
                best_state = to_numpy(batch[batch_idx]).copy()

        if best_state is None:
            best_state = to_numpy(batch[int(energies.argmin())])
        matrix, metrics = self.build(best_state)
        return Result(
            matrix=matrix,
            metrics=metrics,
            elapsed=time.perf_counter() - started,
            seed=seed,
            sequences=best_state,
        )
