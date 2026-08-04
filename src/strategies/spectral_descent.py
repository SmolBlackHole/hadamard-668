"""FFT-guided local search via GPU batch — gradient-ranked multi-trajectory."""

from __future__ import annotations

import time

import numpy as np

from correlations import _npa_f_residual
from gpu import xp

from .base import Result, TurynStrategy


def _apply_flips(batch, correlations, energies, seq_idx, col_idx, lengths):
    """Apply flips in-place. correlations shape is (batch, n-1): index k = shift k+1."""
    weights = xp.asarray((1, 1, 2, 2), dtype=xp.int64)
    for i in range(len(batch)):
        si, ci = int(seq_idx[i]), int(col_idx[i])
        wgt, old = int(weights[si]), int(batch[i, si, ci])
        L = int(lengths[si])
        for shift in range(1, L):
            nb = 0
            if ci + shift < L:
                nb += int(batch[i, si, ci + shift])
            if ci >= shift:
                nb += int(batch[i, si, ci - shift])
            correlations[i, shift - 1] -= 2 * wgt * old * nb
        batch[i, si, ci] = -old
    energies[:] = xp.sum(correlations**2, axis=1)


class TurynSpectralDescentSearch(TurynStrategy):
    """FFT-gradient guided search over many parallel GPU trajectories."""

    def __init__(
        self,
        *,
        n: int = TurynStrategy.DEFAULT_N,
        batch_size: int = 256,
        sieve: bool = True,
    ):
        super().__init__(n=n, sieve=sieve)
        self.batch_size = batch_size

    @property
    def name(self) -> str:
        return "spectral_descent"

    def gradient(self, sequences: np.ndarray) -> np.ndarray:
        """FFT gradient of the weighted NPAF energy. O(n log n)."""
        fft_size = 2 * self.N - 1
        spectrum = np.fft.fft(sequences, n=fft_size, axis=1)
        autoco = np.fft.ifft(spectrum * spectrum.conj(), axis=1).real
        w = self.WEIGHTS.astype(np.float64)
        total = np.tensordot(w, autoco, axes=1)
        coeffs = np.zeros(fft_size, dtype=np.float64)
        coeffs[1 : self.N] = total[1 : self.N]
        coeffs[-(self.N - 1) :] = total[1 : self.N][::-1]
        grad = 2.0 * w[:, None] * np.fft.ifft(np.fft.fft(coeffs)[None, :] * spectrum, axis=1).real
        return grad[:, : self.N]

    def search(self, steps: int, seed: int) -> Result:
        started = time.perf_counter()
        rng = xp.random.default_rng(seed)
        batch = self.seed_batch(self.batch_size, rng, module=xp)
        correlations = _npa_f_residual(batch, lengths=self.LENGTHS, weights=self.WEIGHTS, module=xp)
        energies = xp.sum(correlations**2, axis=1)

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

        best = xp.asnumpy(batch[int(energies.argmin().get())])
        matrix, metrics = self.build(best)
        return Result(matrix, metrics, time.perf_counter() - started)
