"""FFT-guided local search via GPU batch — gradient-ranked multi-trajectory."""

from __future__ import annotations

import time

import numpy as np

from gpu import xp
from sieve import seed_turyn_batch

from .base import Result, TurynStrategy


def _gpu_correlations(batch):
    w = xp.asarray([1, 1, 2, 2], dtype=xp.int64)
    fft_size = 2 * batch.shape[2] - 1
    spectrum = xp.fft.fft(batch, n=fft_size, axis=2)
    corr = xp.fft.ifft(xp.abs(spectrum) ** 2, axis=2).real
    return xp.sum(w[None, :, None] * corr, axis=1)


def _gpu_energies_from_correlations(correlations):
    return xp.sum(correlations[:, 1 : correlations.shape[1]] ** 2, axis=1)


def _gpu_apply_flips(batch, correlations, energies, seq_idx, col_idx, lengths):
    w = xp.asarray([1, 1, 2, 2], dtype=xp.int64)
    for i in range(len(batch)):
        si = int(seq_idx[i])
        ci = int(col_idx[i])
        weight = int(w[si])
        old = int(batch[i, si, ci])
        L = int(lengths[si])
        for s in range(1, L):
            n = 0
            if ci + s < L:
                n += int(batch[i, si, ci + s])
            if ci >= s:
                n += int(batch[i, si, ci - s])
            correlations[i, s] -= 2 * weight * old * n
        batch[i, si, ci] = -old
    energies[:] = _gpu_energies_from_correlations(correlations)


class TurynSpectralDescentSearch(TurynStrategy):
    """FFT-gradient guided search over many parallel GPU trajectories."""

    def __init__(
        self,
        *,
        n: int = TurynStrategy.DEFAULT_N,
        gradient_interval: int = 7,
        candidates: int = 8,
        batch_size: int = 256,
        sieve: bool = True,
    ):
        super().__init__(n=n, sieve=sieve)
        if gradient_interval < 1 or candidates < 1:
            raise ValueError("gradient_interval and candidates must be positive")
        self.gradient_interval = gradient_interval
        self.candidates = candidates
        self.batch_size = batch_size

    @property
    def name(self) -> str:
        return "spectral_descent"

    def gradient(self, sequences: np.ndarray) -> np.ndarray:
        """FFT gradient of the weighted NPAF energy. O(n log n)."""
        fft_size = 2 * self.N - 1
        spectrum = np.fft.fft(sequences, n=fft_size, axis=1)
        correlations = np.fft.ifft(spectrum * spectrum.conj(), axis=1).real
        w = self.WEIGHTS.astype(np.float64)
        total = np.tensordot(w, correlations, axes=1)
        coefficients = np.zeros(fft_size, dtype=np.float64)
        coefficients[1 : self.N] = total[1 : self.N]
        coefficients[-(self.N - 1) :] = total[1 : self.N][::-1]
        grad = (
            2.0
            * w[:, None]
            * np.fft.ifft(np.fft.fft(coefficients)[None, :] * spectrum, axis=1).real
        )
        return grad[:, : self.N]

    def search(self, steps: int, seed: int) -> Result:
        started = time.perf_counter()
        rng = xp.random.default_rng(seed)
        batch = (
            seed_turyn_batch(self.N, self.batch_size, rng, module=xp)
            if self.sieve
            else rng.integers(0, 2, size=(self.batch_size, 4, self.N), dtype=xp.int8) * 2 - 1
        )
        batch[:, 3, -1] = 0
        correlations = _gpu_correlations(batch)
        energies = _gpu_energies_from_correlations(correlations)
        for step in range(steps):
            if step % self.gradient_interval == 0:
                fft_size = 2 * self.N - 1
                spectrum = xp.fft.fft(batch, n=fft_size, axis=2)
                autocorrelations = xp.fft.ifft(xp.abs(spectrum) ** 2, axis=2).real
                total = xp.sum(xp.asarray(self.WEIGHTS)[None, :, None] * autocorrelations, axis=1)
                coefficients = xp.zeros_like(total)
                coefficients[:, 1 : self.N] = total[:, 1 : self.N]
                coefficients[:, -(self.N - 1) :] = total[:, 1 : self.N][:, ::-1]
                gradient = (
                    2
                    * xp.asarray(self.WEIGHTS)[None, :, None]
                    * xp.fft.ifft(
                        xp.fft.fft(coefficients, axis=1)[:, None, :] * spectrum, axis=2
                    ).real[:, :, : self.N]
                )
                score = -2 * batch * gradient
                score[:, 3, -1] = xp.inf
                choice = xp.argmin(score.reshape(self.batch_size, -1), axis=1)
                sequence = (choice // self.N).astype(xp.int64)
                column = (choice % self.N).astype(xp.int64)
            else:
                position = rng.integers(0, int(self.LENGTHS.sum()), size=self.batch_size)
                sequence = xp.minimum(position // self.N, 3).astype(xp.int64)
                column = (position - sequence * self.N).astype(xp.int64)
            _gpu_apply_flips(batch, correlations, energies, sequence, column, self.LENGTHS)
        best = xp.asnumpy(batch[int(energies.argmin().get())])
        matrix, metrics = self.build(best)
        return Result(matrix, metrics, time.perf_counter() - started)
