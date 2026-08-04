"""Mean-field relaxation for Turyn-type sequences."""
from __future__ import annotations

import time

import numpy as np

from builders import build_turyn
from correlations import nonperiodic_batch_energy
from gpu import check_orthogonality, to_numpy, xp
from sieve import seed_turyn_batch
from .base import SearchStrategy


class IsingSearch(SearchStrategy):
    """Optimize a continuous weighted Turyn energy before sign rounding.

    Zweck: Sucht Turyn-Kandidaten im kontinuierlichen Spinraum.
    Mechanik: Aktualisiert Spins mit einem zero-padded FFT-Gradienten und ``tanh``.
    Grundlage: Die freie Energie ist die gewichtete nichtperiodische NPAF-Energie.
    Pipeline: Erzeugt einen Kandidaten für eine nachfolgende Verfeinerung.
    Grenzen: Die Rundung kann den kontinuierlichen Fortschritt wieder verlieren.
    """

    WEIGHTS = np.array((1, 1, 2, 2), dtype=np.float64)

    def __init__(self, order: int = SearchStrategy.ORDER, beta_end: float = 3.0,
                 evaluation_interval: int = 25, batch_size: int = 256, sieve: bool = True) -> None:
        n = (order // 4 + 1) // 3
        if order != 4 * (3 * n - 1) or n < 2:
            raise ValueError(
                "ising search requires an order with a TT(n) construction")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.ORDER, self.N, self.beta_end = order, n, beta_end
        self.evaluation_interval, self.batch_size, self.sieve = evaluation_interval, batch_size, sieve
        self.LENGTHS = np.array((n, n, n, n - 1), dtype=np.int64)

    @property
    def name(self) -> str:
        return "ising"

    def _energy_gradient(self, state, *, module=np):
        fft_size = 2 * self.N - 1
        spectrum = module.fft.fft(state, n=fft_size, axis=-1)
        correlations = module.fft.ifft(module.abs(spectrum) ** 2, axis=-1).real
        weights = module.asarray(self.WEIGHTS, dtype=state.dtype)
        total = module.sum(weights[None, :, None] * correlations, axis=1)
        total[:, 0] = 0.0
        coefficients = module.zeros_like(total)
        coefficients[:, 1:self.N] = total[:, 1:self.N]
        coefficients[:, self.N:] = total[:, 1:self.N][:, ::-1]
        gradient = 2 * weights[None, :, None] * module.fft.ifft(
            module.fft.fft(coefficients, axis=-1)[:, None, :] * spectrum,
            axis=-1).real[:, :, :self.N]
        return module.sum(total[:, 1:self.N] ** 2, axis=1), gradient.astype(module.float32)

    def _sign(self, state, *, module=np):
        signs = module.where(state >= 0, 1, -1).astype(module.int8)
        signs[:, 3, -1] = 0
        return signs

    def search(self, steps: int, seed: int):
        started = time.perf_counter()
        module = xp
        batch_size = self.batch_size if module.__name__ == "cupy" else 1
        rng = module.random.default_rng(seed)
        state = (seed_turyn_batch(self.N, batch_size, rng, module=module).astype(module.float32)
                 if self.sieve else rng.uniform(
                     -1, 1, size=(batch_size, 4, self.N)).astype(module.float32))
        state[:, 3, -1] = 0.0
        best = self._sign(state, module=module)
        best_energy = nonperiodic_batch_energy(
            best, lengths=self.LENGTHS, weights=self.WEIGHTS, module=module)
        for step in range(max(steps, 1)):
            _, gradient = self._energy_gradient(state, module=module)
            gradient /= module.maximum(module.max(module.abs(gradient),
                                       axis=(1, 2), keepdims=True), 1.0)
            beta = 0.5 + (self.beta_end - 0.5) * (step + 1) / max(steps, 1)
            state = (0.9 * state + 0.1 * module.tanh(beta *
                     (state - gradient))).astype(module.float32)
            state[:, 3, -1] = 0.0
            if (step + 1) % self.evaluation_interval == 0 or step + 1 == max(steps, 1):
                candidate = self._sign(state, module=module)
                energy = nonperiodic_batch_energy(
                    candidate, lengths=self.LENGTHS, weights=self.WEIGHTS, module=module)
                improved = energy < best_energy
                best = module.where(improved[:, None, None], candidate, best)
                best_energy = module.minimum(best_energy, energy)
        best = to_numpy(best[int(module.argmin(best_energy).item())])
        matrix = build_turyn(
            *(best[index, :self.LENGTHS[index]] for index in range(4)))
        return matrix, check_orthogonality(matrix), time.perf_counter() - started
