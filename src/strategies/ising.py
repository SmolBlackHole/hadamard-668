"""Mean-field relaxation for Turyn-type sequences."""
from __future__ import annotations

import time

import numpy as np

from builders import build_turyn
from correlations import nonperiodic_batch_energy
from gpu import check_orthogonality
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

    def __init__(self, order: int = SearchStrategy.ORDER, beta_end: float = 3.0, evaluation_interval: int = 25) -> None:
        n = (order // 4 + 1) // 3
        if order != 4 * (3 * n - 1) or n < 2:
            raise ValueError("ising search requires an order with a TT(n) construction")
        self.ORDER, self.N, self.beta_end, self.evaluation_interval = order, n, beta_end, evaluation_interval
        self.LENGTHS = np.array((n, n, n, n - 1), dtype=np.int64)

    @property
    def name(self) -> str:
        return "ising"

    def _energy_gradient(self, state: np.ndarray) -> tuple[float, np.ndarray]:
        fft_size = 2 * self.N - 1
        spectrum = np.fft.fft(state, n=fft_size, axis=1)
        correlations = np.fft.ifft(np.abs(spectrum) ** 2, axis=1).real
        total = np.tensordot(self.WEIGHTS, correlations, axes=1)
        total[0] = 0.0
        coefficients = total.copy()
        coefficients[self.N:] = total[1:self.N][::-1]
        gradient = 2 * self.WEIGHTS[:, None] * np.fft.ifft(
            np.fft.fft(coefficients)[None, :] * spectrum, axis=1).real[:, :self.N]
        return float(np.dot(total[1:self.N], total[1:self.N])), gradient.astype(np.float32)

    def _sign(self, state: np.ndarray) -> np.ndarray:
        signs = np.where(state >= 0, 1, -1).astype(np.int8)
        signs[3, -1] = 0
        return signs

    def search(self, steps: int, seed: int):
        started = time.perf_counter()
        rng = np.random.default_rng(seed)
        state = rng.uniform(-1, 1, size=(4, self.N)).astype(np.float32)
        state[3, -1] = 0.0
        best = self._sign(state)
        best_energy = int(nonperiodic_batch_energy(best[None, ...], lengths=self.LENGTHS, weights=self.WEIGHTS)[0])
        for step in range(max(steps, 1)):
            _, gradient = self._energy_gradient(state)
            gradient /= max(float(np.max(np.abs(gradient))), 1.0)
            beta = 0.5 + (self.beta_end - 0.5) * (step + 1) / max(steps, 1)
            state = (0.9 * state + 0.1 * np.tanh(beta * (state - gradient))).astype(np.float32)
            state[3, -1] = 0.0
            if (step + 1) % self.evaluation_interval == 0 or step + 1 == max(steps, 1):
                candidate = self._sign(state)
                energy = int(nonperiodic_batch_energy(candidate[None, ...], lengths=self.LENGTHS, weights=self.WEIGHTS)[0])
                if energy < best_energy:
                    best, best_energy = candidate, energy
        matrix = build_turyn(*(best[index, :self.LENGTHS[index]] for index in range(4)))
        return matrix, check_orthogonality(matrix), time.perf_counter() - started
