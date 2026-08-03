"""Mean-Field-Relaxation für vier komplementäre zyklische Folgen."""
from __future__ import annotations

import time

import numpy as np

from builders import build_goethals_seidel
from correlations import periodic_autocorrelation_energy
from gpu import check_orthogonality
from .base import SearchStrategy


class IsingSearch(SearchStrategy):
    """Optimiert eine kontinuierliche Autokorrelationsenergie und rundet auf Vorzeichen.

    Zweck: Modelliert die vier Folgen als kontinuierliche Spins in ``[-1, 1]``.
    Mechanik: Berechnet den exakten FFT-Gradienten und wendet ein temperaturgesteuertes Mean-Field-``tanh``-Update an.
    Grundlage: Die freie Energie ist die Summe der Quadrate aller summierten periodischen Nichtnull-Autokorrelationen.
    Pipeline: Kann nur eine Pipeline eröffnen, weil keine ``refine``-Methode existiert.
    Grenzen: Die kontinuierliche Relaxation und ihre diskrete Rundung garantieren keine Hadamard-Lösung.
    """

    def __init__(
        self,
        order: int = SearchStrategy.ORDER,
        beta_end: float = 3.0,
        evaluation_interval: int = 25,
    ) -> None:
        if order < 4 or order % 4:
            raise ValueError("ising search requires an order divisible by four")
        if beta_end <= 0 or evaluation_interval < 1:
            raise ValueError("beta_end and evaluation_interval must be positive")
        self.ORDER = order
        self.K = order // 4
        self.beta_end = beta_end
        self.evaluation_interval = evaluation_interval

    @property
    def name(self) -> str:
        return "ising"

    @staticmethod
    def _energy_gradient(state: np.ndarray) -> tuple[float, np.ndarray]:
        """Return continuous periodic-correlation energy and its FFT gradient."""
        spectrum = np.fft.fft(state, axis=1)
        correlations = np.fft.ifft(
            np.abs(spectrum) ** 2, axis=1).real.sum(axis=0)
        correlations[0] = 0.0
        energy = float(np.dot(correlations, correlations))
        gradient = 4.0 * np.fft.ifft(
            np.fft.fft(correlations)[None, :] * spectrum,
            axis=1,
        ).real
        return energy, gradient.astype(np.float32)

    @staticmethod
    def _sign(state: np.ndarray) -> np.ndarray:
        return np.where(state >= 0, 1, -1).astype(np.int8)

    def search(self, steps: int, seed: int) -> tuple[np.ndarray, dict[str, int], float]:
        started = time.perf_counter()
        rng = np.random.default_rng(seed)
        state = rng.uniform(-1, 1, size=(4, self.K)).astype(np.float32)
        best = self._sign(state)
        best_energy = periodic_autocorrelation_energy(tuple(best))
        iterations = max(1, steps)

        for step in range(iterations):
            beta = 0.5 + (self.beta_end - 0.5) * (step + 1) / iterations
            _, gradient = self._energy_gradient(state)
            gradient /= max(float(np.max(np.abs(gradient))), 1.0)
            proposal = np.tanh(beta * (state - gradient)).astype(np.float32)
            state = (0.9 * state + 0.1 * proposal).astype(np.float32)
            if (step + 1) % self.evaluation_interval and step + 1 < iterations:
                continue
            candidate = self._sign(state)
            energy = periodic_autocorrelation_energy(tuple(candidate))
            if energy < best_energy:
                best, best_energy = candidate, energy
                if energy == 0:
                    break

        matrix = build_goethals_seidel(*best)
        metrics = check_orthogonality(matrix)
        return matrix, metrics, time.perf_counter() - started
