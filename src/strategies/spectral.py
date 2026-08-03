"""Douglas-Rachford projection for Turyn-type sequences."""
from __future__ import annotations

import time

import numpy as np
from tqdm import tqdm

from builders import build_turyn
from correlations import nonperiodic_batch_energy
from fourier import project_weighted_nonperiodic_power
from gpu import check_orthogonality
from .base import SearchStrategy


class SpectralSearch(SearchStrategy):
    """Alternate weighted non-periodic power and sign projections.

    Zweck: Überträgt Douglas-Rachford auf den Turyn-Raum.
    Mechanik: Projiziert zero-padded Folgen auf die NPAF-Leistungsschale und auf Vorzeichen.
    Grundlage: Konstante gewichtete Leistung entspricht verschwindenden Nichtnull-NPAFs.
    Pipeline: Erzeugt einen Kandidaten für eine nachfolgende Verfeinerung.
    Grenzen: Die diskrete Vorzeichenprojektion bleibt heuristisch.
    """

    WEIGHTS = np.array((1, 1, 2, 2), dtype=np.float64)

    def __init__(self, inner_steps: int = 50, *, ORDER: int = 668) -> None:
        n = (ORDER // 4 + 1) // 3
        if ORDER != 4 * (3 * n - 1) or n < 2:
            raise ValueError("spectral search requires an order with a TT(n) construction")
        self.ORDER, self.N, self.inner_steps = ORDER, n, inner_steps
        self.LENGTHS = np.array((n, n, n, n - 1), dtype=np.int64)

    @property
    def name(self) -> str:
        return "spectral"

    def _project_fourier(self, state: np.ndarray) -> np.ndarray:
        return project_weighted_nonperiodic_power(
            state, lengths=self.LENGTHS, weights=self.WEIGHTS)

    def _project_sign(self, state: np.ndarray) -> np.ndarray:
        signs = np.where(state >= 0, 1.0, -1.0).astype(np.float32)
        signs[3, -1] = 0.0
        return signs

    def _step(self, state: np.ndarray) -> np.ndarray:
        projected = self._project_fourier(state)
        reflected = 2 * projected - state
        return (0.5 * (state + 2 * self._project_sign(reflected) - reflected)).astype(np.float32)

    def search(self, steps: int, seed: int):
        started = time.perf_counter()
        rng = np.random.default_rng(seed)
        state = rng.normal(size=(4, self.N)).astype(np.float32)
        state[3, -1] = 0.0
        best = self._project_sign(state)
        best_energy = int(nonperiodic_batch_energy(best[None, ...], lengths=self.LENGTHS, weights=self.WEIGHTS)[0])
        with tqdm(total=steps, desc=self.name, unit="steps") as progress:
            for _ in range(steps):
                for _ in range(self.inner_steps):
                    state = self._step(state)
                candidate = self._project_sign(self._project_fourier(state))
                energy = int(nonperiodic_batch_energy(candidate[None, ...], lengths=self.LENGTHS, weights=self.WEIGHTS)[0])
                if energy < best_energy:
                    best, best_energy = candidate, energy
                progress.update(1)
        matrix = build_turyn(*(best[index, :self.LENGTHS[index]] for index in range(4)))
        return matrix, check_orthogonality(matrix), time.perf_counter() - started
