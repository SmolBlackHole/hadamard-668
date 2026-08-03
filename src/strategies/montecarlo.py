"""CUDA-Batch-Suche über viele zyklische Vier-Sequenzen-Kandidaten."""
from __future__ import annotations

import time

import numpy as np

from constructions import build_goethals_seidel
from gpu import check_orthogonality, xp
from .base import SearchStrategy


class MonteCarloSearch(SearchStrategy):
    """Mutiert viele Vier-Sequenzen-Kandidaten parallel auf CUDA.

    Zweck: Nutzt die GPU für einen Batch zirkulanter Goethals-Seidel-Kandidaten.
    Mechanik: Bewertet periodische Korrelation per FFT und akzeptiert pro Batch nur bessere Flips.
    Grundlage: Die inverse FFT von ``|FFT(s)|²`` liefert die periodische Autokorrelation jeder Sequenz (Wiener-Khinchin).
    Pipeline: Kann nur eine Pipeline eröffnen, weil keine ``refine``-Methode existiert.
    Grenzen: Benötigt einen aktiven CuPy-Backend; ohne CUDA wird kein CPU-Fallback angeboten.
    """

    def __init__(self, order: int = SearchStrategy.ORDER, batch_size: int = 256) -> None:
        if order < 4 or order % 4:
            raise ValueError(
                "montecarlo search requires an order divisible by four")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self.ORDER = order
        self.K = order // 4
        self.batch_size = batch_size

    @property
    def name(self) -> str:
        return "montecarlo"

    def _energy(self, batch):
        spectrum = xp.fft.fft(batch, axis=2)
        correlations = xp.rint(xp.fft.ifft(
            xp.abs(spectrum) ** 2, axis=2).real).sum(axis=1)
        return xp.sum(correlations[:, 1:] ** 2, axis=1)

    def search(self, steps: int, seed: int) -> tuple[np.ndarray, dict[str, int], float]:
        if xp.__name__ != "cupy":
            raise RuntimeError(
                "montecarlo search requires an active CuPy backend")
        started = time.perf_counter()
        rng = xp.random.default_rng(seed)
        batch = rng.integers(0, 2, size=(
            self.batch_size, 4, self.K), dtype=xp.int8) * 2 - 1
        energies = self._energy(batch)
        indices = xp.arange(self.batch_size)
        for _ in range(steps):
            candidate = batch.copy()
            sequence = rng.integers(0, 4, size=self.batch_size)
            column = rng.integers(0, self.K, size=self.batch_size)
            candidate[indices, sequence, column] *= -1
            candidate_energies = self._energy(candidate)
            accepted = candidate_energies <= energies
            batch[accepted] = candidate[accepted]
            energies = xp.where(accepted, candidate_energies, energies)
            if int(energies.min().get()) == 0:
                break
        best = xp.asnumpy(batch[int(energies.argmin().get())])
        matrix = build_goethals_seidel(*best)
        return matrix, check_orthogonality(matrix), time.perf_counter() - started
