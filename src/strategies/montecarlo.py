"""CUDA-batch search over Turyn-type sequences."""
from __future__ import annotations

import time

import numpy as np

from builders import build_turyn
from correlations import nonperiodic_batch_energy
from gpu import check_orthogonality, xp
from .base import SearchStrategy


class MonteCarloSearch(SearchStrategy):
    """Mutate many TT(n) candidates in parallel on CUDA.

    Zweck: Führt unabhängige Turyn-Restarts als GPU-Batch aus.
    Mechanik: Testet einen Flip je Kandidat über batched zero-padded FFTs.
    Grundlage: Fitness ist die gewichtete nichtperiodische Turyn-Energie.
    Pipeline: Erzeugt einen Kandidaten für eine nachfolgende Verfeinerung.
    Grenzen: Benötigt CuPy; ein Flip-Schritt berechnet den kleinen FFT-Batch neu.
    """

    WEIGHTS = np.array((1, 1, 2, 2), dtype=np.int64)

    def __init__(self, order: int = SearchStrategy.ORDER, batch_size: int = 256) -> None:
        n = (order // 4 + 1) // 3
        if order != 4 * (3 * n - 1) or n < 2:
            raise ValueError("montecarlo search requires an order with a TT(n) construction")
        self.ORDER, self.N, self.batch_size = order, n, batch_size
        self.LENGTHS = np.array((n, n, n, n - 1), dtype=np.int64)

    @property
    def name(self) -> str:
        return "montecarlo"

    def _energies(self, batch):
        return nonperiodic_batch_energy(
            batch, lengths=self.LENGTHS, weights=self.WEIGHTS, module=xp)

    def search(self, steps: int, seed: int):
        if xp.__name__ != "cupy":
            raise RuntimeError("montecarlo search requires an active CuPy backend")
        started = time.perf_counter()
        rng = xp.random.default_rng(seed)
        batch = rng.integers(0, 2, size=(self.batch_size, 4, self.N), dtype=xp.int8) * 2 - 1
        batch[:, 3, -1] = 0
        energies = self._energies(batch)
        rows = xp.arange(self.batch_size)
        for _ in range(steps):
            candidate = batch.copy()
            sequence = rng.integers(0, 4, size=self.batch_size)
            column = rng.integers(0, self.N, size=self.batch_size)
            invalid = (sequence == 3) & (column == self.N - 1)
            column = xp.where(invalid, self.N - 2, column)
            candidate[rows, sequence, column] *= -1
            candidate_energies = self._energies(candidate)
            accepted = candidate_energies <= energies
            batch[accepted] = candidate[accepted]
            energies = xp.where(accepted, candidate_energies, energies)
        best = xp.asnumpy(batch[int(energies.argmin().get())])
        matrix = build_turyn(*(best[index, :self.LENGTHS[index]] for index in range(4)))
        return matrix, check_orthogonality(matrix), time.perf_counter() - started
