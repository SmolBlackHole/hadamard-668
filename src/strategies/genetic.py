"""Genetische Suche über Populationen von vier zirkulanten Sequenzen."""
from __future__ import annotations

import time

import numpy as np

from constructions import build_goethals_seidel
from gpu import check_orthogonality, to_numpy, xp
from .base import SearchStrategy


class GeneticSearch(SearchStrategy):
    """Evolviert Populationen von vier zirkulanten Sequenzen.

    Zweck: Erprobt Auswahl, Crossover und Mutation ohne vollständige Matrizenpopulationen.
    Mechanik: Behält den Elitekandidaten, wählt Turniersieger und bewertet jede Generation mit einer Batch-FFT.
    Grundlage: Die Fitness ist die Summe der Quadrate komplementärer periodischer Autokorrelationen außerhalb des Null-Shifts.
    Pipeline: Kann nur eine Pipeline eröffnen, weil keine ``refine``-Methode existiert.
    Grenzen: Fitness und Operatoren garantieren weder Diversität noch einen Weg zu Energie null.
    """

    GPU_BATCH_THRESHOLD = 128

    def __init__(self, order: int = SearchStrategy.ORDER, population_size: int = 20, mutation_rate: float = 0.01) -> None:
        if order < 4 or order % 4:
            raise ValueError(
                "genetic search requires an order divisible by four")
        if population_size < 2 or not 0 <= mutation_rate <= 1:
            raise ValueError(
                "population_size must be at least two and mutation_rate must be in [0, 1]")
        self.ORDER = order
        self.K = order // 4
        self.population_size = population_size
        self.mutation_rate = mutation_rate

    @property
    def name(self) -> str:
        return "genetic"

    def _energies(self, population: np.ndarray) -> np.ndarray:
        """Evaluate a population together with one FFT batch."""
        module = (xp if xp.__name__ == "cupy" and
                  len(population) >= self.GPU_BATCH_THRESHOLD else np)
        dtype = np.float64 if module is np else module.float32
        values = module.asarray(population, dtype=dtype)
        spectrum = module.fft.fft(values, axis=2)
        correlations = module.rint(module.fft.ifft(
            module.abs(spectrum) ** 2, axis=2).real).sum(axis=1)
        correlations = correlations.astype(module.int64, copy=False)
        energies = module.sum(correlations[:, 1:] ** 2, axis=1)
        return np.asarray(to_numpy(energies), dtype=np.int64)

    def _energy(self, candidate: np.ndarray) -> int:
        return int(self._energies(candidate[None, ...])[0])

    def search(self, steps: int, seed: int) -> tuple[np.ndarray, dict[str, int], float]:
        started = time.perf_counter()
        rng = np.random.default_rng(seed)
        population = rng.choice(
            (-1, 1), size=(self.population_size, 4, self.K)).astype(np.int8)
        energies = self._energies(population)
        for _ in range(steps):
            elite = population[int(np.argmin(energies))].copy()
            next_population = [elite]

            def select() -> np.ndarray:
                indices = rng.choice(self.population_size,
                                     size=min(3, self.population_size),
                                     replace=False)
                return population[indices[int(np.argmin(energies[indices]))]]

            while len(next_population) < self.population_size:
                first, second = select(), select()
                split = int(rng.integers(1, self.K + 1))
                child = np.concatenate(
                    (first[:, :split], second[:, split:]), axis=1).copy()
                mutations = rng.random(child.shape) < self.mutation_rate
                child[mutations] *= -1
                next_population.append(child)
            population = np.array(next_population, dtype=np.int8)
            energies = self._energies(population)
            if energies.min() == 0:
                break
        best = population[int(np.argmin(energies))]
        matrix = build_goethals_seidel(*best)
        return matrix, check_orthogonality(matrix), time.perf_counter() - started
