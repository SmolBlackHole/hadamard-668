"""Genetic search over Turyn-type sequences."""
from __future__ import annotations

import time

import numpy as np

from builders import build_turyn
from correlations import nonperiodic_batch_energy
from gpu import check_orthogonality, to_numpy, xp
from .base import SearchStrategy


class GeneticSearch(SearchStrategy):
    """Evolve a population of TT(n) sign sequences.

    Zweck: Erzeugt nichtlokale Turyn-Kandidaten durch Crossover und Mutation.
    Mechanik: Behält Eliten und bewertet jede Population als FFT-Batch.
    Grundlage: Fitness ist die gewichtete nichtperiodische Turyn-Energie.
    Pipeline: Erzeugt einen Kandidaten für eine nachfolgende Verfeinerung.
    Grenzen: Rekombination garantiert weder Diversität noch Energie null.
    """

    GPU_BATCH_THRESHOLD = 128
    WEIGHTS = np.array((1, 1, 2, 2), dtype=np.int64)

    def __init__(self, order: int = SearchStrategy.ORDER, population_size: int = 20, mutation_rate: float = 0.01) -> None:
        n = (order // 4 + 1) // 3
        if order != 4 * (3 * n - 1) or n < 2:
            raise ValueError("genetic search requires an order with a TT(n) construction")
        if population_size < 2 or not 0 <= mutation_rate <= 1:
            raise ValueError("population_size must be at least two and mutation_rate must be in [0, 1]")
        self.ORDER, self.N = order, n
        self.LENGTHS = np.array((n, n, n, n - 1), dtype=np.int64)
        self.population_size, self.mutation_rate = population_size, mutation_rate

    @property
    def name(self) -> str:
        return "genetic"

    def _energies(self, population: np.ndarray) -> np.ndarray:
        module = xp if xp.__name__ == "cupy" and len(population) >= self.GPU_BATCH_THRESHOLD else np
        values = module.asarray(population, dtype=module.float32 if module is xp else np.float64)
        energies = nonperiodic_batch_energy(
            values, lengths=self.LENGTHS, weights=self.WEIGHTS, module=module)
        return np.asarray(to_numpy(energies), dtype=np.int64)

    def _build(self, sequences: np.ndarray):
        matrix = build_turyn(*(sequences[index, :self.LENGTHS[index]] for index in range(4)))
        return matrix, check_orthogonality(matrix)

    def search(self, steps: int, seed: int):
        started = time.perf_counter()
        rng = np.random.default_rng(seed)
        population = rng.choice((-1, 1), size=(self.population_size, 4, self.N)).astype(np.int8)
        population[:, 3, -1] = 0
        energies = self._energies(population)
        for _ in range(steps):
            elite = population[int(np.argmin(energies))].copy()
            next_population = [elite]
            while len(next_population) < self.population_size:
                choices = rng.choice(self.population_size, size=6, replace=True)
                first = population[choices[:3][np.argmin(energies[choices[:3]])]]
                second = population[choices[3:][np.argmin(energies[choices[3:]])]]
                split = int(rng.integers(1, self.N))
                child = np.concatenate((first[:, :split], second[:, split:]), axis=1).copy()
                child[rng.random(child.shape) < self.mutation_rate] *= -1
                child[3, -1] = 0
                next_population.append(child)
            population = np.asarray(next_population, dtype=np.int8)
            energies = self._energies(population)
        matrix, metrics = self._build(population[int(np.argmin(energies))])
        return matrix, metrics, time.perf_counter() - started
