"""Genetic search over Turyn-type sequences."""
from __future__ import annotations

import time

import numpy as np

from builders import build_turyn
from correlations import nonperiodic_batch_energy
from gpu import check_orthogonality, to_numpy, xp
from sieve import seed_turyn_batch
from .base import SearchStrategy


class GeneticSearch(SearchStrategy):
    """Evolve a population of TT(n) sign sequences.

    Zweck: Erzeugt nichtlokale Turyn-Kandidaten durch Crossover und Mutation.
    Mechanik: Behält Eliten und bewertet jede Population als FFT-Batch.
    Grundlage: Fitness ist die gewichtete nichtperiodische Turyn-Energie.
    Pipeline: Erzeugt einen Kandidaten für eine nachfolgende Verfeinerung.
    Grenzen: Rekombination garantiert weder Diversität noch Energie null.
    """

    WEIGHTS = np.array((1, 1, 2, 2), dtype=np.int64)

    def __init__(self, order: int = SearchStrategy.ORDER, population_size: int = 20,
                 mutation_rate: float = 0.01, sieve: bool = True) -> None:
        n = (order // 4 + 1) // 3
        if order != 4 * (3 * n - 1) or n < 2:
            raise ValueError(
                "genetic search requires an order with a TT(n) construction")
        if population_size < 2 or not 0 <= mutation_rate <= 1:
            raise ValueError(
                "population_size must be at least two and mutation_rate must be in [0, 1]")
        self.ORDER, self.N = order, n
        self.LENGTHS = np.array((n, n, n, n - 1), dtype=np.int64)
        self.population_size, self.mutation_rate, self.sieve = population_size, mutation_rate, sieve

    @property
    def name(self) -> str:
        return "genetic"

    @property
    def construction(self) -> str:
        return f"turyn_tt_{self.N}"

    def _energies(self, population, *, module=np):
        return nonperiodic_batch_energy(
            population, lengths=self.LENGTHS, weights=self.WEIGHTS, module=module)

    def _build(self, sequences: np.ndarray):
        matrix = build_turyn(
            *(sequences[index, :self.LENGTHS[index]] for index in range(4)))
        return matrix, check_orthogonality(matrix)

    def search(self, steps: int, seed: int):
        started = time.perf_counter()
        module = xp
        rng = module.random.default_rng(seed)
        population = (seed_turyn_batch(self.N, self.population_size, rng, module=module)
                      if self.sieve else rng.integers(
                          0, 2, size=(self.population_size, 4, self.N), dtype=module.int8) * 2 - 1)
        population[:, 3, -1] = 0
        energies = self._energies(population, module=module)
        for _ in range(steps):
            elite_index = module.argmin(energies)
            tournament = rng.integers(0, self.population_size,
                                      size=(self.population_size - 1, 2, 3))
            winners = module.take_along_axis(
                tournament, module.argmin(energies[tournament], axis=2)[..., None], axis=2)[..., 0]
            parents = population[winners]
            split = rng.integers(1, self.N, size=self.population_size - 1)
            positions = module.arange(self.N)[None, None, :]
            children = module.where(positions < split[:, None, None],
                                    parents[:, 0], parents[:, 1]).copy()
            children[rng.random(children.shape) < self.mutation_rate] *= -1
            children[:, 3, -1] = 0
            population = module.concatenate(
                (population[elite_index][None, ...], children), axis=0)
            energies = self._energies(population, module=module)
        best = to_numpy(population[int(module.argmin(energies).item())])
        matrix, metrics = self._build(best)
        return matrix, metrics, time.perf_counter() - started
