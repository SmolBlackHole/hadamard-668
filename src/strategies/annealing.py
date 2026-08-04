"""Simulated annealing in the Turyn-type sequence space."""
from __future__ import annotations

import math
import time

import numpy as np

from builders import build_turyn
from correlations import (
    apply_nonperiodic_flip,
    nonperiodic_autocorrelation_state,
    nonperiodic_correlation_energy,
)
from gpu import check_orthogonality
from sieve import seed_turyn_batch
from .base import SearchStrategy


class TurynAnnealingSearch(SearchStrategy):
    """Escape Turyn local minima with temperature-controlled bit flips.

    Zweck: Durchbricht lokale Minima der Turyn-Greedy-Suche.
    Mechanik: Akzeptiert schlechtere Flips mit exponentiell sinkender Wahrscheinlichkeit.
    Grundlage: Bewertet die gewichtete nichtperiodische Turyn-Energie exakt.
    Pipeline: Erzeugt einen Kandidaten für eine nachfolgende Verfeinerung.
    Grenzen: Die Temperaturschedule ist eine Heuristik ohne Konvergenzgarantie.
    """

    N = 56
    WEIGHTS = np.array((1, 1, 2, 2), dtype=np.int64)

    def __init__(self, t_start: float = 5.0, t_end: float = 0.001, *, n: int = N, sieve: bool = True) -> None:
        if n < 2:
            raise ValueError("Turyn type requires n >= 2")
        self.t_start = t_start
        self.t_end = t_end
        self.N = n
        self.LENGTHS = np.array((n, n, n, n - 1), dtype=np.int64)
        self.ORDER = 4 * (3 * n - 1)
        self.sieve = sieve

    @property
    def name(self) -> str:
        return "annealing"

    @property
    def construction(self) -> str:
        return f"turyn_tt_{self.N}"

    def _seed(self, rng: np.random.Generator) -> np.ndarray:
        if self.sieve:
            return seed_turyn_batch(self.N, 1, rng)[0]
        sequences = np.zeros((4, self.N), dtype=np.int8)
        for index, length in enumerate(self.LENGTHS):
            sequences[index, :length] = rng.choice((-1, 1), size=length)
        return sequences

    def _build(self, sequences: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
        matrix = build_turyn(*(sequences[index, :self.LENGTHS[index]]
                               for index in range(4)))
        return matrix, check_orthogonality(matrix)

    def search(self, steps: int, seed: int) -> tuple[np.ndarray, dict[str, int], float]:
        started = time.perf_counter()
        rng = np.random.default_rng(seed)
        sequences = self._seed(rng)
        correlations = nonperiodic_autocorrelation_state(
            sequences, lengths=self.LENGTHS, weights=self.WEIGHTS)
        energy = best_energy = nonperiodic_correlation_energy(correlations)
        best_sequences = sequences.copy()
        energy_scale = max(energy, 1)
        accepted = uphill = best_at = 0
        for step in range(steps):
            temperature = self.t_start * \
                (self.t_end / self.t_start) ** (step / max(steps, 1))
            sequence_index = int(rng.integers(0, 4))
            value_index = int(rng.integers(0, self.LENGTHS[sequence_index]))
            new_energy = apply_nonperiodic_flip(
                sequences, correlations, sequence_index, value_index,
                lengths=self.LENGTHS, weight=int(self.WEIGHTS[sequence_index]))
            delta = new_energy - energy
            if delta <= 0 or rng.random() < math.exp(-delta / max(temperature * energy_scale, 1e-12)):
                if delta > 0:
                    uphill += 1
                energy, accepted = new_energy, accepted + 1
                if new_energy < best_energy:
                    best_energy, best_at = new_energy, step
                    best_sequences = sequences.copy()
                    if not best_energy:
                        break
            else:
                apply_nonperiodic_flip(
                    sequences, correlations, sequence_index, value_index,
                    lengths=self.LENGTHS, weight=int(self.WEIGHTS[sequence_index]))
        matrix, metrics = self._build(best_sequences)
        elapsed = time.perf_counter() - started
        print(f"  seed={seed} best={best_energy} found@step={best_at} "
              f"accept={accepted} uphill={uphill} {elapsed:.1f}s")
        return matrix, metrics, elapsed
