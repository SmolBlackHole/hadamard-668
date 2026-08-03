"""Lokale Suche im zyklischen Vier-Sequenzen- und Difference-Set-Raum."""
from __future__ import annotations

import time

import numpy as np

from constructions import build_goethals_seidel, periodic_autocorrelation_energy
from gpu import check_orthogonality
from .base import SearchStrategy


class DiffsetSearch(SearchStrategy):
    """Sucht vier zyklische Difference-Set-Indikatoren mit Vorzeichenflips.

    Zweck: Erzeugt Goethals-Seidel-Kandidaten im allgemeinen Vier-Sequenzen-Raum.
    Mechanik: Startet mit quadratischen Resten und minimiert volle periodische Autokorrelationsenergie.
    Grundlage: Mitgliedschaft in vier Teilmengen von ``C_K`` wird als Vorzeichenfolge kodiert; komplementäre Differenzen liefern Goethals-Seidel-Kandidaten.
    Pipeline: Kann nur eine Pipeline eröffnen, weil keine ``refine``-Methode existiert.
    Grenzen: Die lokalen Flips erhalten weder Blockgrößen noch eine Difference-Set-Garantie.
    """

    def __init__(self, order: int = SearchStrategy.ORDER) -> None:
        if order < 4 or order % 4:
            raise ValueError(
                "diffset search requires an order divisible by four")
        self.ORDER = order
        self.K = order // 4

    @property
    def name(self) -> str:
        return "diffset"

    def _seed(self) -> list[np.ndarray]:
        residues = {value * value % self.K for value in range(1, self.K)}
        base = -np.ones(self.K, dtype=np.int8)
        base[list(residues)] = 1
        return [np.roll(base, shift).copy() for shift in range(4)]

    def search(self, steps: int, seed: int) -> tuple[np.ndarray, dict[str, int], float]:
        started = time.perf_counter()
        rng = np.random.default_rng(seed)
        current = self._seed()
        energy = best_energy = periodic_autocorrelation_energy(tuple(current))
        best = [sequence.copy() for sequence in current]
        for _ in range(steps):
            sequence, index = rng.integers(0, 4), rng.integers(0, self.K)
            current[sequence][index] *= -1
            candidate_energy = periodic_autocorrelation_energy(tuple(current))
            if candidate_energy <= energy:
                energy = candidate_energy
                if energy < best_energy:
                    best_energy = energy
                    best = [value.copy() for value in current]
                    if energy == 0:
                        break
            else:
                current[sequence][index] *= -1
        matrix = build_goethals_seidel(*best)
        return matrix, check_orthogonality(matrix), time.perf_counter() - started
