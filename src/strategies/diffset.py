"""Lokale Suche im zyklischen Vier-Sequenzen- und Difference-Set-Raum."""
from __future__ import annotations

import time

import numpy as np

from builders import build_goethals_seidel
from correlations import (
    apply_sequence_flip,
    autocorrelation_state,
    correlation_energy,
)
from gpu import check_orthogonality
from .base import SearchStrategy


class DiffsetSearch(SearchStrategy):
    """Sucht vier zyklische Difference-Set-Indikatoren mit Vorzeichenflips.

    Zweck: Erzeugt Goethals-Seidel-Kandidaten im allgemeinen Vier-Sequenzen-Raum.
    Mechanik: Startet mit quadratischen Resten und bewertet Einzelflips über exakte inkrementelle Korrelationsdeltas.
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

    @property
    def construction(self) -> str:
        return f"goethals_seidel_k_{self.K}"

    def _seed(self) -> list[np.ndarray]:
        residues = {value * value % self.K for value in range(1, self.K)}
        base = -np.ones(self.K, dtype=np.int8)
        base[list(residues)] = 1
        return [np.roll(base, shift).copy() for shift in range(4)]

    def search(self, steps: int, seed: int) -> tuple[np.ndarray, dict[str, int], float]:
        started = time.perf_counter()
        rng = np.random.default_rng(seed)
        current = np.stack(self._seed())
        correlations = autocorrelation_state(current)
        energy = best_energy = correlation_energy(correlations)
        best = current.copy()
        for _ in range(steps):
            sequence, index = rng.integers(0, 4), rng.integers(0, self.K)
            candidate_energy = apply_sequence_flip(
                current, correlations, int(sequence), int(index))
            if candidate_energy <= energy:
                energy = candidate_energy
                if energy < best_energy:
                    best_energy = energy
                    best = current.copy()
                    if energy == 0:
                        break
            else:
                apply_sequence_flip(
                    current, correlations, int(sequence), int(index))
        matrix = build_goethals_seidel(*best)
        return matrix, check_orthogonality(matrix), time.perf_counter() - started
