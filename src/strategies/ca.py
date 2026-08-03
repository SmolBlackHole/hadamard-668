"""Regel-annealter Zellularautomat für Vorzeichenmatrizen."""
from __future__ import annotations

import math
import time

import numpy as np
from scipy.ndimage import convolve

from gpu import check_orthogonality
from .base import SearchStrategy


class CASearch(SearchStrategy):
    """Optimiert eine lokale 3x3-Regel statt einzelner Matrixeinträge.

    Zweck: Erprobt lokale Zellularautomaten als alternative Kandidatendynamik.
    Mechanik: Mutiert Regelgewichte und akzeptiert deren per Convolution erzeugte Matrix nach Energie.
    Grundlage: Ein Schritt berechnet ``H' = sign(K ∗ H)`` mit einem 3×3-Kern ``K`` und zyklischem Rand.
    Pipeline: Kann jede Vorzeichenmatrix passender Form als Kopie weiterverfeinern.
    Grenzen: Lokale Regeln besitzen keine bekannte Korrektheits- oder Reparaturgarantie.
    """

    def __init__(self, order: int = SearchStrategy.ORDER, ca_steps: int = 1, rule_seed: int = 42) -> None:
        if order < 1 or ca_steps < 1:
            raise ValueError(
                "cellular search requires a positive order and ca_steps")
        self.ORDER = order
        self.ca_steps = ca_steps
        self.rule_seed = rule_seed

    @property
    def name(self) -> str:
        return "cellular"

    @staticmethod
    def _apply_ca(matrix: np.ndarray, rule: np.ndarray) -> np.ndarray:
        values = convolve(matrix.astype(np.float64), rule, mode="wrap")
        return np.where(values > 0, 1, -1).astype(np.int8)

    @staticmethod
    def _random_rule(rng: np.random.Generator) -> np.ndarray:
        return rng.choice((-1.0, 0.0, 1.0), size=(3, 3))

    @staticmethod
    def _mutate_rule(rule: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        candidate = rule.copy()
        row, column = rng.integers(0, 3), rng.integers(0, 3)
        choices = np.array((-1.0, 0.0, 1.0))
        candidate[row, column] = rng.choice(
            choices[choices != candidate[row, column]])
        return candidate

    def _evolve(self, matrix: np.ndarray, rule: np.ndarray) -> np.ndarray:
        candidate = matrix
        for _ in range(self.ca_steps):
            candidate = self._apply_ca(candidate, rule)
        return candidate

    def _optimize(self, matrix: np.ndarray, steps: int, seed: int) -> tuple[np.ndarray, dict[str, int], float]:
        started = time.perf_counter()
        rng = np.random.default_rng(seed + self.rule_seed)
        current = matrix.copy()
        metrics = best_metrics = check_orthogonality(current)
        best = current.copy()
        rule = self._random_rule(rng)
        energy_scale = max(metrics["energy"], 1)
        for step in range(steps):
            candidate_rule = self._mutate_rule(rule, rng)
            candidate = self._evolve(current, candidate_rule)
            candidate_metrics = check_orthogonality(candidate)
            delta = candidate_metrics["energy"] - metrics["energy"]
            temperature = 1.0 * 0.001 ** (step / max(steps - 1, 1))
            if delta <= 0 or rng.random() < math.exp(-delta / (temperature * energy_scale)):
                current, metrics, rule = candidate, candidate_metrics, candidate_rule
            if candidate_metrics["energy"] < best_metrics["energy"]:
                best, best_metrics = candidate.copy(), candidate_metrics
                if best_metrics["energy"] == 0:
                    break
        return best, best_metrics, time.perf_counter() - started

    def search(self, steps: int, seed: int) -> tuple[np.ndarray, dict[str, int], float]:
        rng = np.random.default_rng(seed)
        matrix = rng.choice(
            (-1, 1), size=(self.ORDER, self.ORDER)).astype(np.int8)
        return self._optimize(matrix, steps, seed)

    def refine(self, matrix: np.ndarray, steps: int, seed: int) -> tuple[np.ndarray, dict[str, int], float]:
        if matrix.shape != (self.ORDER, self.ORDER) or not np.all(np.isin(matrix, (-1, 1))):
            raise ValueError(
                f"cellular needs a {self.ORDER}x{self.ORDER} sign matrix")
        return self._optimize(matrix, steps, seed)
