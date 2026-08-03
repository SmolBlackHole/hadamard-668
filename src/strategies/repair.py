"""Greedy-Reparatur anhand des stärksten Gram-Matrix-Verstoßes."""
from __future__ import annotations


import time

import numpy as np
from tqdm import tqdm

from gpu import check_orthogonality
from .base import SearchStrategy


class RepairSearch(SearchStrategy):
    """Sucht strikt bessere Flips für das am stärksten korrelierte Zeilenpaar.

    Zweck: Verbessert vorhandene Kandidaten mit einer gezielten lokalen Reparatur.
    Mechanik: Testet bis zu drei gleiche Spalten des Verletzerpaars und akzeptiert nur Verbesserungen.
    Grundlage: Der größte Betrag eines Off-Diagonal-Eintrags von ``H Hᵀ`` bestimmt das zu reparierende Zeilenpaar.
    Pipeline: Kann jede Matrix mit passender Form als Kopie weiterverfeinern.
    Grenzen: Greedy-Descent kann auf Plateaus stoppen und besitzt keine Erfolgsschwelle oder Neustarts.
    """

    def __init__(self, order: int = SearchStrategy.ORDER) -> None:
        self.ORDER = order

    @property
    def name(self) -> str:
        return "repair"

    @staticmethod
    def _gram_energy(matrix: np.ndarray) -> tuple[int, np.ndarray]:
        values = matrix.astype(np.int64)
        gram = values @ values.T
        np.fill_diagonal(gram, 0)
        return int(np.sum(gram * gram) // 2), gram

    @staticmethod
    def _most_violated_pair(gram: np.ndarray) -> tuple[int, int, int]:
        index = int(np.argmax(np.abs(gram)))
        row, other = divmod(index, gram.shape[0])
        return row, other, int(gram[row, other])

    def _improve(self, matrix: np.ndarray, steps: int, seed: int) -> tuple[np.ndarray, dict[str, int], float]:
        started = time.perf_counter()
        rng = np.random.default_rng(seed)
        energy, gram = self._gram_energy(matrix)
        best, best_energy = matrix.copy(), energy
        accepted = best_at = 0
        with tqdm(total=steps, desc=self.name, unit="steps", dynamic_ncols=True) as bar:
            for step in range(steps):
                row, other, dot_product = self._most_violated_pair(gram)
                if dot_product == 0:
                    break
                columns = rng.permutation(np.flatnonzero(
                    matrix[row] == matrix[other]))[:3]
                if not len(columns):
                    columns = np.array([rng.integers(0, self.ORDER)])
                for column in columns:
                    candidate = matrix.copy()
                    candidate[row, column] *= -1
                    new_energy, new_gram = self._gram_energy(candidate)
                    if new_energy < energy:
                        matrix, energy, gram = candidate, new_energy, new_gram
                        accepted += 1
                        if energy < best_energy:
                            best, best_energy, best_at = matrix.copy(), energy, step
                        break
                if step % 50 == 0:
                    bar.set_postfix(e=energy, best=best_energy, acc=accepted)
                    bar.update(50)
                if best_energy == 0:
                    break
        elapsed = time.perf_counter() - started
        print(
            f"  seed={seed} best_energy={best_energy} found@step={best_at} accepted={accepted} {elapsed:.1f}s")
        return best, check_orthogonality(best), elapsed

    def search(self, steps: int, seed: int) -> tuple[np.ndarray, dict[str, int], float]:
        rng = np.random.default_rng(seed)
        matrix = rng.choice(
            [-1, 1], size=(self.ORDER, self.ORDER)).astype(np.int8)
        return self._improve(matrix, steps, seed)

    def refine(self, matrix: np.ndarray, steps: int, seed: int) -> tuple[np.ndarray, dict[str, int], float]:
        if matrix.shape != (self.ORDER, self.ORDER):
            raise ValueError(
                f"{self.name} needs a {self.ORDER}x{self.ORDER} matrix")
        return self._improve(matrix.copy(), steps, seed)
