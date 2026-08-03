"""Lokale Vollmatrix-Suche mit gekoppelten Eintragsflips."""
from __future__ import annotations


import time

import numpy as np
from tqdm import tqdm

from gpu import check_orthogonality, to_numpy, xp
from .base import SearchStrategy


class DirectSearch(SearchStrategy):
    """Optimiert gekoppelte Einträge einer vollständigen Kandidatenmatrix.

    Zweck: Liefert eine strukturfreie lokale Verfeinerung für Vorzeichenmatrizen.
    Mechanik: Flipt gleichzeitig ``(i, j)`` und ``(j, i)`` und akzeptiert keine Energieverschlechterung.
    Grundlage: Die Energie ist ``Σᵢ<ⱼ (rᵢ · rⱼ)²`` für die Zeilen ``rᵢ``; sie verschwindet genau bei paarweiser Orthogonalität.
    Pipeline: Kann jede Matrix mit passender Form als Kopie weiterverfeinern.
    Grenzen: Jeder Schritt berechnet die vollständige Gram-Metrik; der Flip-Pfad ist nicht frei über alle Matrizen.
    """

    def __init__(self, order: int = SearchStrategy.ORDER) -> None:
        self.ORDER = order

    @property
    def name(self) -> str:
        return "direct"

    def _improve(self, matrix: np.ndarray, steps: int, seed: int) -> tuple[np.ndarray, dict[str, int], float]:
        started = time.perf_counter()
        rng = np.random.default_rng(seed)
        matrix = xp.asarray(matrix, dtype=xp.int8).copy()
        best = matrix.copy()
        energy = best_energy = check_orthogonality(matrix)["energy"]
        accepted = best_at = 0
        with tqdm(total=steps, desc=self.name, unit="steps", dynamic_ncols=True) as bar:
            for step in range(steps):
                row, column = rng.integers(
                    0, self.ORDER), rng.integers(0, self.ORDER)
                if row == column:
                    continue
                matrix[row, column] *= -1
                matrix[column, row] *= -1
                new_energy = check_orthogonality(matrix)["energy"]
                if new_energy <= energy:
                    energy, accepted = new_energy, accepted + 1
                    if new_energy < best_energy:
                        best_energy, best, best_at = new_energy, matrix.copy(), step
                        if new_energy == 0:
                            break
                else:
                    matrix[row, column] *= -1
                    matrix[column, row] *= -1
                if step % 50 == 0:
                    bar.set_postfix(e=energy, best=best_energy, acc=accepted)
                    bar.update(50)
        elapsed = time.perf_counter() - started
        print(
            f"  seed={seed} best_energy={best_energy} found@step={best_at} accepted={accepted} {elapsed:.1f}s")
        return to_numpy(best), check_orthogonality(best), elapsed

    def search(self, steps: int, seed: int) -> tuple[np.ndarray, dict[str, int], float]:
        rng = np.random.default_rng(seed)
        matrix = rng.choice(
            [-1, 1], size=(self.ORDER, self.ORDER)).astype(np.int8)
        return self._improve(matrix, steps, seed)

    def refine(self, matrix: np.ndarray, steps: int, seed: int) -> tuple[np.ndarray, dict[str, int], float]:
        self._validate_matrix(matrix)
        return self._improve(matrix.copy(), steps, seed)

    def _validate_matrix(self, matrix: np.ndarray) -> None:
        if matrix.shape != (self.ORDER, self.ORDER):
            raise ValueError(
                f"{self.name} needs a {self.ORDER}x{self.ORDER} matrix")
