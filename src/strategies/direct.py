"""Lokale Vollmatrix-Suche mit gekoppelten Eintragsflips."""
from __future__ import annotations


import time

import numpy as np
from tqdm import tqdm

from gpu import apply_entry_flip, gram_matrix, metrics_from_gram
from .base import SearchStrategy


class DirectSearch(SearchStrategy):
    """Optimiert gekoppelte Einträge einer vollständigen Kandidatenmatrix.

    Zweck: Liefert eine strukturfreie lokale Verfeinerung für Vorzeichenmatrizen.
    Mechanik: Flipt gleichzeitig ``(i, j)`` und ``(j, i)``, aktualisiert die Gram-Matrix inkrementell und akzeptiert keine Energieverschlechterung.
    Grundlage: Die Energie ist ``Σᵢ<ⱼ (rᵢ · rⱼ)²`` für die Zeilen ``rᵢ``; sie verschwindet genau bei paarweiser Orthogonalität.
    Pipeline: Kann jede Matrix mit passender Form als Kopie weiterverfeinern.
    Grenzen: ``search`` bleibt im symmetrischen Unterraum; ``refine`` erhält die paarweisen Symmetrierelationen des Eingangs und kann auf Plateaus stoppen.
    """

    compute_backend = "numpy-incremental"

    def __init__(self, order: int = SearchStrategy.ORDER) -> None:
        self.ORDER = order

    @property
    def name(self) -> str:
        return "direct"

    def _improve(self, matrix: np.ndarray, steps: int, seed: int) -> tuple[np.ndarray, dict[str, int], float]:
        started = time.perf_counter()
        rng = np.random.default_rng(seed)
        matrix = np.asarray(matrix, dtype=np.int8).copy()
        gram = gram_matrix(matrix, backend=np)
        energy = best_energy = metrics_from_gram(gram)["energy"]
        accepted = best_at = 0
        with tqdm(total=steps, desc=self.name, unit="steps", dynamic_ncols=True) as bar:
            for step in range(steps):
                bar.update(1)
                row, column = rng.integers(
                    0, self.ORDER), rng.integers(0, self.ORDER)
                if row == column:
                    continue
                first_delta = apply_entry_flip(matrix, gram, row, column)
                second_delta = apply_entry_flip(matrix, gram, column, row)
                delta = first_delta + second_delta
                if delta <= 0:
                    energy += delta
                    accepted += 1
                    if energy < best_energy:
                        best_energy, best_at = energy, step
                        if energy == 0:
                            break
                else:
                    apply_entry_flip(matrix, gram, column, row)
                    apply_entry_flip(matrix, gram, row, column)
                if step % 50 == 0:
                    bar.set_postfix(e=energy, best=best_energy, acc=accepted)
        elapsed = time.perf_counter() - started
        print(
            f"  seed={seed} best_energy={best_energy} found@step={best_at} accepted={accepted} {elapsed:.1f}s")
        return matrix, metrics_from_gram(gram), elapsed

    def search(self, steps: int, seed: int) -> tuple[np.ndarray, dict[str, int], float]:
        rng = np.random.default_rng(seed)
        random = rng.choice(
            [-1, 1], size=(self.ORDER, self.ORDER)).astype(np.int8)
        matrix = np.triu(random)
        matrix += np.triu(random, k=1).T
        return self._improve(matrix, steps, seed)

    def refine(self, matrix: np.ndarray, steps: int, seed: int) -> tuple[np.ndarray, dict[str, int], float]:
        self._validate_matrix(matrix)
        return self._improve(matrix.copy(), steps, seed)

    def _validate_matrix(self, matrix: np.ndarray) -> None:
        if matrix.shape != (self.ORDER, self.ORDER):
            raise ValueError(
                f"{self.name} needs a {self.ORDER}x{self.ORDER} matrix")
