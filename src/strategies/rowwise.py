"""Zeilenweiser Aufbau orthogonaler Vorzeichenkandidaten."""
from __future__ import annotations


import itertools
import time

import numpy as np

from .base import SearchStrategy
from gpu import check_orthogonality


class RowwiseSearch(SearchStrategy):
    """Baut Vorzeichenmatrizen Zeile für Zeile aus orthogonalen Kandidaten auf.

    Zweck: Liefert einen konstruktiven Suchversuch statt eines globalen Energieflips.
    Mechanik: Enumeriert für kleine Ordnungen balancierte Zeilen mit Backtracking; sonst sampled es Zufallszeilen.
    Grundlage: Jede neue Zeile muss zu allen bereits gesetzten Zeilen das Skalarprodukt null besitzen.
    Pipeline: Kann nur eine Pipeline eröffnen, weil keine ``refine``-Methode existiert.
    Grenzen: Echtes Backtracking ist auf Ordnung höchstens 20 begrenzt; 668 nutzt nur begrenzte Zufallsversuche.
    """

    def __init__(self, order: int = SearchStrategy.ORDER) -> None:
        if order < 1 or order % 2:
            raise ValueError("rowwise search requires a positive even order")
        self.ORDER = order
        self._attempts = 0
        self._attempt_limit = 0

    @property
    def name(self) -> str:
        return "rowwise"

    def _candidates(self) -> list[np.ndarray]:
        positives = self.ORDER // 2 - 1
        candidates: list[np.ndarray] = []
        for positive_columns in itertools.combinations(range(1, self.ORDER), positives):
            row = -np.ones(self.ORDER, dtype=np.int8)
            row[0] = 1
            row[list(positive_columns)] = 1
            candidates.append(row)
        return candidates

    def _solve_row(
        self,
        matrix: np.ndarray,
        row_index: int,
        candidates: list[np.ndarray],
        rng: np.random.Generator,
    ) -> bool:
        if row_index == self.ORDER:
            return True
        for candidate_index in rng.permutation(len(candidates)):
            if self._attempts >= self._attempt_limit:
                return False
            self._attempts += 1
            candidate = candidates[int(candidate_index)]
            if np.any(matrix[:row_index].astype(np.int64) @ candidate.astype(np.int64)):
                continue
            matrix[row_index] = candidate
            if self._solve_row(matrix, row_index + 1, candidates, rng):
                return True
            self._backtrack(matrix, row_index)
        return False

    @staticmethod
    def _backtrack(matrix: np.ndarray, row_index: int) -> None:
        matrix[row_index] = 1

    def search(self, steps: int, seed: int) -> tuple[np.ndarray, dict[str, int], float]:
        started = time.perf_counter()
        rng = np.random.default_rng(seed)
        matrix = np.ones((self.ORDER, self.ORDER), dtype=np.int8)
        self._attempts = 0
        self._attempt_limit = max(steps, 1)

        # ponytail: exact candidate enumeration is limited to small orders; use a SAT solver if 668-row proof search becomes viable.
        if self.ORDER <= 20:
            self._solve_row(matrix, 1, self._candidates(), rng)
        else:
            for row_index in range(1, self.ORDER):
                solved = False
                for _ in range(max(1, steps // self.ORDER)):
                    candidate = rng.choice(
                        [-1, 1], size=self.ORDER).astype(np.int8)
                    if np.any(matrix[:row_index].astype(np.int64) @ candidate.astype(np.int64)):
                        continue
                    matrix[row_index] = candidate
                    solved = True
                    break
                if not solved:
                    break

        elapsed = time.perf_counter() - started
        metrics = check_orthogonality(matrix)
        print(
            f"  seed={seed} attempts={self._attempts} energy={metrics['energy']} {elapsed:.1f}s")
        return matrix, metrics, elapsed
