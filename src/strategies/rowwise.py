"""Zeilenweiser Aufbau orthogonaler Vorzeichenkandidaten."""
from __future__ import annotations


import itertools
import time

import numpy as np

from .base import SearchStrategy
from gpu import check_orthogonality


class RowwiseSearch(SearchStrategy):
    """Baut Vorzeichenmatrizen Zeile für Zeile aus residualarmen Kandidaten auf.

    Zweck: Liefert einen konstruktiven Suchversuch statt eines globalen Energieflips.
    Mechanik: Enumeriert klein exakt; große Ordnungen bewerten Zufallszeilen batchweise und verbessern die beste per Bitflips.
    Grundlage: Die Summe quadrierter Residuen zu vorhandenen Zeilen ist der neue Beitrag zur Gram-Energie.
    Pipeline: Kann nur eine Pipeline eröffnen, weil keine ``refine``-Methode existiert.
    Grenzen: Echtes Backtracking ist auf Ordnung höchstens 20 begrenzt; große Ordnungen bleiben eine stochastische Greedy-Heuristik.
    """

    def __init__(
        self,
        order: int = SearchStrategy.ORDER,
        batch_size: int = 256,
        local_flips: int = 2,
    ) -> None:
        if order < 1 or order % 2:
            raise ValueError("rowwise search requires a positive even order")
        if batch_size < 1 or local_flips < 0:
            raise ValueError("batch_size must be positive and local_flips nonnegative")
        self.ORDER = order
        self.batch_size = batch_size
        self.local_flips = local_flips
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

    @staticmethod
    def _improve_candidate(
        existing: np.ndarray,
        candidate: np.ndarray,
        max_flips: int,
    ) -> np.ndarray:
        improved = candidate.copy()
        wide = existing.astype(np.int64, copy=False)
        residual = wide @ improved.astype(np.int64)
        for _ in range(max_flips):
            projections = residual @ wide
            deltas = 4 * len(existing) - 4 * improved * projections
            deltas[0] = np.iinfo(np.int64).max
            column = int(np.argmin(deltas))
            if deltas[column] >= 0:
                break
            old_value = int(improved[column])
            residual -= 2 * old_value * wide[:, column]
            improved[column] = -old_value
        return improved

    def _best_batch(
        self,
        existing: np.ndarray,
        count: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        best: np.ndarray | None = None
        best_score = float("inf")
        remaining = count
        wide = existing.astype(np.float32, copy=False)
        while remaining:
            size = min(self.batch_size, remaining)
            candidates = rng.choice(
                (-1, 1), size=(size, self.ORDER)).astype(np.int8)
            candidates[:, 0] = 1
            residuals = wide @ candidates.astype(np.float32).T
            scores = np.sum(residuals * residuals, axis=0)
            index = int(np.argmin(scores))
            if scores[index] < best_score:
                best, best_score = candidates[index].copy(), float(scores[index])
            remaining -= size
        assert best is not None
        return self._improve_candidate(existing, best, self.local_flips)

    def search(self, steps: int, seed: int) -> tuple[np.ndarray, dict[str, int], float]:
        started = time.perf_counter()
        rng = np.random.default_rng(seed)
        matrix = np.ones((self.ORDER, self.ORDER), dtype=np.int8)
        self._attempts = 0
        self._attempt_limit = max(steps, 1)

        if self.ORDER <= 20:
            self._solve_row(matrix, 1, self._candidates(), rng)
        else:
            # ponytail: random batches are a heuristic ceiling; use structured row generators if residual quality plateaus.
            rows = self.ORDER - 1
            remaining_budget = max(steps, rows)
            for row_index in range(1, self.ORDER):
                remaining_rows = self.ORDER - row_index
                count = max(1, remaining_budget // remaining_rows)
                matrix[row_index] = self._best_batch(
                    matrix[:row_index], count, rng)
                self._attempts += count
                remaining_budget -= count

        elapsed = time.perf_counter() - started
        metrics = check_orthogonality(matrix)
        print(
            f"  seed={seed} attempts={self._attempts} energy={metrics['energy']} {elapsed:.1f}s")
        return matrix, metrics, elapsed
