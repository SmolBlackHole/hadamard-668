"""Greedy-Reparatur anhand des stärksten Gram-Matrix-Verstoßes."""

from __future__ import annotations

import time

import numpy as np
from tqdm import tqdm  # pyright: ignore[reportMissingModuleSource]

from gpu import (
    apply_entry_flip,
    entry_flip_deltas,
    gram_matrix,
    metrics_from_gram,
)

from .base import Result, SearchStrategy


class RepairSearch(SearchStrategy):
    """Sucht strikt bessere Flips für das am stärksten korrelierte Zeilenpaar.

    Zweck: Verbessert vorhandene Kandidaten mit einer gezielten lokalen Reparatur.
    Mechanik: Bewertet Flips beider verletzter Zeilen mit exakten O(n)-Deltas und akzeptiert den besten strikt verbessernden Flip.
    Grundlage: Der größte Betrag eines Off-Diagonal-Eintrags von ``H Hᵀ`` bestimmt das Zeilenpaar; die feinen O(n)-Updates bleiben wegen der geringeren Startkosten auf der CPU.
    Pipeline: Kann jede Matrix mit passender Form als Kopie weiterverfeinern.
    Grenzen: Greedy-Descent kann auf Plateaus stoppen und besitzt keine Erfolgsschwelle oder Neustarts.
    """

    compute_backend = "numpy-incremental"

    def __init__(self, order: int = SearchStrategy.ORDER) -> None:
        self.ORDER = order

    @property
    def name(self) -> str:
        return "repair"

    @property
    def construction(self) -> str:
        return "matrix_repair"

    @staticmethod
    def _most_violated_pair(
        gram: np.ndarray,
        row_argmax: np.ndarray,
        row_max: np.ndarray,
    ) -> tuple[int, int, int]:
        row = int(np.argmax(row_max))
        other = int(row_argmax[row])
        return row, other, int(gram[row, other])

    @staticmethod
    def _refresh_violations(
        gram: np.ndarray,
        changed_row: int,
        row_argmax: np.ndarray,
        row_max: np.ndarray,
    ) -> None:
        affected = np.flatnonzero(row_argmax == changed_row)
        if changed_row not in affected:
            affected = np.append(affected, changed_row)
        for row in affected:
            other = int(np.argmax(np.abs(gram[row])))
            row_argmax[row] = other
            row_max[row] = abs(int(gram[row, other]))
        changed = np.abs(gram[:, changed_row])
        improved = changed > row_max
        row_argmax[improved] = changed_row
        row_max[improved] = changed[improved]

    @staticmethod
    def _candidate_columns(
        matrix,
        row: int,
        other: int,
        dot_product: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        same_sign = matrix[row] == matrix[other]
        eligible = np.flatnonzero(same_sign if dot_product > 0 else ~same_sign)
        count = int(eligible.size)
        if count == 0:
            return np.empty(0, dtype=np.int64)
        chosen = rng.choice(count, size=min(3, count), replace=False)
        return np.asarray(eligible[chosen], dtype=np.int64)

    @staticmethod
    def _candidate_moves(matrix, gram, rows: tuple[int, int], columns: np.ndarray):
        moves: list[tuple[int, int, int]] = []
        for row in rows:
            deltas = entry_flip_deltas(matrix, gram, row, columns)
            moves.extend(
                (int(delta), row, int(column))
                for delta, column in zip(deltas, columns, strict=False)
            )
        return moves

    def _improve(self, matrix: np.ndarray, steps: int, seed: int) -> Result:
        started = time.perf_counter()
        rng = np.random.default_rng(seed)
        matrix = np.asarray(matrix, dtype=np.int8).copy()
        gram = gram_matrix(matrix, backend=np)
        energy = metrics_from_gram(gram)["energy"]
        row_argmax = np.argmax(np.abs(gram), axis=1)
        row_max = np.abs(gram[np.arange(self.ORDER), row_argmax])
        best_energy = energy
        accepted = best_at = 0
        with tqdm(total=steps, desc=self.name, unit="steps", dynamic_ncols=True) as bar:
            for step in range(steps):
                bar.update(1)
                row, other, dot_product = self._most_violated_pair(gram, row_argmax, row_max)
                if dot_product == 0:
                    break
                columns = self._candidate_columns(matrix, row, other, dot_product, rng)
                moves = self._candidate_moves(matrix, gram, (row, other), columns)
                if moves:
                    delta, move_row, column = min(moves)
                    if delta < 0:
                        apply_entry_flip(matrix, gram, move_row, column, known_delta=delta)
                        self._refresh_violations(gram, move_row, row_argmax, row_max)
                        energy += delta
                        best_energy = energy
                        best_at = step
                        accepted += 1
                if step % 50 == 0:
                    bar.set_postfix(e=energy, best=best_energy, acc=accepted)
                if best_energy == 0:
                    break
        elapsed = time.perf_counter() - started
        print(
            f"  seed={seed} best_energy={best_energy} found@step={best_at} accepted={accepted} {elapsed:.1f}s"
        )
        metrics = metrics_from_gram(gram)
        return Result(matrix, metrics, elapsed)

    def search(self, steps: int, seed: int) -> Result:
        rng = np.random.default_rng(seed)
        matrix = rng.choice([-1, 1], size=(self.ORDER, self.ORDER)).astype(np.int8)
        return self._improve(matrix, steps, seed)

    def refine(self, matrix: np.ndarray, steps: int, seed: int) -> Result:
        if matrix.shape != (self.ORDER, self.ORDER):
            raise ValueError(f"{self.name} needs a {self.ORDER}x{self.ORDER} matrix")
        return self._improve(matrix.copy(), steps, seed)
