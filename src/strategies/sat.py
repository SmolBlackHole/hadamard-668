"""Inkrementelle Z3-Suche mit Pseudo-Boolean-Zeilenpaarbedingungen."""
from __future__ import annotations

import time

import numpy as np
import z3

from gpu import check_orthogonality
from .base import SearchStrategy


class SatSearch(SearchStrategy):
    """Ergänzt verletzte Zeilenpaarbedingungen inkrementell in Z3.

    Zweck: Erprobt eine Pseudo-Boolean-SAT-Formulierung ohne alle Bedingungen vorab zu materialisieren.
    Mechanik: Normalisiert erste Zeile und Spalte, löst, misst und fügt den größten Verstoß hinzu.
    Grundlage: Für jedes Zeilenpaar müssen genau ``n / 2`` Spalten unterschiedliche Vorzeichen besitzen, also ist ihr Skalarprodukt null.
    Pipeline: Kann nur eine Pipeline eröffnen, weil keine ``refine``-Methode existiert.
    Grenzen: Das Schrittbudget begrenzt nachgeladene Constraints und ist kein Vollbeweis für Ordnung 668.
    """

    def __init__(self, order: int = SearchStrategy.ORDER) -> None:
        if order < 2 or order % 2:
            raise ValueError("sat search requires a positive even order")
        self.ORDER = order

    @property
    def name(self) -> str:
        return "sat"

    def _matrix(self, model: z3.ModelRef, variables: list[list[z3.BoolRef]]) -> np.ndarray:
        return np.array([
            [1 if z3.is_true(model.eval(value, model_completion=True)) else -1
             for value in row]
            for row in variables
        ], dtype=np.int8)

    def search(self, steps: int, seed: int) -> tuple[np.ndarray, dict[str, int], float]:
        del seed
        started = time.perf_counter()
        solver = z3.Solver()
        solver.set(timeout=max(1, min(steps, 1_000)) * 100)
        variables = [[z3.Bool(f"h_{row}_{column}") for column in range(self.ORDER)]
                     for row in range(self.ORDER)]
        for value in variables[0]:
            solver.add(value)
        for row in variables:
            solver.add(row[0])
        best_matrix = np.ones((self.ORDER, self.ORDER), dtype=np.int8)
        best_metrics = check_orthogonality(best_matrix)
        constrained: set[tuple[int, int]] = set()
        for _ in range(max(1, steps)):
            if solver.check() != z3.sat:
                break
            matrix = self._matrix(solver.model(), variables)
            metrics = check_orthogonality(matrix)
            if metrics["energy"] < best_metrics["energy"]:
                best_matrix, best_metrics = matrix, metrics
            if metrics["energy"] == 0:
                break
            gram = matrix.astype(np.int64) @ matrix.astype(np.int64).T
            np.fill_diagonal(gram, 0)
            index = int(np.argmax(np.abs(gram)))
            first, second = divmod(index, self.ORDER)
            pair = tuple(sorted((first, second)))
            if first == second or pair in constrained:
                break
            constrained.add(pair)
            differences = [z3.If(variables[first][column] != variables[second][column], 1, 0)
                           for column in range(self.ORDER)]
            solver.add(z3.Sum(differences) == self.ORDER // 2)
        return best_matrix, best_metrics, time.perf_counter() - started
