"""Z3-Suche über vier zyklische Bool-Folgen und ihre Shift-Differenzen."""
from __future__ import annotations

import math
import time

import numpy as np
import z3

from constructions import build_goethals_seidel
from gpu import check_orthogonality
from .base import SearchStrategy


class SatSearch(SearchStrategy):
    """Löst die komplementäre Vier-Folgen-Bedingung als Pseudo-Boolean-Modell.

    Zweck: Reduziert den SAT-Grundraum von einer Vollmatrix auf vier zyklische Folgen.
    Mechanik: Erzwingt pro unabhängigem Shift genau ``2K`` XOR-Unterschiede und bricht globale Vorzeichensymmetrien.
    Grundlage: ``sum_i PAF_i(s)=0`` ist äquivalent zu insgesamt ``2K`` Vorzeichenwechseln beim Shift ``s``.
    Pipeline: Kann nur eine Pipeline eröffnen, weil keine ``refine``-Methode existiert.
    Grenzen: Der Solver darf nach einer echten Zeitgrenze mit ``unknown`` abbrechen; das ist kein Nichtexistenzbeweis.
    """

    def __init__(
        self,
        order: int = SearchStrategy.ORDER,
        timeout_seconds: float = 50.0,
    ) -> None:
        if order < 4 or order % 4:
            raise ValueError("sat search requires an order divisible by four")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self.ORDER = order
        self.K = order // 4
        self.timeout_seconds = timeout_seconds

    @property
    def name(self) -> str:
        return "sat"

    def _variables(self) -> list[list[z3.BoolRef]]:
        return [[z3.Bool(f"s_{sequence}_{index}")
                 for index in range(self.K)] for sequence in range(4)]

    def _solver(
        self,
        variables: list[list[z3.BoolRef]],
        seed: int,
    ) -> z3.Solver:
        solver = z3.Solver()
        solver.set(
            timeout=max(1, math.ceil(self.timeout_seconds * 1000)),
            random_seed=seed,
        )
        for sequence in variables:
            solver.add(sequence[0])
        for shift in range(1, self.K // 2 + 1):
            differences = [
                (variables[sequence][index] !=
                 variables[sequence][(index + shift) % self.K], 1)
                for sequence in range(4)
                for index in range(self.K)
            ]
            solver.add(z3.PbEq(differences, 2 * self.K))
        return solver

    @staticmethod
    def _sequences(
        model: z3.ModelRef,
        variables: list[list[z3.BoolRef]],
    ) -> np.ndarray:
        return np.array([
            [1 if z3.is_true(model.eval(value, model_completion=True)) else -1
             for value in sequence]
            for sequence in variables
        ], dtype=np.int8)

    def search(self, steps: int, seed: int) -> tuple[np.ndarray, dict[str, int], float]:
        del steps
        started = time.perf_counter()
        variables = self._variables()
        solver = self._solver(variables, seed)
        result = solver.check()
        if result == z3.sat:
            sequences = self._sequences(solver.model(), variables)
        else:
            sequences = np.random.default_rng(seed).choice(
                (-1, 1), size=(4, self.K)).astype(np.int8)
            sequences[:, 0] = 1
        matrix = build_goethals_seidel(*sequences)
        return matrix, check_orthogonality(matrix), time.perf_counter() - started
