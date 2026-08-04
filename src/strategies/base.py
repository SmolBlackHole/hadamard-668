"""Gemeinsamer Vertrag und sequentielle Pipeline fuer Suchstrategien."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import NamedTuple

import numpy as np

from gpu import check_orthogonality


class Result(NamedTuple):
    matrix: np.ndarray
    metrics: dict[str, int]
    elapsed: float


class SearchStrategy(ABC):
    ORDER = 668

    @abstractmethod
    def search(self, steps: int, seed: int) -> Result: ...
    @property
    @abstractmethod
    def name(self) -> str: ...
    @property
    def construction(self) -> str: return "unknown"
    @property
    def order(self) -> int: return self.ORDER

    def refine(self, matrix: np.ndarray, steps: int, seed: int) -> Result:
        raise NotImplementedError(f"{self.name} cannot refine")

    def check(self, matrix: np.ndarray) -> dict[str, int]:
        from gpu import check_orthogonality
        return check_orthogonality(matrix)


class Pipeline(SearchStrategy):
    def __init__(self, stages: list[tuple[SearchStrategy, int]]) -> None:
        if not stages:
            raise ValueError("a pipeline needs at least one stage")
        order = stages[0][0].ORDER
        if any(s.ORDER != order for s, _ in stages):
            raise ValueError("all pipeline stages must use the same order")
        for s, _ in stages[1:]:
            if type(s).refine is SearchStrategy.refine:
                raise ValueError(f"{s.name} cannot refine a candidate")
        self._stages = stages
        self.ORDER = order

    @property
    def name(self) -> str:
        return "->".join(s.name for s, _ in self._stages)

    @property
    def construction(self) -> str:
        return self._stages[0][0].construction

    def search(self, steps: int, seed: int) -> Result:
        matrix = np.empty((1, 1), dtype=np.int8)
        metrics = {"energy": 2**63, "orthogonal_pairs": 0,
                   "max_abs_correlation": 0}
        elapsed = 0.0
        for idx, (strategy, steps_s) in enumerate(self._stages):
            matrix, metrics, step_elapsed = (strategy.search(steps_s, seed + idx) if idx == 0
                                             else strategy.refine(matrix, steps_s, seed + idx))
            elapsed += step_elapsed
            if metrics["energy"] == 0:
                metrics = check_orthogonality(matrix)
                if metrics["energy"] == 0:
                    break
        return Result(matrix, metrics, elapsed)
