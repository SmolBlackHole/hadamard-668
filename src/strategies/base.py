"""Shared contract: Result, Metrics, SearchStrategy, Pipeline."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from gpu import Metrics, check_orthogonality


@dataclass
class Result:
    matrix: np.ndarray
    metrics: Metrics
    elapsed: float
    seed: int
    sequences: np.ndarray | None = None
    hamming: int | None = None


class SearchStrategy(ABC):
    ORDER = 668
    gpu_exclusive: bool = False

    @abstractmethod
    def search(self, steps: int, seed: int) -> Result: ...

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    def construction(self) -> str:
        return "unknown"

    @property
    def order(self) -> int:
        return self.ORDER

    def hamming(self) -> int | None:
        return None

    def refine(self, matrix, steps, seed, sequences=None):
        raise NotImplementedError(f"{self.name} cannot refine")


class Pipeline(SearchStrategy):
    def __init__(self, stages: list[tuple[SearchStrategy, int]]) -> None:
        if not stages:
            raise ValueError("a pipeline needs at least one stage")
        order = stages[0][0].ORDER
        if any(order != s.ORDER for s, _ in stages):
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

    def hamming(self) -> int | None:
        return self._stages[0][0].hamming()

    def search(self, steps: int, seed: int) -> Result:
        s, steps_s = self._stages[0]
        carry = s.search(steps_s, seed)
        total_elapsed = carry.elapsed
        if carry.sequences is not None:
            self._last_seq = carry.sequences
        if carry.metrics.energy == 0:
            m = check_orthogonality(carry.matrix)
            if m.energy == 0:
                return Result(carry.matrix, m, total_elapsed, seed, carry.sequences)

        for idx, (s, steps_s) in enumerate(self._stages[1:], start=1):
            carry = s.refine(carry.matrix, steps_s, seed + idx, carry.sequences)
            total_elapsed += carry.elapsed
            if carry.sequences is not None:
                self._last_seq = carry.sequences
            if carry.metrics.energy == 0:
                m = check_orthogonality(carry.matrix)
                if m.energy == 0:
                    return Result(carry.matrix, m, total_elapsed, seed, carry.sequences)

        return Result(carry.matrix, carry.metrics, total_elapsed, seed, carry.sequences)
