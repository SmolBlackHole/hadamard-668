"""Shared contract and pipeline for search strategies."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np

from builders import build_turyn
from gpu import check_orthogonality
from sieve import seed_turyn_batch


class Result(NamedTuple):
    matrix: np.ndarray
    metrics: dict[str, int]
    elapsed: float


@dataclass
class RunResult:
    seed: int
    matrix: np.ndarray
    energy: int
    orthogonal_pairs: int
    max_off_diagonal: int
    wall: float
    hamming: int | None = None


class SearchStrategy(ABC):
    ORDER = 668

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

    def refine(self, matrix: np.ndarray, steps: int, seed: int) -> Result:
        raise NotImplementedError(f"{self.name} cannot refine")

    def seed(self, rng: np.random.Generator) -> np.ndarray:
        """Generate a starting candidate. Override for custom seeding."""
        raise NotImplementedError(f"{self.name} has no seed method")

    def build(self, sequences: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
        """Build matrix from sequences and compute metrics."""
        raise NotImplementedError(f"{self.name} has no build method")


# ── Turyn shared base ─────────────────────────────────────────────────────────


class TurynStrategy(SearchStrategy):
    """Shared seed + build for all Turyn-type sequence solvers."""

    DEFAULT_N = 56

    def __init__(self, *, n: int = DEFAULT_N, sieve: bool = True):
        if n < 2:
            raise ValueError("Turyn type requires n >= 2")
        self.N = n
        self.LENGTHS = np.array((n, n, n, n - 1), dtype=np.int64)
        self.WEIGHTS = np.array((1, 1, 2, 2), dtype=np.int64)
        self.ORDER = 4 * (3 * n - 1)
        self.sieve = sieve

    @property
    def construction(self) -> str:
        return f"turyn_tt_{self.N}"

    def seed(self, rng: np.random.Generator) -> np.ndarray:
        return self.seed_batch(1, rng)[0]

    def seed_batch(self, size: int, rng: np.random.Generator, *, module=np) -> np.ndarray:
        """Generate ``size`` starting sequences. Returns (size, 4, N)."""
        if self.sieve:
            batch = seed_turyn_batch(self.N, size, rng, module=module)
        else:
            batch = np.zeros((size, 4, self.N), dtype=np.int8)
            for i, L in enumerate(self.LENGTHS):
                batch[:, i, : int(L)] = rng.choice((-1, 1), size=(size, int(L))).astype(np.int8)
        batch[:, 3, -1] = 0
        return batch

    def build(self, sequences: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
        matrix = build_turyn(*(sequences[i, : self.LENGTHS[i]] for i in range(4)))
        self._last_seq = sequences
        return matrix, check_orthogonality(matrix)

    def hamming(self) -> int | None:
        """Bits differing from the known TT(N) solution, or None if unknown."""
        from fixtures import hamming_distance, known_solution

        sol = known_solution(self.N)
        if sol is None or not hasattr(self, "_last_seq"):
            return None
        ref, lengths = sol
        return hamming_distance(self._last_seq, ref, lengths)


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

    def search(self, steps: int, seed: int) -> Result:
        s, steps_s = self._stages[0]
        carry = s.search(steps_s, seed)
        total_elapsed = carry.elapsed
        if carry.metrics["energy"] == 0:
            m = check_orthogonality(carry.matrix)
            if m["energy"] == 0:
                return Result(carry.matrix, m, total_elapsed)
        for idx, (s, steps_s) in enumerate(self._stages[1:], start=1):
            carry = s.refine(carry.matrix, steps_s, seed + idx)
            total_elapsed += carry.elapsed
            if carry.metrics["energy"] == 0:
                m = check_orthogonality(carry.matrix)
                if m["energy"] == 0:
                    return Result(carry.matrix, m, total_elapsed)
        return Result(carry.matrix, carry.metrics, total_elapsed)
