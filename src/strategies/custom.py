"""CustomSolver — Hadamard search via KFlip with GramTracker."""

from __future__ import annotations

import time

import numpy as np

from builders import Builder
from gpu import check_orthogonality
from strategies.base import Result, SearchStrategy
from tracker import GramTracker

from .kflip import search as ils_search


class CustomSolver(SearchStrategy):
    """Hadamard search via KFlip + GramTracker.

    ``CustomSolver(kind="gs4", n)`` — 4 negacyclic blocks, GS4 (default).
    ``CustomSolver(kind="golay_2n", n)`` — 2 negacyclic blocks, Golay pair.
    ``CustomSolver(kind="gs4_group", n)`` — 4 group-circulant blocks, GS4.
    """

    def __init__(self, *, kind: str = "gs4", n: int):
        self._builder = Builder(kind=kind, n=n)
        self.tensor_n: tuple[int, int] | None = None
        self.ORDER = self._builder.order

    @classmethod
    def tensor(cls, *, n1: int, n2: int) -> CustomSolver:
        cs = cls(n=max(n1, n2))
        cs.tensor_n = (n1, n2)
        cs.ORDER = 16 * n1 * n2
        return cs

    @property
    def name(self) -> str:
        return self._builder.kind

    @property
    def construction(self) -> str:
        return self._builder.kind

    @staticmethod
    def _fail_result(order, elapsed, seed):
        return Result(
            matrix=np.ones((order, order), dtype=np.int8),
            metrics=check_orthogonality(np.ones((order, order), dtype=np.int8)),
            elapsed=elapsed,
            seed=seed,
        )

    def search(self, steps=None, seed=0) -> Result:
        started = time.perf_counter()
        rng = np.random.default_rng(seed)

        if self.tensor_n is not None:
            return self._tensor_search(steps, seed, started)

        b = self._builder
        budget = steps if steps is not None else None

        sequences = np.zeros((b.k, b.n), dtype=np.int8)
        for i in range(b.k):
            sequences[i] = rng.choice((-1, 1), size=b.n).astype(np.int8)

        tracker = GramTracker(b.build)
        best_seq, best_e = ils_search(sequences, tracker, rng, budget=budget)
        elapsed = time.perf_counter() - started

        matrix = b.build(best_seq) if best_e == 0 else np.ones((b.order, b.order), dtype=np.int8)

        metrics = check_orthogonality(matrix)
        if best_e == 0 and metrics.energy == 0:
            print(
                f"  seed={seed} VALID {matrix.shape[0]}x{matrix.shape[1]}{b.label} {elapsed:.1f}s"
            )
        else:
            print(f"  seed={seed} best_e={best_e}{b.label} {elapsed:.1f}s")

        return Result(
            matrix=matrix, metrics=metrics, elapsed=elapsed, seed=seed, sequences=best_seq
        )

    def _tensor_search(self, steps, seed, started):
        n1, n2 = self.tensor_n  # type: ignore[reportGeneralTypeIssues]

        def _fail():
            return self._fail_result(self.ORDER, time.perf_counter() - started, seed)

        r1 = CustomSolver(kind="gs4", n=n1).search(steps=steps, seed=seed)
        if r1.metrics.energy != 0:
            return _fail()

        r2 = CustomSolver(kind="gs4", n=n2).search(steps=steps, seed=seed + 1)
        if r2.metrics.energy != 0:
            return _fail()

        H = np.kron(r1.matrix, r2.matrix)
        m = check_orthogonality(H)
        elapsed = time.perf_counter() - started
        if m.energy == 0:
            print(
                f"  seed={seed} VALID {H.shape[0]}x{H.shape[1]}"
                f" [tensor {r1.matrix.shape[0]}x{r2.matrix.shape[0]}] {elapsed:.1f}s"
            )
        return Result(matrix=H, metrics=m, elapsed=elapsed, seed=seed)
