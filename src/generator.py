"""Generator — search orchestration: Builder → Tracker → Solver."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from builder import Builder
from metrics import Metrics, check_orthogonality
from solver import search as ils_search
from tracker import GramTracker


@dataclass
class Result:
    matrix: np.ndarray
    metrics: Metrics
    elapsed: float
    seed: int
    sequences: np.ndarray | None = None


class Generator:
    """Hadamard search via KFlip + GramTracker.

    ``Generator(kind="gs4", n)`` — 4 negacyclic blocks, Goethals-Seidel.
    ``Generator(kind="golay_2n", n)`` — 2 negacyclic blocks, Golay pair.
    ``Generator(kind="gs4_group", n)`` — 4 group-circulant blocks, GS4.
    """

    _KIND_DIVISOR: ClassVar[dict[str, int]] = {"gs4": 4, "golay_2n": 2, "gs4_group": 4}

    def __init__(self, *, kind: str, n: int):
        self._builder = Builder(kind=kind, n=n)
        self.tensor_n: tuple[int, int] | None = None
        self.ORDER = self._builder.order

    @classmethod
    def tensor(cls, *, n1: int, n2: int) -> Generator:
        g = cls(kind="gs4", n=max(n1, n2))
        g.tensor_n = (n1, n2)
        g.ORDER = 16 * n1 * n2
        return g

    @classmethod
    def from_cli(cls, name: str, order: int) -> Generator:
        if name == "tensor":
            n = order // 16
            dims = _best_factorization(n)
            return cls.tensor(n1=dims[0], n2=dims[1])
        if name not in cls._KIND_DIVISOR:
            raise ValueError(
                f"unknown strategy {name!r}, choose from {[*cls._KIND_DIVISOR, 'tensor']}"
            )
        return cls(kind=name, n=order // cls._KIND_DIVISOR[name])

    @property
    def name(self) -> str:
        return self._builder.kind

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
            return Result(
                matrix=np.ones((self.ORDER, self.ORDER), dtype=np.int8),
                metrics=check_orthogonality(np.ones((self.ORDER, self.ORDER), dtype=np.int8)),
                elapsed=time.perf_counter() - started,
                seed=seed,
            )

        r1 = Generator(kind="gs4", n=n1).search(steps=steps, seed=seed)
        if r1.metrics.energy != 0:
            return _fail()

        r2 = Generator(kind="gs4", n=n2).search(steps=steps, seed=seed + 1)
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


def _best_factorization(n: int) -> list[int]:
    pairs = [(p, n // p) for p in range(3, int(n**0.5) + 1) if n % p == 0 and n // p >= 3]
    if not pairs:
        return [n]
    p, q = min(pairs, key=lambda pq: abs(pq[0] - pq[1]))
    return sorted([p, q], reverse=True)
