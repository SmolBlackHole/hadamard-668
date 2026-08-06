"""Generator — search orchestration: Builder -> Tracker -> Solver."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import numpy.typing as npt

from builder import Builder
from metrics import Metrics, check_orthogonality
from solver import SearchStats
from solver import search as ils_search

if TYPE_CHECKING:
    from solver import SolverConfig


@dataclass
class Result:
    matrix: npt.NDArray[np.int8]
    metrics: Metrics
    elapsed: float
    seed: int
    iterations: int = 0
    sequences: npt.NDArray[np.int8] | None = None
    stats: SearchStats | None = None
    solver_e: int = 0  # tracker.energy() at best_seq (internal search metric)

    def __str__(self) -> str:
        order = self.matrix.shape[0]
        head = f"OK {order}x{order}" if self.metrics.energy == 0 else f"e={self.metrics.energy}"
        line = f"seed={self.seed:<4} {head}  {self.elapsed:.1f}s"
        if self.stats:
            line += "  " + self.stats.display()
        return line


class Generator:
    """Hadamard search via KFlip + NAF Tracker (Goethals-Seidel GS4)."""

    def __init__(self, *, kind: str, n: int) -> None:
        self._builder = Builder(kind=kind, n=n)
        self.tensor_n: tuple[int, int] | None = None
        self.order = self._builder.order

    @classmethod
    def tensor(cls, *, n1: int, n2: int) -> Generator:
        g = cls(kind="gs4", n=max(n1, n2))
        g.tensor_n = (n1, n2)
        g.order = 16 * n1 * n2
        return g

    @classmethod
    def from_cli(cls, name: str, order: int) -> Generator:
        if name == "tensor":
            n = order // 16
            dims = Builder.factorize(n)
            return cls.tensor(n1=dims[0], n2=dims[1])
        divisor = Builder.k_for(name)
        return cls(kind=name, n=order // divisor)

    @property
    def name(self) -> str:
        return self._builder.kind

    def search(self, steps: int, seed: int, *, config: SolverConfig | None = None) -> Result:
        started = time.perf_counter()
        rng = np.random.default_rng(seed)

        if self.tensor_n is not None:
            return self._tensor_search(steps, seed, started)

        b = self._builder

        sequences = rng.choice(np.array([-1, 1], dtype=np.int8), size=(b.k, b.n))

        from tracker import Tracker

        tracker = Tracker()
        tracker.build(sequences)
        best_seq, best_e, iters, stats = ils_search(
            sequences, tracker, rng, steps=steps, config=config
        )
        elapsed = time.perf_counter() - started

        matrix = b.build(best_seq)
        metrics = check_orthogonality(matrix)
        return Result(
            matrix=matrix,
            metrics=metrics,
            elapsed=elapsed,
            seed=seed,
            iterations=iters,
            sequences=best_seq,
            stats=stats,
            solver_e=best_e,
        )

    def _tensor_search(self, steps: int, seed: int, started: float) -> Result:
        n1, n2 = self.tensor_n  # type: ignore[reportGeneralTypeIssues]

        def _fail() -> Result:
            # best_seq not available across the kron boundary; report sentinel
            return Result(
                matrix=np.ones((self.order, self.order), dtype=np.int8),
                metrics=Metrics(energy=-1, orthogonal_pairs=0, max_abs_correlation=0),
                elapsed=time.perf_counter() - started,
                seed=seed,
                solver_e=-1,
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
        return Result(matrix=H, metrics=m, elapsed=elapsed, seed=seed)
