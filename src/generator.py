"""Generator — search orchestration: Builder -> Tracker -> Solver."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from .builder import Builder
from .constructions import double_gs4, paley_ng_sequences, supports_paley_ng
from .metrics import Metrics, check_orthogonality
from .solver import SearchStats, SolverConfig
from .solver import search as ils_search


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


def _cyclic_start(n: int, rng: np.random.Generator) -> npt.NDArray[np.int8]:
    """One random base sequence, cyclic shifts for the other three."""
    base = rng.choice(np.array([-1, 1], dtype=np.int8), size=n)
    shift = rng.integers(1, n, size=3)
    seqs = np.empty((4, n), dtype=np.int8)
    seqs[0] = base
    for i, s in enumerate(shift, 1):
        seqs[i] = np.roll(base, int(s))
    return seqs


class Generator:
    """Hadamard search via KFlip + NAF Tracker (Goethals-Seidel GS4)."""

    def __init__(self, *, kind: str, n: int) -> None:
        if kind == "paley-ng":
            if n <= 0:
                raise ValueError("paley-ng requires positive n")
            if n % 2:
                raise ValueError("paley-ng requires even n")
            if not supports_paley_ng(n):
                raise ValueError(f"paley-ng requires p=2n-1 to be prime, got {2 * n - 1}")
            self._builder = Builder(kind="gs4", n=n)
        elif kind == "construct":
            self._builder = Builder(kind="gs4", n=n)
        else:
            self._builder = Builder(kind=kind, n=n)
        self._kind = kind
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
        divisor = 4 if name in {"construct", "paley-ng"} else Builder.k_for(name)
        if order <= 0 or order % divisor:
            raise ValueError(f"{name} requires a positive order divisible by {divisor}")
        return cls(kind=name, n=order // divisor)

    @property
    def name(self) -> str:
        return self._kind

    def search(
        self,
        steps: int,
        seed: int,
        *,
        config: SolverConfig | None = None,
        start_kind: str = "random",
    ) -> Result:
        started = time.perf_counter()
        rng = np.random.default_rng(seed)

        if self.tensor_n is not None:
            return self._tensor_search(steps, seed, started)

        if self._kind == "paley-ng":
            sequences = paley_ng_sequences(self._builder.n)
            matrix = self._builder.build(sequences)
            return Result(
                matrix=matrix,
                metrics=check_orthogonality(matrix),
                elapsed=time.perf_counter() - started,
                seed=seed,
                sequences=sequences,
            )

        if self._kind == "construct":
            constructed = self._construct(steps, seed, started, config)
            if constructed is not None:
                return constructed

        b = self._builder
        n = b.n

        if start_kind == "cyclic":
            sequences = _cyclic_start(n, rng)
        else:
            sequences = rng.choice(np.array([-1, 1], dtype=np.int8), size=(b.k, n))

        from .tracker import Tracker

        tracker = Tracker()
        tracker.build(sequences)
        solver_config = config if config is not None else SolverConfig()
        best_seq, best_e, iters, stats = ils_search(
            sequences,
            tracker,
            rng,
            steps=steps,
            config=solver_config,
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
        assert self.tensor_n is not None
        n1, n2 = self.tensor_n

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

    def _construct(
        self,
        steps: int,
        seed: int,
        started: float,
        config: SolverConfig | None,
    ) -> Result | None:
        n = self._builder.n
        if supports_paley_ng(n):
            sequences = paley_ng_sequences(n)
            matrix = self._builder.build(sequences)
            return Result(
                matrix=matrix,
                metrics=check_orthogonality(matrix),
                elapsed=time.perf_counter() - started,
                seed=seed,
                sequences=sequences,
            )
        if n % 2:
            return None

        base = Generator(kind="construct", n=n // 2).search(steps=steps, seed=seed, config=config)
        if base.metrics.energy != 0 or base.sequences is None:
            return None
        sequences = double_gs4(base.sequences)
        matrix = self._builder.build(sequences)
        return Result(
            matrix=matrix,
            metrics=check_orthogonality(matrix),
            elapsed=time.perf_counter() - started,
            seed=seed,
            iterations=base.iterations,
            sequences=sequences,
            stats=base.stats,
        )

    def _paley_ng_sequences(self) -> npt.NDArray[np.int8]:
        """Compatibility hook for experiments; production logic lives in constructions."""
        return paley_ng_sequences(self._builder.n)
