"""CustomSolver — Negacyclic-GS4 Hadamard-Suche via KFlip."""

from __future__ import annotations

import time

import numpy as np

from builders import build_group_gs4, build_negacyclic_gs4
from energy import group_gs4_energy, negacyclic_gs4_energy
from gpu import check_orthogonality
from strategies.base import Result, SearchStrategy

from .kflip import search as ils_search


class CustomSolver(SearchStrategy):
    """Negacyclic GS4 Hadamard search.

    ``CustomSolver.negacyclic(n)`` — negacyclic GS4 (default).
    ``CustomSolver.group(n)`` — group-circulant GS4.
    ``CustomSolver.tensor(n1, n2)`` — Kronecker product of two negacyclic.
    """

    def __init__(self, *, n: int, mode: str = "negacyclic", group_dims: list[int] | None = None):
        self._n = n
        self._mode = mode
        self.group_dims = group_dims
        self.tensor_n = None
        self.gpu_exclusive = False
        self.ORDER = 4 * n

    @classmethod
    def negacyclic(cls, *, n: int) -> CustomSolver:
        return cls(n=n, mode="negacyclic")

    @classmethod
    def group(cls, *, n: int) -> CustomSolver:
        dims = _best_factorization(n)
        return cls(n=n, mode="group", group_dims=dims)

    @classmethod
    def tensor(cls, *, n1: int, n2: int) -> CustomSolver:
        cs = cls(n=max(n1, n2), mode="negacyclic")
        cs.tensor_n = (n1, n2)
        cs.ORDER = 16 * n1 * n2
        return cs

    @classmethod
    def from_order(cls, order: int) -> CustomSolver:
        return cls.negacyclic(n=order // 4)

    @property
    def name(self) -> str:
        if self._mode == "group" and self.group_dims:
            return f"custom[{self.group_dims[0]}x{self.group_dims[1]}]"
        return "custom"

    @property
    def construction(self) -> str:
        return "negacyclic_gs4" if self._mode != "group" else "group_gs4"

    def search(self, steps=None, seed=0, sequences=None) -> Result:
        started = time.perf_counter()
        rng = np.random.default_rng(seed)

        if self.tensor_n is not None:
            return self._tensor_search(steps, seed, started)

        budget = steps if steps is not None else None
        n = self._n

        if sequences is None:
            sequences = np.zeros((4, n), dtype=np.int8)
            for i in range(4):
                sequences[i] = rng.choice((-1, 1), size=n).astype(np.int8)

        if self._mode == "group" and self.group_dims is not None:
            dims = self.group_dims

            def ef(s):
                return group_gs4_energy(s, dims)
        else:
            ef = negacyclic_gs4_energy

        best_seq, best_e = ils_search(sequences, ef, rng, budget=budget)
        elapsed = time.perf_counter() - started

        dims_label = ""
        if best_e == 0:
            if self._mode == "group" and self.group_dims is not None:
                matrix = build_group_gs4(best_seq, self.group_dims)
                dims_label = f" [{self.group_dims}]"
            else:
                matrix = build_negacyclic_gs4(
                    best_seq[0, :n], best_seq[1, :n], best_seq[2, :n], best_seq[3, :n]
                )
        else:
            matrix = np.ones((self.ORDER, self.ORDER), dtype=np.int8)

        metrics = check_orthogonality(matrix)
        if best_e == 0 and metrics.energy == 0:
            print(
                f"  seed={seed} VALID {matrix.shape[0]}x{matrix.shape[1]}{dims_label} {elapsed:.1f}s"
            )
        else:
            print(f"  seed={seed} best_e={best_e}{dims_label} {elapsed:.1f}s")

        return Result(
            matrix=matrix, metrics=metrics, elapsed=elapsed, seed=seed, sequences=best_seq
        )

    def _tensor_search(self, steps, seed, started):
        n1, n2 = self.tensor_n
        r1 = CustomSolver.negacyclic(n=n1).search(steps=steps, seed=seed)
        if r1.metrics.energy != 0:
            return Result(
                matrix=np.ones((self.ORDER, self.ORDER), dtype=np.int8),
                metrics=check_orthogonality(np.ones((self.ORDER, self.ORDER), dtype=np.int8)),
                elapsed=time.perf_counter() - started,
                seed=seed,
            )
        r2 = CustomSolver.negacyclic(n=n2).search(steps=steps, seed=seed + 1)
        if r2.metrics.energy != 0:
            return Result(
                matrix=np.ones((self.ORDER, self.ORDER), dtype=np.int8),
                metrics=check_orthogonality(np.ones((self.ORDER, self.ORDER), dtype=np.int8)),
                elapsed=time.perf_counter() - started,
                seed=seed,
            )
        H = np.kron(r1.matrix, r2.matrix)
        m = check_orthogonality(H)
        elapsed = time.perf_counter() - started
        if m.energy == 0:
            print(
                f"  seed={seed} VALID {H.shape[0]}x{H.shape[1]} [tensor {r1.matrix.shape[0]}x{r2.matrix.shape[0]}] {elapsed:.1f}s"
            )
        return Result(matrix=H, metrics=m, elapsed=elapsed, seed=seed)


def _best_factorization(n: int) -> list[int]:
    best = [n]
    best_diff = n
    for p in range(3, int(n**0.5) + 1):
        if n % p == 0:
            q = n // p
            if q >= 3:
                diff = abs(p - q)
                if diff < best_diff:
                    best_diff = diff
                    best = sorted([p, q], reverse=True)
    return best
