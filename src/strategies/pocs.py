"""Douglas-Rachford splitting between NPAF power shell and {+-1}."""

from __future__ import annotations

import time

import numpy as np

from correlations import npa_f_residual, project_fourier, project_sign
from gpu import to_numpy, xp

from .base import Result, TurynStrategy


class PocsSearch(TurynStrategy):
    """Douglas-Rachford projection between NPAF power shell and {+-1}."""

    gpu_exclusive = True

    def __init__(self, *, n: int, inner_steps: int = 50, batch_size: int = 256, sieve: bool = True):
        super().__init__(n=n, sieve=sieve)
        self.inner_steps = inner_steps
        self.batch_size = batch_size

    @classmethod
    def from_order(cls, order: int) -> PocsSearch:
        n = (order // 4 + 1) // 3
        if order != 4 * (3 * n - 1) or n < 2:
            raise ValueError(f"order {order} has no TT(n) construction")
        return cls(n=n)

    @property
    def name(self) -> str:
        return "pocs"

    def search(self, steps: int, seed: int) -> Result:
        started = time.perf_counter()
        rng = xp.random.default_rng(seed)
        batch_size = 1 if xp is np else self.batch_size
        state = (
            self.seed_batch(batch_size, rng).astype(xp.float32)
            if self.sieve
            else rng.standard_normal((batch_size, 4, self.N)).astype(xp.float32)
        )
        best = project_sign(state)
        best_energy = xp.sum(
            npa_f_residual(best, lengths=self.LENGTHS, weights=self.WEIGHTS) ** 2, axis=1
        )
        for _ in range(steps):
            for _ in range(self.inner_steps):
                proj = project_fourier(state, lengths=self.LENGTHS, weights=self.WEIGHTS)
                state = (state + project_sign(2 * proj - state) - proj).astype(xp.float32)
            candidate = project_sign(
                project_fourier(state, lengths=self.LENGTHS, weights=self.WEIGHTS)
            )
            energy = xp.sum(
                npa_f_residual(candidate, lengths=self.LENGTHS, weights=self.WEIGHTS) ** 2, axis=1
            )
            improved = energy < best_energy
            best = xp.where(improved[:, None, None], candidate, best)
            best_energy = xp.minimum(best_energy, energy)

        best_np = to_numpy(best[int(xp.argmin(best_energy))])
        matrix, metrics = self.build(best_np)
        return Result(
            matrix=matrix,
            metrics=metrics,
            elapsed=time.perf_counter() - started,
            seed=seed,
            sequences=best_np,
        )
