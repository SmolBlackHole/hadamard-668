"""Hierarchical construction: TT(n_seed) prefix + spectral_descent + KFlip."""

from __future__ import annotations

import time

import numpy as np

from .base import Result, TurynStrategy


class HierarchicalConstruction(TurynStrategy):
    """Fix TT(n_seed) as prefix, search full TT(n) via spectral_descent + KFlip."""

    def __init__(
        self,
        *,
        n: int = TurynStrategy.DEFAULT_N,
        n_seed: int = 6,
        restarts: int = 15,
        sieve: bool = True,
    ):
        super().__init__(n=n, sieve=sieve)
        if n_seed < 2 or n_seed >= n:
            raise ValueError(f"n_seed={n_seed} must be in [2, {n - 1}]")
        self.n_seed = n_seed
        self.restarts = restarts
        self.ORDER = 4 * (3 * n - 1)

    @property
    def name(self) -> str:
        return "hierarchical"

    def search(self, steps: int, seed: int, sequences: np.ndarray | None = None) -> Result:
        from fixtures import tt_sequences
        from strategies.kflip import KFlipRepair
        from strategies.spectral_descent import TurynSpectralDescentSearch

        started = time.perf_counter()
        rng = np.random.default_rng(seed)
        small_sol = tt_sequences(self.n_seed, 0)
        best_e_ever = float("inf")
        best_state_ever = None

        for restart in range(self.restarts):
            # Step 1: spectral_descent
            sd = TurynSpectralDescentSearch(n=self.N, batch_size=128, sieve=self.sieve)
            rr = np.random.default_rng(seed + restart * 1000)
            sr = sd.search(steps=max(50, 10 * self.N), seed=int(rr.integers(0, 2**31)))
            state = sr.sequences.copy() if sr.sequences is not None else self.seed(rr)

            # Step 2: soft prefix
            state[0, : self.n_seed] = small_sol[0, : self.n_seed]
            state[1, : self.n_seed] = small_sol[1, : self.n_seed]
            state[2, : self.n_seed] = small_sol[2, : self.n_seed]
            state[3, : self.n_seed - 1] = small_sol[3, : self.n_seed - 1]
            state[3, self.N - 1] = 0

            # Step 3: KFlip on all positions
            kf = KFlipRepair(n=self.N, sieve=False, max_k=4)
            kr = kf.search(steps=2000, seed=seed + restart, sequences=state)
            e = kr.metrics["energy"]
            state = kr.sequences if kr.sequences is not None else state

            if e < best_e_ever:
                best_e_ever = e
                best_state_ever = state.copy()
            if best_e_ever == 0:
                break

        if best_state_ever is None:
            best_state_ever = self.seed(rng)
            best_e_ever = float("inf")

        matrix, metrics = self.build(best_state_ever)
        elapsed = time.perf_counter() - started
        print(f"  seed={seed} best_e={best_e_ever:.0f} {elapsed:.1f}s")
        return Result(matrix, metrics, elapsed, best_state_ever)

    def refine(self, matrix, steps, seed, sequences=None):
        if sequences is not None:
            return self.search(steps, seed, sequences)
        return self.search(steps, seed)
