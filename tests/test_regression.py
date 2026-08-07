"""Regression tests: must solve 100% with random seeds."""

from __future__ import annotations

import numpy as np
import pytest

from src.generator import Generator

REGRESSION_PARAMS = [
    (24, 100_000, 30),
    (32, 100_000, 30),
]


@pytest.mark.parametrize("n,steps,n_seeds", REGRESSION_PARAMS)
def test_regression_solves_all_random_seeds(n: int, steps: int, n_seeds: int) -> None:
    gen = Generator(kind="gs4", n=n)
    rng = np.random.default_rng(12345)
    seeds = rng.integers(0, 1_000_000, size=n_seeds)
    for s in seeds:
        r = gen.search(steps=steps, seed=int(s))
        assert r.metrics.energy == 0, f"n={n} seed={s} failed with e={r.metrics.energy}"
