"""Regression tests: must solve 100% with random seeds."""

from __future__ import annotations

import numpy as np
import pytest

from src.pipeline import execute
from src.solver import SolverConfig

REGRESSION_PARAMS = [
    (24, 50_000, 30),
    (26, 50_000, 30),
    (28, 50_000, 30),
    (30, 50_000, 30),
    (32, 50_000, 30),
    (34, 50_000, 30),
    (36, 50_000, 30),
    (38, 50_000, 30),
    (40, 50_000, 30),
]


@pytest.mark.parametrize("n,steps,n_seeds", REGRESSION_PARAMS)
def test_regression_solves_all_random_seeds(n: int, steps: int, n_seeds: int) -> None:
    config = SolverConfig()
    rng = np.random.default_rng(12345)
    seeds = rng.integers(0, 1_000_000, size=n_seeds)
    for s in seeds:
        result = execute("gs4", n, steps, int(s), config)
        assert result.solved, f"n={n} seed={s} failed with e={result.energy}"
