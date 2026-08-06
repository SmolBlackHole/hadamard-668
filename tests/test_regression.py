"""Regression tests: n=24 must solve 100% with random seeds."""

from __future__ import annotations

import numpy as np

from src.generator import Generator


def test_n24_solves_all_random_seeds() -> None:
    """30 random seeds at n=24, 200k steps — must all reach energy 0."""
    gen = Generator(kind="gs4", n=24)
    seeds = np.random.default_rng(12345).integers(0, 1_000_000, size=30)
    for s in seeds:
        r = gen.search(steps=200_000, seed=int(s))
        assert r.metrics.energy == 0, f"seed={s} failed with e={r.metrics.energy}"
