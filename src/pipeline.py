"""End-to-end execution from sequence construction through verification."""

from __future__ import annotations

import time

import numpy as np

from .constructions import double_gs4
from .generator import exact_sequences, initial_sequences
from .models import RunResult, SearchStats
from .solver import SolverConfig, search
from .tracker import Tracker
from .verify import verify_candidate


def execute(
    strategy: str,
    n: int,
    steps: int,
    seed: int,
    solver_config: SolverConfig,
    start_kind: str = "random",
) -> RunResult:
    if n <= 0:
        raise ValueError("n must be positive")
    if steps < 1:
        raise ValueError("steps must be positive")
    started = time.perf_counter()
    exact = exact_sequences(strategy, n)
    if exact is not None:
        verify_candidate(exact)
        return RunResult(
            strategy,
            n,
            seed,
            exact,
            0,
            0,
            time.perf_counter() - started,
            SearchStats(),
            True,
        )

    if strategy == "construct" and n % 2 == 0:
        base = execute("construct", n // 2, steps, seed, solver_config, start_kind)
        if base.solved:
            sequences = double_gs4(base.sequences)
            verify_candidate(sequences)
            return RunResult(
                strategy,
                n,
                seed,
                sequences,
                0,
                base.steps,
                time.perf_counter() - started,
                base.stats,
                True,
            )

    rng = np.random.default_rng(seed)
    sequences = initial_sequences(n, rng, start_kind)
    solved = search(sequences, Tracker(), rng, steps=steps, config=solver_config)
    verified = False
    if solved.solved:
        verify_candidate(solved.sequences)
        verified = True
    return RunResult(
        strategy,
        n,
        seed,
        solved.sequences,
        solved.energy,
        solved.steps,
        time.perf_counter() - started,
        solved.stats,
        verified,
    )
