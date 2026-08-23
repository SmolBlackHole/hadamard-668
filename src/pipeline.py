"""End-to-end execution from sequence construction through verification."""

from __future__ import annotations

import time
from dataclasses import asdict

import numpy as np

from .constructions import double_gs4
from .generator import RANDOM_START, StartConstruction, exact_sequences
from .models import RunResult, SearchStats
from .solver import SolverConfig, search
from .tracker import Tracker
from .verify import verify_candidate


def execute(
    strategy: str,
    n: int,
    candidate_budget: int,
    seed: int,
    solver_config: SolverConfig,
    start: StartConstruction = RANDOM_START,
) -> RunResult:
    """Execute one exact-construction or heuristic-search run.

    Exact and solved heuristic candidates pass the independent GS4 residual
    and full-matrix audit before the returned result is marked as verified.

    Args:
        strategy: One of ``gs4``, ``paley-ng``, or ``construct``.
        n: Positive length of each of the four sequences.
        candidate_budget: Maximum logical evaluations for each invoked search.
        seed: Seed for the NumPy random generator used by heuristic search.
        solver_config: Search-phase configuration persisted with the result.
        start: Construction for the heuristic initial state.

    Returns:
        The final state, verification status, timing in seconds, and search
        accounting for the requested run.

    Raises:
        ValueError: If an input or requested exact construction is unsupported.
        InvalidMatrix: If an exact or zero-energy candidate fails verification.

    Note:
        A failed recursive ``construct`` base attempt receives its own full
        budget. Its work is not included in the outer result's candidate count
        before the fallback search starts.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if candidate_budget < 1:
        raise ValueError("candidate budget must be positive")
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
            start.name,
            candidate_budget,
            asdict(solver_config),
        )

    if strategy == "construct" and n % 2 == 0:
        base = execute("construct", n // 2, candidate_budget, seed, solver_config, start)
        if base.solved:
            sequences = double_gs4(base.sequences)
            verify_candidate(sequences)
            return RunResult(
                strategy,
                n,
                seed,
                sequences,
                0,
                base.candidate_evals,
                time.perf_counter() - started,
                base.stats,
                True,
                start.name,
                candidate_budget,
                asdict(solver_config),
            )

    rng = np.random.default_rng(seed)
    sequences = start.build(n, rng)
    solved = search(
        sequences,
        Tracker(),
        rng,
        candidate_budget=candidate_budget,
        config=solver_config,
    )
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
        solved.candidate_evals,
        time.perf_counter() - started,
        solved.stats,
        verified,
        start.name,
        candidate_budget,
        asdict(solver_config),
    )
