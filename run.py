"""Hadamard-668 search entry point."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from strategies.turyn_steepest import TurynSteepestSearch
from strategies.pocs import TurynPocsSearch
from strategies.spectral import SpectralSearch
from strategies.repair import RepairSearch
from strategies.montecarlo import MonteCarloSearch
from strategies.ising import IsingSearch
from strategies.genetic import GeneticSearch
from strategies.diffset import DiffsetSearch
from strategies.circulant import TurynGreedySearch
from strategies.base import Pipeline, SearchStrategy
from strategies.annealing import TurynAnnealingSearch
from gpu import check_orthogonality
from output import save_run

import argparse
from concurrent.futures import ProcessPoolExecutor
import datetime
import time

import numpy as np

ALL: dict[str, SearchStrategy] = {
    "turyn_greedy": TurynGreedySearch(),
    "turyn_annealing": TurynAnnealingSearch(),
    "turyn_pocs": TurynPocsSearch(),
    "turyn_steepest": TurynSteepestSearch(),
    "repair": RepairSearch(),
    "spectral": SpectralSearch(),
    "ising": IsingSearch(),
    "diffset": DiffsetSearch(),
    "genetic": GeneticSearch(),
    "montecarlo": MonteCarloSearch(),
}

GPU_INTENSIVE_STRATEGIES = frozenset({"montecarlo"})


def parse_strategy(specification: str) -> SearchStrategy:
    """Parse a single strategy or a comma-separated pipeline.

    Syntax: s1:N,s2:M    → Pipeline([(s1, N), (s2, M)])
    """
    if ":" not in specification:
        if specification not in ALL:
            raise ValueError(f"unknown strategy: {specification}")
        return ALL[specification]

    # Pipeline
    stages = []
    for part in specification.split(","):
        name, steps_text = part.rsplit(":", 1)
        if name not in ALL:
            raise ValueError(f"unknown strategy in pipeline: {name}")
        stages.append((ALL[name], int(steps_text)))
    return Pipeline(stages)


def method_family(strategy: SearchStrategy) -> str:
    if isinstance(strategy, Pipeline):
        return "other"
    if strategy.name.startswith("turyn_"):
        return "turyn"
    if strategy.name == "repair":
        return "local_search"
    return "other"


def derive_seeds(seed: int, runs: int) -> list[int]:
    """Return a deterministic seed for every requested run."""
    if runs < 1:
        raise ValueError("runs must be positive")
    return [seed + offset for offset in range(runs)]


def _strategy_names(specification: str) -> set[str]:
    return {
        part.rsplit(":", 1)[0] if ":" in part else part
        for part in specification.split(",")
    }


def worker_count(specification: str, runs: int, workers: int) -> int:
    """Limit parallelism to one process for sustained GPU workloads."""
    if workers < 1:
        raise ValueError("workers must be positive")
    if _strategy_names(specification) & GPU_INTENSIVE_STRATEGIES:
        return 1
    return min(runs, workers)


def _execute_run(
    specification: str,
    steps: int,
    seed: int,
) -> tuple[int, np.ndarray, dict[str, int], float]:
    strategy = parse_strategy(specification)
    started = time.perf_counter()
    matrix, _, _ = strategy.search(steps, seed)
    wall_seconds = time.perf_counter() - started
    metrics = check_orthogonality(matrix)
    return seed, matrix, metrics, wall_seconds


def select_best_run(
    results: list[tuple[int, np.ndarray, dict[str, int], float]],
) -> tuple[int, np.ndarray, dict[str, int], float]:
    """Select the run with the smallest exact Gram energy."""
    if not results:
        raise ValueError("at least one run result is required")
    return min(results, key=lambda result: result[2]["energy"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Hadamard-668 search")
    parser.add_argument("--strategy", default="circulant",
                        help="turyn_greedy | turyn_annealing | ... | s1:N,s2:M")
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--runs-dir", default="runs")
    args = parser.parse_args()

    try:
        strategy = parse_strategy(args.strategy)
        seeds = derive_seeds(args.seed, args.runs)
        workers = worker_count(args.strategy, args.runs, args.workers)
    except ValueError as error:
        parser.error(str(error))
    if workers != min(args.runs, args.workers):
        print("GPU-intensive strategy detected; using one worker.")

    if workers == 1:
        results = [_execute_run(args.strategy, args.steps, seed)
                   for seed in seeds]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            results = list(executor.map(
                _execute_run,
                [args.strategy] * len(seeds),
                [args.steps] * len(seeds),
                seeds,
            ))

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    for seed, matrix, metrics, wall_seconds in results:
        output = Path(args.runs_dir) / \
            f"{strategy.name.replace('->', '_')}_{seed}_{timestamp}"
        save_run(
            matrix,
            metrics,
            output,
            method_family=method_family(strategy),
            search_scope=args.strategy,
            seed=seed,
            steps=args.steps,
            wall_seconds=wall_seconds,
            hardware_summary="numpy-or-cupy",
        )

    best_seed, _, best_metrics, best_seconds = select_best_run(results)
    print("\nRun summary")
    for seed, _, metrics, wall_seconds in results:
        marker = " *" if seed == best_seed else ""
        print(
            f"  seed={seed} energy={metrics['energy']} "
            f"elapsed={wall_seconds:.1f}s{marker}")
    if best_metrics["energy"] == 0:
        print("*** HADAMARD! ***")
    print(
        f"best seed={best_seed} energy={best_metrics['energy']} "
        f"elapsed={best_seconds:.1f}s")


if __name__ == "__main__":
    main()
