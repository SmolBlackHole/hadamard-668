"""Hadamard-668 search entry point."""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
import datetime
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent / "src"))

from output import save_run
from gpu import check_orthogonality
from strategies.annealing import AnnealingSearch
from strategies.backtrack import BacktrackSearch
from strategies.baumert import BaumertHallSearch
from strategies.base import Pipeline, SearchStrategy
from strategies.ca import CASearch
from strategies.circulant import CirculantSearch
from strategies.diffset import DiffsetSearch
from strategies.direct import DirectSearch
from strategies.genetic import GeneticSearch
from strategies.gold import GoldSearch
from strategies.ising import IsingSearch
from strategies.montecarlo import MonteCarloSearch
from strategies.repair import RepairSearch
from strategies.rowwise import RowwiseSearch
from strategies.sat import SatSearch
from strategies.spectral import SpectralSearch
from strategies.walsh import WalshSearch


ALL: dict[str, SearchStrategy] = {
    "circulant": CirculantSearch(),
    "hybrid": CirculantSearch(constructions="all"),
    "annealing": AnnealingSearch(),
    "annealing_w": AnnealingSearch(constructions="williamson"),
    "repair": RepairSearch(),
    "direct": DirectSearch(),
    "rowwise": RowwiseSearch(),
    "spectral": SpectralSearch(),
    "walsh": WalshSearch(),
    "gold": GoldSearch(),
    "sat": SatSearch(),
    "ising": IsingSearch(),
    "diffset": DiffsetSearch(),
    "genetic": GeneticSearch(),
    "montecarlo": MonteCarloSearch(),
    "cellular": CASearch(),
    "ca_spectral": CASearch(mode="spectral", kernel_size=5),
    "baumert": BaumertHallSearch(),
}

GPU_INTENSIVE_STRATEGIES = frozenset({"montecarlo"})


def parse_strategy(specification: str) -> SearchStrategy:
    """Parse a single strategy, pipeline, or backtrack-wrapped strategy.

    Syntax: s1:N,s2:M    → Pipeline([(s1, N), (s2, M)])
            bt:circulant → BacktrackSearch(CirculantSearch())
    """
    # Backtrack wrapper
    if specification.startswith("bt:"):
        inner_spec = specification[3:]
        inner = parse_strategy(inner_spec)
        return BacktrackSearch(inner)

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
        return "hybrid"
    if strategy.name in {"circulant", "hybrid", "annealing", "annealing_w"}:
        return "williamson_propus"
    if strategy.name in {"direct", "repair"}:
        return "local_search"
    return "other"


def derive_seeds(seed: int, runs: int) -> list[int]:
    """Return a deterministic seed for every requested run."""
    if runs < 1:
        raise ValueError("runs must be positive")
    return [seed + offset for offset in range(runs)]


def _strategy_names(specification: str) -> set[str]:
    value = specification.removeprefix("bt:")
    return {
        part.rsplit(":", 1)[0] if ":" in part else part
        for part in value.split(",")
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
                        help="circulant | hybrid | annealing | ... | s1:N,s2:M")
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
        results = [_execute_run(args.strategy, args.steps, seed) for seed in seeds]
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
