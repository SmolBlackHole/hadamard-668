"""Hadamard-668 search entry point."""
from __future__ import annotations

import argparse
import datetime
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from output import save_run
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Hadamard-668 search")
    parser.add_argument("--strategy", default="circulant",
                        help="circulant | hybrid | annealing | ... | s1:N,s2:M")
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs-dir", default="runs")
    args = parser.parse_args()

    strategy = parse_strategy(args.strategy)
    started = time.perf_counter()
    matrix, metrics, _elapsed = strategy.search(args.steps, args.seed)
    wall_seconds = time.perf_counter() - started
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output = Path(args.runs_dir) / \
        f"{strategy.name.replace('->', '_')}_{args.seed}_{timestamp}"
    output.mkdir(parents=True, exist_ok=True)
    save_run(
        matrix,
        metrics,
        output,
        method_family=method_family(strategy),
        search_scope=args.strategy,
        seed=args.seed,
        steps=args.steps,
        wall_seconds=wall_seconds,
        hardware_summary="numpy-or-cupy",
    )
    if metrics["energy"] == 0:
        print("*** HADAMARD! ***")
    print(
        f"seed={args.seed} energy={metrics['energy']} elapsed={wall_seconds:.0f}s")


if __name__ == "__main__":
    main()
