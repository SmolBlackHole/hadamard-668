"""Hadamard search entry point."""
from __future__ import annotations
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from output import save
from strategies.registry import DEFAULT, GPU, parse as parse_strategy

import numpy as np
from concurrent.futures import ProcessPoolExecutor
import time
import datetime
import argparse


def select_best_run(
    results: list[tuple[int, np.ndarray, dict[str, int], float]],
) -> tuple[int, np.ndarray, dict[str, int], float]:
    if not results:
        raise ValueError("at least one run result is required")
    return min(results, key=lambda r: r[2]["energy"])


def derive_seeds(seed: int, runs: int) -> list[int]:
    return [seed + offset for offset in range(runs)]


def worker_count(specification: str, runs: int, workers: int) -> int:
    names = {p.rsplit(":", 1)[0] if ":" in p else p
             for p in specification.split(",")}
    if names & GPU:
        return 1
    return min(runs, workers)


def _execute_run(spec: str, steps: int, seed: int, order: int,
                 time_budget: float = 0.0):
    strategy = parse_strategy(spec, order)
    started = time.perf_counter()
    matrix, metrics = None, None
    if time_budget > 0:
        t0 = time.perf_counter()
        strategy.search(steps=50, seed=seed)
        cal = max(time.perf_counter() - t0, 0.001)
        chunk = max(50, int(0.5 / cal * 50))
        for _ in range(int(time_budget / (chunk * cal / 50)) + 1):
            matrix, metrics, _ = strategy.search(steps=chunk, seed=seed)
            if metrics["energy"] == 0:
                break
    if matrix is None:
        matrix, metrics, _ = strategy.search(steps, seed)
    wall = time.perf_counter() - started
    return seed, matrix, metrics, wall


def main() -> None:
    parser = argparse.ArgumentParser(description="Hadamard search")
    parser.add_argument("--strategy", default=DEFAULT,
                        help="strategy name or s1:steps,s2:steps")
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--order", type=int, default=668)
    parser.add_argument("--time", type=float, default=0)
    args = parser.parse_args()

    try:
        strategy = parse_strategy(args.strategy, args.order)
        seeds = derive_seeds(args.seed, args.runs)
        workers = worker_count(args.strategy, args.runs, args.workers)
    except ValueError as e:
        parser.error(str(e))

    if workers == 1:
        results = [_execute_run(args.strategy, args.steps, s, args.order, args.time)
                   for s in seeds]
    else:
        with ProcessPoolExecutor(max_workers=workers) as e:
            results = list(e.map(
                _execute_run,
                [args.strategy] * len(seeds),
                [args.steps] * len(seeds),
                seeds,
                [args.order] * len(seeds),
                [args.time] * len(seeds),
            ))

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    for seed, matrix, metrics, wall in results:
        out = Path(args.runs_dir) / f"{strategy.name}_{seed}_{timestamp}"
        save(matrix, metrics, out,
             strategy=args.strategy, seed=seed, steps=args.steps,
             wall=wall, order=matrix.shape[0], construction=strategy.construction)

    best = min(results, key=lambda r: r[2]["energy"])
    print("\nRun summary")
    for seed, _, m, wall in results:
        star = " *" if seed == best[0] else ""
        print(f"  seed={seed} energy={m['energy']} elapsed={wall:.1f}s{star}")
    if best[2]["energy"] == 0:
        print("*** HADAMARD! ***")
    print(
        f"best seed={best[0]} energy={best[2]['energy']} elapsed={best[3]:.1f}s")


if __name__ == "__main__":
    main()
