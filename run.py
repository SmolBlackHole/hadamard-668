# ruff: noqa: I001

"""Hadamard search entry point."""

from __future__ import annotations
from strategies.registry import DEFAULT, parse as parse_strategy
from strategies.base import Result
from output import save
from concurrent.futures import ProcessPoolExecutor
import time
import datetime
import argparse

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))


def _gpu_available() -> bool:
    try:
        import cupy

        cupy.array([1.0])
        return True
    except Exception:
        return False


def derive_seeds(seed: int, runs: int) -> list[int]:
    return [seed + offset for offset in range(runs)]


def worker_count(strategy, runs: int, workers: int) -> int:
    if getattr(strategy, "gpu_exclusive", False) and _gpu_available():
        return 1
    return min(runs, workers)


def select_best_run(results: list[Result]) -> Result:
    if not results:
        raise ValueError("at least one run result is required")
    return min(results, key=lambda r: r.metrics.energy)


def _execute_run(spec: str, steps: int, seed: int, order: int, time_budget: float = 0.0) -> Result:
    strategy = parse_strategy(spec, order)
    started = time.perf_counter()
    result = None

    if time_budget > 0:
        t0 = time.perf_counter()
        strategy.search(steps=50, seed=seed)
        cal = max(time.perf_counter() - t0, 0.001)
        chunk = max(50, int(0.5 / cal * 50))
        for _ in range(int(time_budget / (chunk * cal / 50)) + 1):
            result = strategy.search(steps=chunk, seed=seed)
            if result.metrics.energy == 0:
                break

    if result is None:
        result = strategy.search(steps=steps, seed=seed)

    wall = time.perf_counter() - started
    result.elapsed = wall
    result.hamming = strategy.hamming()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Hadamard search")
    parser.add_argument("--strategy", default=DEFAULT)
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
        workers = worker_count(strategy, args.runs, args.workers)
    except ValueError as e:
        parser.error(str(e))

    if workers == 1:
        results = []
        for s in seeds:
            try:
                results.append(_execute_run(args.strategy, args.steps, s, args.order, args.time))
            except Exception as exc:
                print(f"  run seed={s} FAILED: {exc}")
    else:
        with ProcessPoolExecutor(max_workers=workers) as e:
            results = list(
                e.map(
                    _execute_run,
                    [args.strategy] * len(seeds),
                    [args.steps] * len(seeds),
                    seeds,
                    [args.order] * len(seeds),
                    [args.time] * len(seeds),
                )
            )

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    for r in results:
        out = Path(args.runs_dir) / f"{strategy.name.replace('->', '_')}_{r.seed}_{timestamp}"
        save(
            r.matrix,
            r.metrics,
            out,
            strategy=args.strategy,
            seed=r.seed,
            steps=args.steps,
            wall=r.elapsed,
            order=r.matrix.shape[0],
            construction=strategy.construction,
            hamming=r.hamming,
        )

    best = select_best_run(results)
    print("\nRun summary")
    for r in results:
        star = " *" if r.seed == best.seed else ""
        extra = f" hamming={r.hamming}" if r.hamming is not None else ""
        print(f"  seed={r.seed} energy={r.metrics.energy} elapsed={r.elapsed:.1f}s{extra}{star}")
    if best.metrics.energy == 0:
        print("*** HADAMARD! ***")
    print(f"best seed={best.seed} energy={best.metrics.energy} elapsed={best.elapsed:.1f}s")


if __name__ == "__main__":
    main()
