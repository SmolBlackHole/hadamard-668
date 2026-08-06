"""Hadamard search entry point."""

from __future__ import annotations

import argparse
import datetime
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from generator import Generator, Result
from output import save

sys.path.insert(0, str(Path(__file__).parent / "src"))


def derive_seeds(seed: int, runs: int) -> list[int]:
    return [seed + offset for offset in range(runs)]


def select_best_run(results: list) -> Result:
    if not results:
        raise ValueError("at least one run result is required")
    return min(results, key=lambda r: r.metrics.energy)


def _execute_run(
    strategy_name: str, steps: int, seed: int, order: int, time_budget: float = 0.0
) -> Result:
    gen = Generator.from_cli(strategy_name, order)
    started = time.perf_counter()
    result = None

    if time_budget > 0:
        t0 = time.perf_counter()
        gen.search(steps=50, seed=seed)
        cal = max(time.perf_counter() - t0, 0.001)
        chunk = max(50, int(0.5 / cal * 50))
        for _ in range(int(time_budget / (chunk * cal / 50)) + 1):
            result = gen.search(steps=chunk, seed=seed)
            if result.metrics.energy == 0:
                break

    if result is None:
        result = gen.search(steps=steps, seed=seed)

    result.elapsed = time.perf_counter() - started
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Hadamard search")
    parser.add_argument("--strategy", default="gs4")
    parser.add_argument("--steps", type=int, default=200_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--runs-dir", default="runs")
    parser.add_argument("--order", type=int, default=668)
    parser.add_argument("--time", type=float, default=0)
    args = parser.parse_args()

    try:
        gen = Generator.from_cli(args.strategy, args.order)
        seeds = derive_seeds(args.seed, args.runs)
        workers = min(args.runs, args.workers)
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
        out = Path(args.runs_dir) / f"{gen.name}_{r.seed}_{timestamp}"
        save(
            r.matrix,
            r.metrics,
            out,
            strategy=gen.name,
            seed=r.seed,
            steps=args.steps,
            elapsed=r.elapsed,
            order=r.matrix.shape[0],
            iterations=r.iterations,
        )

    best = select_best_run(results)
    print("\nRun summary")
    for r in results:
        star = " *" if r.seed == best.seed else ""
        print(f"  seed={r.seed} energy={r.metrics.energy} elapsed={r.elapsed:.1f}s{star}")
    if best.metrics.energy == 0:
        print("*** HADAMARD! ***")
    print(f"best seed={best.seed} energy={best.metrics.energy} elapsed={best.elapsed:.1f}s")


if __name__ == "__main__":
    main()
