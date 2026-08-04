"""Hadamard search entry point."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import argparse
import datetime
import time
from concurrent.futures import ProcessPoolExecutor

from output import save
from strategies.base import RunResult
from strategies.registry import DEFAULT, GPU
from strategies.registry import parse as parse_strategy


def select_best_run(results: list[RunResult]) -> RunResult:
    if not results:
        raise ValueError("at least one run result is required")
    return min(results, key=lambda r: r.energy)


def derive_seeds(seed: int, runs: int) -> list[int]:
    return [seed + offset for offset in range(runs)]


def worker_count(spec: str, runs: int, workers: int) -> int:
    names = {p.rsplit(":", 1)[0] if ":" in p else p for p in spec.split(",")}
    return 1 if names & GPU else min(runs, workers)


def _execute_run(
    spec: str, steps: int, seed: int, order: int, time_budget: float = 0.0
) -> RunResult:
    strategy = parse_strategy(spec, order)
    started = time.perf_counter()
    matrix = None
    metrics: dict[str, int] = {"energy": 2**63, "orthogonal_pairs": 0, "max_abs_correlation": 0}
    if time_budget > 0:
        t0 = time.perf_counter()
        strategy.search(steps=50, seed=seed)
        cal = max(time.perf_counter() - t0, 0.001)
        chunk = max(50, int(0.5 / cal * 50))
        for _ in range(int(time_budget / (chunk * cal / 50)) + 1):
            matrix, metrics, _, _sequences = strategy.search(steps=chunk, seed=seed)
            if metrics["energy"] == 0:
                break
    if matrix is None:
        matrix, metrics, _, _sequences = strategy.search(steps, seed)
    wall = time.perf_counter() - started
    return RunResult(
        seed=seed,
        matrix=matrix,
        energy=metrics["energy"],
        orthogonal_pairs=metrics["orthogonal_pairs"],
        max_off_diagonal=metrics["max_abs_correlation"],
        wall=wall,
        hamming=strategy.hamming(),
    )


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
        workers = worker_count(args.strategy, args.runs, args.workers)
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
            {
                "energy": r.energy,
                "orthogonal_pairs": r.orthogonal_pairs,
                "max_abs_correlation": r.max_off_diagonal,
            },
            out,
            strategy=args.strategy,
            seed=r.seed,
            steps=args.steps,
            wall=r.wall,
            order=r.matrix.shape[0],
            construction=strategy.construction,
            hamming=r.hamming,
        )

    best = select_best_run(results)
    print("\nRun summary")
    for r in results:
        star = " *" if r.seed == best.seed else ""
        extra = f" hamming={r.hamming}" if r.hamming is not None else ""
        print(f"  seed={r.seed} energy={r.energy} elapsed={r.wall:.1f}s{extra}{star}")
    if best.energy == 0:
        print("*** HADAMARD! ***")
    print(f"best seed={best.seed} energy={best.energy} elapsed={best.wall:.1f}s")


if __name__ == "__main__":
    main()
