"""Tabu parameter tuning."""

import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any

from src.benchmark_stats import wilson_ci
from src.generator import Generator
from src.solver import SolverConfig


def _run_one(args: tuple[int, int, int, SolverConfig]) -> dict[str, Any]:
    n, seed, steps, config = args
    gen = Generator(kind="gs4", n=n)
    r = gen.search(steps=steps, seed=seed, config=config)
    s = r.stats.to_dict() if r.stats else {}
    return {"energy": r.metrics.energy, "elapsed": r.elapsed, "stats": s}


def main() -> None:
    BUDGETS = [200_000]
    N_VAL = 38
    seeds = 200

    cfg = SolverConfig()
    configs = [("S+P+K+Tabu", cfg)]
    tasks = [(N_VAL, s, b, cfg) for b in BUDGETS for s in range(seeds)]
    print(f"Optimization test: n={N_VAL}, {seeds} seeds, {len(tasks)} runs\n")

    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=12) as pool:
        for i, r in enumerate(pool.map(_run_one, tasks), 1):
            results.append(r)
            if i % 80 == 0:
                print(f"  [{i}/{len(tasks)}] {time.perf_counter() - started:.0f}s", flush=True)

    wall = time.perf_counter() - started
    print(f"\n  done in {wall:.0f}s\n")

    print(f"  {'budget':>7}  {'solved':>7}  {'95% CI':>16}  {'mean_e':>8}  {'avg':>8}")
    print(f"  {'-' * 60}")
    for bi, b in enumerate(BUDGETS):
        runs = results[bi * seeds : (bi + 1) * seeds]
        solved = sum(1 for r in runs if r["energy"] == 0)
        avg_t = sum(r["elapsed"] for r in runs) / len(runs)
        mean_e = sum(r["energy"] for r in runs) / len(runs)
        lo, hi = wilson_ci(solved, len(runs))

        print(
            f"  {b:>7}  {solved:>4}/{seeds:<3}  "
            f"[{lo:.4f},{hi:.4f}]  {mean_e:>8.0f}  {avg_t * 1000:>6.0f}ms"
        )


if __name__ == "__main__":
    main()
