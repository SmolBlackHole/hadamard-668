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
    cfg = lambda tenure=5, decay=0.7, noise=0.0: SolverConfig(
        tabu=True, tabu_steps=200, tabu_tenure=tenure, tabu_decay=decay, tabu_noise=noise
    )
    N_VALS = [35, 37, 39, 41, 43]
    n_val, seeds, steps = 0, 100, 200_000

    all_configs = [("default (5,0.7,0.0)", cfg())]

    tasks = []
    for n in N_VALS:
        for c in all_configs:
            for s in range(seeds):
                tasks.append((n, s, steps, c[1]))
    print(f"Odd-n with Tabu: n={N_VALS}, {seeds} seeds, {len(tasks)} runs\n")

    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=12) as pool:
        for i, r in enumerate(pool.map(_run_one, tasks), 1):
            results.append(r)
            if i % 80 == 0:
                print(f"  [{i}/{len(tasks)}] {time.perf_counter() - started:.0f}s", flush=True)

    wall = time.perf_counter() - started
    print(f"\n  done in {wall:.0f}s\n")

    print(f"  {'n':>4}  {'solved':>7}  {'95% CI':>16}  {'mean_e':>8}  {'avg':>8}  {'per_sol':>8}  {'TB_hits':>8}  {'TB_walks':>8}")
    print(f"  {'-'*90}")
    for ni, n in enumerate(N_VALS):
        runs = results[ni * seeds : (ni + 1) * seeds]
        solved = sum(1 for r in runs if r["energy"] == 0)
        avg_t = sum(r["elapsed"] for r in runs) / len(runs)
        mean_e = sum(r["energy"] for r in runs) / len(runs)
        tb_hits = sum(r["stats"].get("tabu_hits", 0) for r in runs)
        tb_walks = sum(r["stats"].get("tabu_walks", 0) for r in runs)
        lo, hi = wilson_ci(solved, len(runs))
        per_sol = avg_t * len(runs) / solved if solved else float("inf")

        print(
            f"  {n:>4}  {solved:>4}/{seeds:<3}  "
            f"[{lo:.4f},{hi:.4f}]  {mean_e:>8.0f}  {avg_t * 1000:>6.0f}ms  "
            f"{per_sol:>6.2f}s  {tb_hits:>8}  {tb_walks:>8}"
        )



if __name__ == "__main__":
    main()
