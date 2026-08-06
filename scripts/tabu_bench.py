"""Quick comparison: S+P+K vs S+P+K+Tabu."""

import math
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any

from src.generator import Generator
from src.solver import SolverConfig


def _run_one(args: tuple[int, int, int, SolverConfig]) -> dict[str, Any]:
    n, seed, steps, config = args
    gen = Generator(kind="gs4", n=n)
    r = gen.search(steps=steps, seed=seed, config=config)
    s = r.stats.to_dict() if r.stats else {}
    return {
        "energy": r.metrics.energy,
        "elapsed": r.elapsed,
        "stats": s,
    }


def main() -> None:
    configs = [
        ("S+P+K", SolverConfig()),
        ("S+P+K+Tabu", SolverConfig(tabu=True)),
    ]
    n_val, seeds, steps = 36, 100, 200_000

    tasks = [(n_val, s, steps, c[1]) for c in configs for s in range(seeds)]
    print(f"Tabu test: n={n_val}, {seeds} seeds, {len(tasks)} runs\n")

    started = time.perf_counter()
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=12) as pool:
        for i, r in enumerate(pool.map(_run_one, tasks), 1):
            results.append(r)
            if i % 40 == 0:
                print(f"  [{i}/{len(tasks)}] {time.perf_counter() - started:.0f}s", flush=True)

    wall = time.perf_counter() - started
    print(f"\n  done in {wall:.0f}s\n")

    for ci, (name, _) in enumerate(configs):
        runs = [r for i, r in enumerate(results) if i // seeds == ci]
        solved = sum(1 for r in runs if r["energy"] == 0)
        avg_t = sum(r["elapsed"] for r in runs) / len(runs)
        mean_e = sum(r["energy"] for r in runs) / len(runs)

        tb_hits = sum(r["stats"].get("tabu_hits", 0) for r in runs)
        tb_walks = sum(r["stats"].get("tabu_walks", 0) for r in runs)
        p_hits = sum(r["stats"].get("pairs", 0) for r in runs)

        k, n = solved, len(runs)
        p = k / n
        z = 1.96
        den = 1 + z * z / n
        center = (p + z * z / (2 * n)) / den
        half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den

        print(
            f"  {name:>15}: {solved}/{seeds} solved  "
            f"ci=[{max(0,center-half):.4f},{min(1,center+half):.4f}]  "
            f"mean_e={mean_e:.0f}  avg {avg_t * 1000:.0f}ms  "
            f"P={p_hits}  TB={tb_hits}/{tb_walks}"
        )

    # Z-test
    k1 = sum(1 for i in range(seeds) if results[i]["energy"] == 0)
    k2 = sum(1 for i in range(seeds, 2 * seeds) if results[i]["energy"] == 0)
    n1 = n2 = seeds
    p1, p2 = k1 / n1, k2 / n2
    p_pool = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    z_val = (p2 - p1) / se if se > 0 else 0.0
    p_val = 2 * (1 - 0.5 * (1 + math.erf(abs(z_val) / math.sqrt(2))))
    sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
    print(f"\n  Z-Test (Tabu - Baseline): z={z_val:+.3f}  p={p_val:.4f}  {sig}")


if __name__ == "__main__":
    main()
