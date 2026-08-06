"""Budget scaling experiment: test solve rate vs step budget."""

from __future__ import annotations

import math
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from generator import Generator


def _run_one(args: tuple[int, int, int]) -> dict[str, object]:
    n, seed, steps = args
    gen = Generator(kind="gs4", n=n)
    r = gen.search(steps=steps, seed=seed)
    s = r.stats.to_dict() if r.stats else {}
    return {
        "n": n,
        "seed": seed,
        "steps": steps,
        "energy": r.metrics.energy,
        "solver_e": r.solver_e,
        "elapsed": r.elapsed,
        "single_evals": s.get("single_evals", 0),
        "rescue_evals": s.get("rescue_evals", 0),
        "kick_evals": s.get("kick_evals", 0),
    }


def _wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return center - half, center + half


def main() -> None:
    n_vals = [34, 36, 38]
    seeds_count = 50
    budgets = [50_000, 100_000, 200_000, 400_000, 800_000, 1_600_000]
    workers = max(1, int((__import__("os").cpu_count() or 2) * 0.8))

    tasks: list[tuple[int, int, int]] = []
    for n_val in n_vals:
        for b in budgets:
            for s in range(seeds_count):
                tasks.append((n_val, s, b))

    print(
        f"Budget scaling: n={n_vals}, {len(budgets)} budgets x {seeds_count} seeds = {len(tasks)} runs ({workers} workers)"
    )
    print()

    started = time.perf_counter()
    results: list[dict[str, object]] = []
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for i, r in enumerate(pool.map(_run_one, tasks), 1):
            results.append(r)
            if i % 60 == 0:
                elapsed = time.perf_counter() - started
                eta = (elapsed / i) * (len(tasks) - i)
                print(f"  [{i}/{len(tasks)}] {elapsed:.0f}s elapsed, ETA {eta:.0f}s", flush=True)

    total_wall = time.perf_counter() - started

    by_key: dict[tuple[int, int], list[dict[str, object]]] = {}
    for r, (n_v, _, b) in zip(results, tasks):
        by_key.setdefault((n_v, b), []).append(r)

    print(f"\n  done in {total_wall:.0f}s\n")
    for n_val in n_vals:
        print(f"-- n={n_val} --")
        for b in budgets:
            rs = by_key[(n_val, b)]
            solved = sum(1 for r in rs if r["energy"] == 0)
            avg_t = sum(float(r["elapsed"]) for r in rs) / len(rs)
            lo, hi = _wilson(solved, len(rs))

            s_avg = sum(int(r["single_evals"]) for r in rs) / len(rs)
            p_avg = sum(int(r["rescue_evals"]) for r in rs) / len(rs)
            k_avg = sum(int(r["kick_evals"]) for r in rs) / len(rs)

            print(
                f"  steps={b:>8}  {solved:>3}/{seeds_count} solved  "
                f"ci=[{lo:.3f}, {hi:.3f}]  "
                f"avg {avg_t * 1000:.0f}ms/run  "
                f"evals={s_avg + p_avg + k_avg:.0f} (S={s_avg:.0f} P={p_avg:.0f} K={k_avg:.0f})"
            )
        print()


if __name__ == "__main__":
    main()
