"""Ablation test: measure each solver component's contribution.

Run with ``python -m scripts.ablation`` from the repository root.

Runs the configured solver variants on identical seed ranges.
Outputs a comparison table showing solved rate, energy, and per-component
hit counts + energy saved.
"""

from __future__ import annotations

import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any

from src.benchmark_stats import fmt_e, wilson_ci, z_test
from src.generator import Generator
from src.solver import SolverConfig


@dataclass
class AblationConfig:
    name: str
    config: SolverConfig


CONFIGS = [
    AblationConfig("singles-only", SolverConfig(kick=False, tabu=False)),
    AblationConfig("+tabu", SolverConfig(kick=False, tabu=True)),
    AblationConfig("default (S+TB+K)", SolverConfig()),
]

NS = [32]
N_SEEDS = 30
STEPS = 200_000
WORKERS = max(1, int((__import__("os").cpu_count() or 2) * 0.8))


def _run_one(args: tuple[int, int, int, SolverConfig]) -> dict[str, Any]:
    """Run a single search.  Pickle-friendly for ProcessPoolExecutor."""
    n, seed, steps, config = args
    gen = Generator(kind="gs4", n=n)
    started = time.perf_counter()
    result = gen.search(steps=steps, seed=seed, config=config)
    elapsed = time.perf_counter() - started
    return {
        "n": n,
        "seed": seed,
        "energy": result.metrics.energy,
        "solver_e": result.solver_e,
        "elapsed": elapsed,
        "stats": result.stats.to_dict() if result.stats else {},
    }


def _aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    solved = sum(1 for r in runs if r["energy"] == 0)
    total = len(runs)
    best_e = min(r["energy"] for r in runs)
    mean_e = sum(r["energy"] for r in runs) / total
    mean_t = sum(r["elapsed"] for r in runs) / total

    agg: dict[str, Any] = {
        "solved": f"{solved}/{total}",
        "best_e": best_e,
        "mean_e": f"{mean_e:.0f}",
        "mean_t": f"{mean_t:.1f}s",
    }

    for comp in ("singles", "kicks"):
        hits = sum(r["stats"].get(comp, 0) for r in runs)
        e_saved = sum(r["stats"].get(f"e_{comp}", 0) for r in runs)
        mean_e_per_hit = e_saved / hits if hits > 0 else 0
        agg[f"{comp}_hits"] = hits
        agg[f"{comp}_e"] = fmt_e(e_saved)
        agg[f"{comp}_e_hit"] = f"{mean_e_per_hit:.0f}" if hits > 0 else "-"

    agg["single_evals"] = sum(r["stats"].get("single_evals", 0) for r in runs)
    agg["kick_evals"] = sum(r["stats"].get("kick_evals", 0) for r in runs)
    agg["tabu_hits"] = sum(r["stats"].get("tabu_hits", 0) for r in runs)
    agg["tabu_evals"] = sum(r["stats"].get("tabu_evals", 0) for r in runs)

    return agg


def main() -> None:
    # Build task list: one (n, seed, steps, config) per run
    tasks: list[tuple[int, int, int, SolverConfig]] = []
    for ac in CONFIGS:
        for n in NS:
            for seed in range(N_SEEDS):
                tasks.append((n, seed, STEPS, ac.config))

    total = len(tasks)
    print(
        f"Ablation: {len(CONFIGS)} configs x {len(NS)} n x {N_SEEDS} seeds = {total} runs ({WORKERS} workers)\n"
    )
    started = time.perf_counter()

    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(max_workers=min(WORKERS, total)) as pool:
        for i, r in enumerate(pool.map(_run_one, tasks), 1):
            results.append(r)
            if i % 20 == 0 or i == total:
                print(f"  [{i}/{total}]", flush=True)

    elapsed = time.perf_counter() - started
    print(f"\n  done in {elapsed:.0f}s\n")

    # Group by config and n
    by_config: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for i, r in enumerate(results):
        ac = CONFIGS[i // (len(NS) * N_SEEDS)]
        by_config[ac.name][r["n"]].append(r)

    # Print table
    by_cfg_n_solved: dict[str, dict[int, tuple[int, int]]] = defaultdict(
        lambda: defaultdict(lambda: (0, 0))
    )  # type: ignore[assignment]
    for ac in CONFIGS:
        print()
        print(f"-- {ac.name} --")
        header = f"{'n':>4}  {'solved':>7}  {'95% CI':>15}  {'mean_e':>7}  {'time':>6}  "
        header += f"{'S-hits':>7} {'TB-hit':>7} {'K-hits':>7}"
        print(header)
        print("-" * len(header))
        for n in NS:
            a = _aggregate(by_config[ac.name][n])
            solved, total = a["solved"].split("/")
            k, tot = int(solved), int(total)
            by_cfg_n_solved[ac.name][n] = (k, tot)
            lo, hi = wilson_ci(k, tot)
            line = (
                f"{n:>4}  {a['solved']:>7}  [{lo:.3f},{hi:.3f}]  "
                f"{a['mean_e']:>7}  {a['mean_t']:>6}  "
                f"{a['singles_hits']:>7} {a['tabu_hits']:>7} {a['kicks_hits']:>7}"
            )
            print(line)
        print()

    # Z-tests against baseline (first config)
    if len(CONFIGS) > 1:
        base_name = CONFIGS[0].name
        print("-- Z-Tests vs", base_name, "--")
        for ac in CONFIGS[1:]:
            for n in NS:
                k1, n1 = by_cfg_n_solved[base_name][n]
                k2, n2 = by_cfg_n_solved[ac.name][n]
                z, p = z_test(k1, n1, k2, n2)
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                print(f"  {ac.name} n={n}: z={z:+.3f}  p={p:.4f}  {sig}")
        print()


if __name__ == "__main__":
    main()
