"""Ablation test: measure each solver component's contribution.

Runs 7 configurations x 3 n-values x 20 seeds = 420 runs.
Outputs a comparison table showing solved rate, energy, and per-component
hit counts + energy saved.
"""

from __future__ import annotations

import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from generator import Generator
from solver import SolverConfig


@dataclass
class AblationConfig:
    name: str
    config: SolverConfig


CONFIGS = [
    AblationConfig("singles-only", SolverConfig(pairs=False, kick=False)),
    AblationConfig("+pairs", SolverConfig(pairs=True, kick=False)),
    AblationConfig("default (S+P+K)", SolverConfig()),
    AblationConfig("all-on", SolverConfig(pairs=True, triples=True, kick=True, restart=True)),
    AblationConfig("+restarts", SolverConfig(pairs=True, kick=True, restart=True)),
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

    for comp in ("singles", "pairs", "triples", "kicks"):
        hits = sum(r["stats"].get(comp, 0) for r in runs)
        e_saved = sum(r["stats"].get(f"e_{comp}", 0) for r in runs)
        mean_e_per_hit = e_saved / hits if hits > 0 else 0
        agg[f"{comp}_hits"] = hits
        agg[f"{comp}_e"] = _fmt_e(e_saved)
        agg[f"{comp}_e_hit"] = f"{mean_e_per_hit:.0f}" if hits > 0 else "-"

    restarts = sum(r["stats"].get("restarts", 0) for r in runs)
    agg["restarts"] = restarts
    agg["single_evals"] = sum(r["stats"].get("single_evals", 0) for r in runs)
    agg["rescue_evals"] = sum(r["stats"].get("rescue_evals", 0) for r in runs)
    agg["kick_evals"] = sum(r["stats"].get("kick_evals", 0) for r in runs)

    return agg


def _fmt_e(e: int) -> str:
    if e == 0:
        return "0"
    if e >= 1_000_000:
        return f"{e / 1_000_000:.1f}M"
    if e >= 1000:
        return f"{e / 1000:.0f}k"
    return str(e)


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
    for ac in CONFIGS:
        print()
        print(f"-- {ac.name} --")
        header = f"{'n':>4}  {'solved':>7}  {'best_e':>6}  {'mean_e':>7}  {'time':>6}  "
        header += f"{'S-hits':>7} {'S-e':>7} {'S/':>4}  "
        header += f"{'P-hits':>7} {'P-e':>7} {'P/':>4}  "
        header += f"{'T-hits':>7} {'T-e':>7} {'T/':>4}  "
        header += f"{'K-hits':>7} {'K-e':>7} {'K/':>4}  "
        header += f"{'R':>5} {'S-eval':>8} {'R-eval':>8} {'K-eval':>8}"
        print(header)
        print("-" * len(header))
        for n in NS:
            a = _aggregate(by_config[ac.name][n])
            line = (
                f"{n:>4}  {a['solved']:>7}  {a['best_e']:>6}  {a['mean_e']:>7}  {a['mean_t']:>6}  "
                f"{a['singles_hits']:>7} {a['singles_e']:>7} {a['singles_e_hit']:>4}  "
                f"{a['pairs_hits']:>7} {a['pairs_e']:>7} {a['pairs_e_hit']:>4}  "
                f"{a['triples_hits']:>7} {a['triples_e']:>7} {a['triples_e_hit']:>4}  "
                f"{a['kicks_hits']:>7} {a['kicks_e']:>7} {a['kicks_e_hit']:>4}  "
                f"{a['restarts']:>5} {a['single_evals']:>8} {a['rescue_evals']:>8} "
                f"{a['kick_evals']:>8}"
            )
            print(line)
        print()


if __name__ == "__main__":
    main()
