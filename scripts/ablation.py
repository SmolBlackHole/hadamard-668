"""Ablation test: measure each solver component's contribution.

Run with ``python -m scripts.ablation`` from the repository root.

Runs the configured solver variants on identical seed ranges.
Outputs a comparison table showing solved rate, energy, and per-component
hit counts + energy saved.
"""

from __future__ import annotations

import argparse
import os
import time
from collections import defaultdict
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any, cast

from src.benchmark_stats import fmt_e, mcnemar, wilson_ci, z_test
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
WORKERS = max(1, int((os.cpu_count() or 2) * 0.8))


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
    parser = argparse.ArgumentParser(
        description="Measure solver component and escape-policy ablations."
    )
    parser.add_argument("--targeted", action="store_true", help="Compare targeted escape policies.")
    parser.add_argument("--n", type=int, default=NS[0], help=f"Sequence length (default: {NS[0]}).")
    parser.add_argument(
        "--seeds", type=int, default=N_SEEDS, help=f"Paired seed count (default: {N_SEEDS})."
    )
    parser.add_argument(
        "--steps", type=int, default=STEPS, help=f"Search steps (default: {STEPS})."
    )
    parser.add_argument(
        "--workers", type=int, default=WORKERS, help=f"Worker count (default: {WORKERS})."
    )
    args = parser.parse_args()
    configs: list[AblationConfig] = (
        [
            AblationConfig("legacy625", SolverConfig(escape_policy="legacy625")),
            AblationConfig("support_lag625", SolverConfig(escape_policy="support_lag625")),
        ]
        if args.targeted
        else CONFIGS
    )
    ns = [args.n] if args.targeted else NS
    workers = args.workers

    # Build task list: one (n, seed, steps, config) per run
    tasks: list[tuple[int, int, int, SolverConfig]] = []
    for ac in configs:
        for n in ns:
            for seed in range(args.seeds):
                tasks.append((n, seed, args.steps, ac.config))

    total = len(tasks)
    print(
        f"Ablation: {len(configs)} configs x {len(ns)} n x {args.seeds} seeds = {total} runs ({workers} workers)\n"
    )
    started = time.perf_counter()

    results: list[dict[str, Any]] = []

    def collect(runs: Iterable[dict[str, Any]]) -> None:
        for i, r in enumerate(runs, 1):
            results.append(r)
            if i % 20 == 0 or i == total:
                print(f"  [{i}/{total}]", flush=True)

    if workers == 1:
        collect(map(_run_one, tasks))
    else:
        with ProcessPoolExecutor(max_workers=min(workers, total)) as pool:
            collect(pool.map(_run_one, tasks))

    elapsed = time.perf_counter() - started
    print(f"\n  done in {elapsed:.0f}s\n")

    # Group by config and n
    by_config: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    for i, r in enumerate(results):
        ac = cast(AblationConfig, configs[i // (len(ns) * args.seeds)])
        by_config[ac.name][r["n"]].append(r)

    # Print table
    by_cfg_n_solved: dict[str, dict[int, tuple[int, int]]] = defaultdict(
        lambda: defaultdict(lambda: (0, 0))
    )  # type: ignore[assignment]
    for ac in configs:
        print()
        print(f"-- {ac.name} --")
        header = f"{'n':>4}  {'solved':>7}  {'95% CI':>15}  {'mean_e':>7}  {'time':>6}  "
        header += f"{'S-hits':>7} {'TB-hit':>7} {'K-hits':>7}"
        print(header)
        print("-" * len(header))
        for n in ns:
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
    if len(configs) > 1:
        base_name = configs[0].name
        print("-- Z-Tests vs", base_name, "--")
        for ac in configs[1:]:
            for n in ns:
                k1, n1 = by_cfg_n_solved[base_name][n]
                k2, n2 = by_cfg_n_solved[ac.name][n]
                z, p = z_test(k1, n1, k2, n2)
                sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""
                print(f"  {ac.name} n={n}: z={z:+.3f}  p={p:.4f}  {sig}")
        print()

    if args.targeted:
        legacy, candidate = configs[0].name, configs[1].name
        print(f"-- Paired McNemar: {candidate} vs {legacy} --")
        for n in ns:
            legacy_runs = {r["seed"]: r["energy"] == 0 for r in by_config[legacy][n]}
            candidate_runs = {r["seed"]: r["energy"] == 0 for r in by_config[candidate][n]}
            a = sum(legacy_runs[s] and candidate_runs[s] for s in legacy_runs)
            b = sum(legacy_runs[s] and not candidate_runs[s] for s in legacy_runs)
            c = sum(not legacy_runs[s] and candidate_runs[s] for s in legacy_runs)
            d = sum(not legacy_runs[s] and not candidate_runs[s] for s in legacy_runs)
            chi2, p = mcnemar(a, b, c, d)
            print(f"  n={n}: a/b/c/d={a}/{b}/{c}/{d}  chi2={chi2:.3f}  p={p:.4f}")


if __name__ == "__main__":
    main()
