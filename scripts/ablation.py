"""Ablation test: measure each solver component's contribution.

Run with ``python -m scripts.ablation`` from the repository root.

Runs the configured solver variants on identical seed ranges.
Outputs a comparison table showing solved rate, energy, and per-component
hit counts + energy saved.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter, defaultdict
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from src.pipeline import execute
from src.solver import SolverConfig


@dataclass(frozen=True)
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


def _run_one(args: tuple[str, int, int, int, str, SolverConfig]) -> dict[str, Any]:
    """Run a single search.  Pickle-friendly for ProcessPoolExecutor."""
    name, n, seed, steps, start_kind, config = args
    started = time.perf_counter()
    result = execute("gs4", n, steps, seed, config, start_kind)
    elapsed = time.perf_counter() - started
    return {
        "config_name": name,
        "config": asdict(config),
        "start_kind": start_kind,
        "n": n,
        "seed": seed,
        "energy": result.energy,
        "elapsed": elapsed,
        "stats": result.stats.to_dict(),
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
        agg[f"{comp}_e"] = e_saved
        agg[f"{comp}_e_hit"] = f"{mean_e_per_hit:.0f}" if hits > 0 else "-"

    agg["single_evals"] = sum(r["stats"].get("single_evals", 0) for r in runs)
    agg["kick_evals"] = sum(r["stats"].get("kick_evals", 0) for r in runs)
    agg["tabu_hits"] = sum(r["stats"].get("tabu_hits", 0) for r in runs)
    agg["tabu_evals"] = sum(r["stats"].get("tabu_evals", 0) for r in runs)
    agg["tabu_solves"] = sum(r["stats"].get("tabu_solves", 0) for r in runs)
    agg["target_quenches"] = sum(r["stats"].get("target_quenches", 0) for r in runs)
    agg["candidate_evals"] = sum(r["stats"].get("total_candidate_evals", 0) for r in runs)
    agg["solve_phases"] = dict(
        Counter(
            cast(str, r["stats"]["solve_phase"])
            for r in runs
            if r["stats"].get("solve_phase") is not None
        )
    )

    return agg


def _write_output(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure solver component and targeted-escape ablations."
    )
    parser.add_argument("--targeted", action="store_true", help="Compare targeted escape on/off.")
    parser.add_argument(
        "--n",
        type=int,
        action="append",
        help=f"Sequence length; repeat for multiple values (default: {NS[0]}).",
    )
    parser.add_argument(
        "--seeds", type=int, default=N_SEEDS, help=f"Paired seed count (default: {N_SEEDS})."
    )
    parser.add_argument(
        "--steps", type=int, default=STEPS, help=f"Search steps (default: {STEPS})."
    )
    parser.add_argument(
        "--workers", type=int, default=WORKERS, help=f"Worker count (default: {WORKERS})."
    )
    parser.add_argument("--start-kind", choices=("random", "cyclic"), default="random")
    parser.add_argument("--output", type=Path, default=Path("runs/ablation.json"))
    args = parser.parse_args()
    configs: list[AblationConfig] = (
        [
            AblationConfig("targeted", SolverConfig()),
            AblationConfig("no-targeted", SolverConfig(targeted_escape=False)),
        ]
        if args.targeted
        else CONFIGS
    )
    ns = args.n or ([NS[0]] if args.targeted else NS)
    workers = args.workers

    # Build task list: one (n, seed, steps, config) per run
    tasks: list[tuple[str, int, int, int, str, SolverConfig]] = []
    for ac in configs:
        for n in ns:
            for seed in range(args.seeds):
                tasks.append((ac.name, n, seed, args.steps, args.start_kind, ac.config))

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
    for result in results:
        by_config[cast(str, result["config_name"])][cast(int, result["n"])].append(result)

    aggregates: dict[str, dict[str, dict[str, Any]]] = {}
    for ac in configs:
        aggregates[ac.name] = {str(n): _aggregate(by_config[ac.name][n]) for n in ns}

    _write_output(
        args.output,
        {
            "steps": args.steps,
            "seeds": args.seeds,
            "workers": workers,
            "start_kind": args.start_kind,
            "elapsed_seconds": elapsed,
            "configs": {ac.name: asdict(ac.config) for ac in configs},
            "aggregates": aggregates,
            "runs": results,
        },
    )

    # Print table
    for ac in configs:
        print()
        print(f"-- {ac.name} --")
        header = f"{'n':>4}  {'solved':>7}  {'mean_e':>7}  {'time':>6}  "
        header += f"{'S-hits':>7} {'TB-hit':>7} {'K-hits':>7}"
        print(header)
        print("-" * len(header))
        for n in ns:
            a = aggregates[ac.name][str(n)]
            line = (
                f"{n:>4}  {a['solved']:>7}  {a['mean_e']:>7}  {a['mean_t']:>6}  "
                f"{a['singles_hits']:>7} {a['tabu_hits']:>7} {a['kicks_hits']:>7}"
            )
            print(line)
        print()

    print(f"Raw results: {args.output}")


if __name__ == "__main__":
    main()
