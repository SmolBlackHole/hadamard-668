"""Ablation test: measure each solver component's contribution.

Run with ``python -m scripts.ablation`` from the repository root.

Runs the configured solver variants on identical seed ranges.
Outputs a comparison table centered on solve yield, wall time, unique solution
orbits, and candidate work. Candidate evaluations are only comparable within
the same solver version because the phases charge different kinds of work.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import time
from collections import Counter, defaultdict
from collections.abc import Iterable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, cast

from src.canonical import orbit_hash, validation_hash
from src.generator import start_construction
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
CANDIDATE_BUDGET = 6_000_000
WORKERS = max(1, int((os.cpu_count() or 2) * 0.8))
_worker_warmed = False


def _warm_worker() -> None:
    """Compile the common Numba path before measuring this worker's runs."""
    global _worker_warmed
    if _worker_warmed:
        return
    execute(
        "gs4",
        32,
        10_000,
        0,
        SolverConfig(targeted_escape=False),
        start_construction("random"),
    )
    _worker_warmed = True


def _run_one(args: tuple[str, int, int, int, str, SolverConfig]) -> dict[str, Any]:
    """Run a single search.  Pickle-friendly for ProcessPoolExecutor."""
    name, n, seed, candidate_budget, start_kind, config = args
    _warm_worker()
    started = time.perf_counter()
    result = execute("gs4", n, candidate_budget, seed, config, start_construction(start_kind))
    elapsed = time.perf_counter() - started
    solved = result.solved
    return {
        "config_name": name,
        "config": asdict(config),
        "start_kind": start_kind,
        "n": n,
        "seed": seed,
        "energy": result.energy,
        "solved": solved,
        "elapsed": elapsed,
        "candidate_evals": result.candidate_evals,
        "validation_hash": validation_hash(result.sequences) if solved else None,
        "orbit_hash": orbit_hash(result.sequences) if solved else None,
        "stats": result.stats.to_dict(),
    }


def _nearest_rank(values: list[float] | list[int], percentile: float) -> float | int | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(percentile * len(ordered)) - 1)]


def _aggregate(runs: list[dict[str, Any]]) -> dict[str, Any]:
    if not runs:
        raise ValueError("runs cannot be empty")
    solved_runs = [r for r in runs if r["solved"]]
    solved = len(solved_runs)
    total = len(runs)
    total_seconds = sum(cast(float, r["elapsed"]) for r in runs)
    candidate_evals = sum(cast(int, r["candidate_evals"]) for r in runs)
    solved_seconds = [cast(float, r["elapsed"]) for r in solved_runs]
    solved_evals = [cast(int, r["candidate_evals"]) for r in solved_runs]
    exact_solutions = {
        cast(str, r["validation_hash"]) for r in solved_runs if r["validation_hash"] is not None
    }
    unique_orbits = {cast(str, r["orbit_hash"]) for r in solved_runs if r["orbit_hash"] is not None}

    agg: dict[str, Any] = {
        "solved": solved,
        "total": total,
        "solve_rate": solved / total,
        "best_energy": min(r["energy"] for r in runs),
        "mean_energy": sum(r["energy"] for r in runs) / total,
        "mean_seconds": total_seconds / total,
        "solutions_per_worker_hour": solved * 3600 / total_seconds if total_seconds else None,
        "seconds_per_solution": total_seconds / solved if solved else None,
        "candidate_evals_per_solve": round(candidate_evals / solved) if solved else None,
        "solved_seconds_p50": statistics.median(solved_seconds) if solved_seconds else None,
        "solved_seconds_p90": _nearest_rank(solved_seconds, 0.9),
        "solved_candidate_evals_p50": (statistics.median(solved_evals) if solved_evals else None),
        "solved_candidate_evals_p90": _nearest_rank(solved_evals, 0.9),
        "unique_exact_solutions": len(exact_solutions),
        "unique_orbits": len(unique_orbits),
        "unique_orbits_per_worker_hour": (
            len(unique_orbits) * 3600 / total_seconds if total_seconds else None
        ),
        "total_seconds": total_seconds,
        "candidate_evals": candidate_evals,
    }

    for metric in (
        "greedy_moves",
        "tabu_moves",
        "random_kicks",
        "targeted_quenches",
        "quench_moves",
        "total_accepted_moves",
        "greedy_candidate_evals",
        "tabu_candidate_evals",
        "escape_candidate_evals",
        "quench_candidate_evals",
        "tabu_improvements",
        "tabu_walks",
    ):
        agg[metric] = sum(cast(int, r["stats"].get(metric, 0)) for r in runs)
    agg["tabu_solves"] = sum(r["stats"].get("tabu_solves", 0) for r in runs)
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
        "--candidate-budget",
        type=int,
        default=CANDIDATE_BUDGET,
        help=f"Candidate evaluations per run (default: {CANDIDATE_BUDGET}).",
    )
    parser.add_argument(
        "--workers", type=int, default=WORKERS, help=f"Worker count (default: {WORKERS})."
    )
    parser.add_argument("--start-kind", choices=("random", "cyclic"), default="random")
    parser.add_argument("--trace-phases", action="store_true")
    parser.add_argument("--output", type=Path, default=Path("runs/ablation.json"))
    args = parser.parse_args()
    if args.candidate_budget < 1 or args.seeds < 1 or args.workers < 1:
        parser.error("candidate-budget, seeds, and workers must be positive")
    configs: list[AblationConfig] = (
        [
            AblationConfig("targeted", SolverConfig()),
            AblationConfig("no-targeted", SolverConfig(targeted_escape=False)),
        ]
        if args.targeted
        else CONFIGS
    )
    if args.trace_phases:
        configs = [
            replace(item, config=replace(item.config, trace_phases=True)) for item in configs
        ]
    ns = args.n or ([NS[0]] if args.targeted else NS)
    workers = args.workers

    # Build task list: one (n, seed, candidate budget, config) per run
    tasks: list[tuple[str, int, int, int, str, SolverConfig]] = []
    for ac in configs:
        for n in ns:
            for seed in range(args.seeds):
                tasks.append((ac.name, n, seed, args.candidate_budget, args.start_kind, ac.config))

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
            "experiment_schema_version": 2,
            "timing_mode": "steady_state_after_per_worker_warmup",
            "candidate_budget": args.candidate_budget,
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
        header = (
            f"{'n':>4}  {'solved':>7}  {'orbits':>6}  {'sol/wh':>7}  "
            f"{'s/sol':>7}  {'Meval/sol':>10}  {'eval p50/p90':>19}"
        )
        print(header)
        print("-" * len(header))
        for n in ns:
            a = aggregates[ac.name][str(n)]
            solved_label = f"{a['solved']}/{a['total']}"
            rate = a["solutions_per_worker_hour"]
            seconds = a["seconds_per_solution"]
            evals_per_solve = a["candidate_evals_per_solve"]
            p50 = a["solved_candidate_evals_p50"]
            p90 = a["solved_candidate_evals_p90"]
            rate_label = f"{rate:.1f}" if rate is not None else "-"
            seconds_label = f"{seconds:.1f}" if seconds is not None else "-"
            evals_label = f"{evals_per_solve / 1_000_000:.2f}" if evals_per_solve else "-"
            line = (
                f"{n:>4}  {solved_label:>7}  {a['unique_orbits']:>6}  "
                f"{rate_label:>7}  {seconds_label:>7}  {evals_label:>10}  "
                f"{str(round(p50)) if p50 is not None else '-':>9}/"
                f"{str(round(p90)) if p90 is not None else '-':<9}"
            )
            print(line)
        print()

    print(f"Raw results: {args.output}")


if __name__ == "__main__":
    main()
