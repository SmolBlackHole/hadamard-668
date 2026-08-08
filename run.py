"""Hadamard matrix search — single runs and multi-n sweeps.

Usage::

    python run.py --strategy gs4 --order 128 --steps 200000 --seed 42 --output data/runs.json
    python run.py --sweep gs4 32 34 36 --seeds 10 --steps 200000 --workers 4 --output data/baseline.json
    python run.py --check data/runs.json
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from src.benchmark_stats import wilson_ci
from src.generator import Generator, Result
from src.output import save_run, verify
from src.solver import SolverConfig


def _fmt_time(t: float) -> str:
    """Format a duration in seconds to human-readable (us / ms / s)."""
    if t < 0.001:
        return f"{t * 1_000_000:.0f}us"
    if t < 1.0:
        return f"{t * 1000:.0f}ms"
    return f"{t:.1f}s"


def _execute_single(
    kind: str, n: int, steps: int, seed: int, solver_config: SolverConfig | None = None
) -> Result:
    """Run one search (pickle-friendly for ProcessPoolExecutor)."""
    gen = Generator(kind=kind, n=n)
    started = time.perf_counter()
    result = gen.search(steps=steps, seed=seed, config=solver_config)
    result.elapsed = time.perf_counter() - started
    return result


def _run_sweep(
    strategy: str,
    ns: list[int],
    seeds: int,
    steps: int,
    workers: int,
    solver_config: SolverConfig,
    output_path: Path | None,
) -> None:
    """Run a sweep: for each n, run ``seeds`` independent searches."""
    tasks: list[tuple[str, int, int, int, SolverConfig]] = []
    for n in ns:
        for s in range(seeds):
            tasks.append((strategy, n, steps, s, solver_config))

    total = len(tasks)
    results: list[Result] = []
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_execute_single, *zip(*tasks, strict=True)))
    else:
        for i, (kind, n, st, seed, cfg) in enumerate(tasks, 1):
            print(f"[{i}/{total}] n={n}:", end=" ", flush=True)
            try:
                results.append(_execute_single(kind, n, st, seed, cfg))
            except Exception as exc:
                print(f"FAILED: {exc}")

    for r in results:
        print(f"  {r}")

    by_n: dict[int, list[Result]] = {}
    for r, (_, n, _, _, _) in zip(results, tasks, strict=True):
        by_n.setdefault(n, []).append(r)

    total_t = sum(r.elapsed for r in results)
    tt = max(total_t, 0.001)
    total_solved = sum(1 for r in results if r.metrics.energy == 0)
    throughput = total_solved / tt * 3600
    print(
        f"\nSweep {strategy} ({seeds} seeds x {len(ns)} n, "
        f"{_fmt_time(total_t)} total, {total_solved} solved, "
        f"{throughput:.0f} sol/h)"
    )
    for n in sorted(by_n):
        rs = by_n[n]
        solved = sum(1 for r in rs if r.metrics.energy == 0)
        best_e = min(r.metrics.energy for r in rs)
        avg_t = sum(r.elapsed for r in rs) / len(rs)
        lo, hi = wilson_ci(solved, len(rs))
        line = f"  n={n:>3}  {solved}/{len(rs)} solved  best_e={best_e}  avg {_fmt_time(avg_t)}  ci=[{lo:.3f}, {hi:.3f}]"
        if solved < len(rs):
            failed = [r.metrics.energy for r in rs if r.metrics.energy > 0]
            line += f"  failed_e={failed}"
        print(line)

    if output_path:
        for r, (_, n, _, _, _) in zip(results, tasks, strict=True):
            save_run(output_path, strategy, n, r)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hadamard matrix search via GS4 construction.",
        epilog="sweep example: python run.py --sweep gs4 32 34 36 --seeds 10 --workers 4 --output data.json",
    )
    parser.add_argument(
        "--strategy",
        default="gs4",
        help="Search strategy (default: gs4)",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=200_000,
        help="Maximum flip evaluations per run (default: 200000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Base RNG seed; when --runs > 1, seeds are seed, seed+1, ...",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=1,
        help="Number of independent searches for single-run mode (default: 1)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Parallel worker processes (default: 1; set to cpu count for sweep)",
    )
    parser.add_argument(
        "--order",
        type=int,
        default=668,
        help="Target Hadamard matrix order for single-run mode (default: 668)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/benchmark.json",
        help="Append runs to JSON file (default: data/benchmark.json)",
    )
    parser.add_argument(
        "--no-output",
        action="store_true",
        help="Disable writing results to file",
    )
    parser.add_argument(
        "--check",
        type=str,
        default=None,
        help="Verify SHA-256 integrity and Hadamard property of a dataset file, then exit",
    )
    parser.add_argument(
        "--no-verify",
        action="store_true",
        help="Skip post-sweep integrity check of the output file",
    )

    # sweep mode
    parser.add_argument(
        "--sweep",
        type=str,
        nargs="*",
        default=None,
        help="Sweep mode: STRATEGY followed by one or more n values to test",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=10,
        help="Number of independent seeds per n in sweep mode (default: 10)",
    )

    # Solver config
    solver_group = parser.add_argument_group("SolverConfig")
    solver_group.add_argument(
        "--tabu-steps",
        type=int,
        default=SolverConfig.tabu_steps,
        help=f"Tabu walk steps (default: {SolverConfig.tabu_steps})",
    )
    solver_group.add_argument(
        "--tabu-tenure",
        type=float,
        default=SolverConfig.tabu_tenure,
        help=f"Tabu tenure (default: {SolverConfig.tabu_tenure})",
    )
    solver_group.add_argument(
        "--tabu-decay",
        type=float,
        default=SolverConfig.tabu_decay,
        help=f"Tabu decay (default: {SolverConfig.tabu_decay})",
    )
    solver_group.add_argument(
        "--tabu-noise",
        type=float,
        default=SolverConfig.tabu_noise,
        help=f"Tabu noise (default: {SolverConfig.tabu_noise})",
    )
    solver_group.add_argument(
        "--geo-weight",
        type=float,
        default=SolverConfig.geo_weight,
        help=f"||d||^2 penalty in greedy (default: {SolverConfig.geo_weight})",
    )
    solver_group.add_argument(
        "--no-targeted-escape", action="store_true", help="Disable targeted escape"
    )
    solver_group.add_argument("--no-tabu", action="store_true", help="Disable tabu walk")
    solver_group.add_argument("--no-kick", action="store_true", help="Disable kick")
    args = parser.parse_args()

    solver_config = SolverConfig(
        tabu_steps=args.tabu_steps,
        tabu_tenure=args.tabu_tenure,
        tabu_decay=args.tabu_decay,
        tabu_noise=args.tabu_noise,
        geo_weight=args.geo_weight,
        targeted_escape=not args.no_targeted_escape,
        tabu=not args.no_tabu,
        kick=not args.no_kick,
    )

    if args.check is not None:
        ok = verify(Path(args.check))
        sys.exit(0 if ok > 0 else 1)

    # --- sweep mode -----------------------------------------------------------
    if args.sweep is not None:
        if len(args.sweep) < 2:
            parser.error("--sweep STRATEGY N1 [N2 ...]")
        strategy = args.sweep[0]
        ns = [int(x) for x in args.sweep[1:]]
        output_path = None if args.no_output else Path(args.output)
        _run_sweep(
            strategy,
            ns,
            args.seeds,

            args.steps,
            args.workers,
            solver_config,
            output_path,
        )
        if output_path and not args.no_verify:
            ok = verify(output_path)
            print(f"Verify {output_path}: {'OK' if ok > 0 else 'FAILED'} "
                  f"({ok} valid entries)")
        return

    # --- single-run mode ------------------------------------------------------
    try:
        gen = Generator.from_cli(args.strategy, args.order)
        seeds = [args.seed + offset for offset in range(args.runs)]
        workers = min(args.runs, args.workers)
    except ValueError as e:
        parser.error(str(e))

    results: list[Result] = []
    if workers == 1:
        for s in seeds:
            try:
                r = _execute_single(gen._builder.kind, gen._builder.n, args.steps, s, solver_config)
                results.append(r)
                print(f"  {r}")
            except Exception as exc:
                print(f"  run seed={s} FAILED: {exc}")
    else:
        tasks = [(gen._builder.kind, gen._builder.n, args.steps, s, solver_config) for s in seeds]
        with ProcessPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(_execute_single, *zip(*tasks, strict=True)))
        for r in results:
            print(f"  {r}")

    if args.output:
        for r in results:
            save_run(Path(args.output), gen.name, gen._builder.n, r)

    best = min(results, key=lambda r: r.metrics.energy)
    print("\nRun summary")
    for r in results:
        print(f"  {r}{' *' if r.seed == best.seed else ''}")
    if best.metrics.energy == 0:
        print("*** HADAMARD! ***")


if __name__ == "__main__":
    main()
