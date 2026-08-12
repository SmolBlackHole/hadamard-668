"""CLI for GS4 search, construction, persistence, and database audits."""

from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

from src.generator import START_KINDS, STRATEGIES, n_from_order
from src.models import RunResult
from src.output import save_run
from src.pipeline import execute
from src.solver import SolverConfig
from src.verify import audit_database


def _format_time(seconds: float) -> str:
    if seconds < 0.001:
        return f"{seconds * 1_000_000:.0f}us"
    if seconds < 1.0:
        return f"{seconds * 1000:.0f}ms"
    return f"{seconds:.1f}s"


def _execute_single(
    strategy: str,
    n: int,
    steps: int,
    seed: int,
    solver_config: SolverConfig,
    start_kind: str,
) -> RunResult:
    return execute(strategy, n, steps, seed, solver_config, start_kind)


def _execute_all(
    tasks: list[tuple[str, int, int, int, SolverConfig, str]], workers: int
) -> list[RunResult]:
    if workers > 1:
        with ProcessPoolExecutor(max_workers=min(workers, len(tasks))) as pool:
            return list(pool.map(_execute_single, *zip(*tasks, strict=True)))

    results: list[RunResult] = []
    for index, task in enumerate(tasks, 1):
        print(f"[{index}/{len(tasks)}] n={task[1]}:", end=" ", flush=True)
        try:
            result = _execute_single(*task)
        except Exception as error:
            print(f"FAILED: {error}")
            continue
        results.append(result)
        print(result)
    return results


def _print_summary(results: list[RunResult]) -> None:
    if not results:
        print("No run completed successfully.")
        return
    total_seconds = sum(result.elapsed_seconds for result in results)
    solved = sum(result.solved for result in results)
    print(
        f"\n{len(results)} runs, {solved} solved, {_format_time(total_seconds)} accumulated runtime"
    )
    by_n: dict[int, list[RunResult]] = {}
    for result in results:
        by_n.setdefault(result.n, []).append(result)
    for n, runs in sorted(by_n.items()):
        count = sum(result.solved for result in runs)
        best = min(result.energy for result in runs)
        average = sum(result.elapsed_seconds for result in runs) / len(runs)
        print(f"  n={n:>3}  {count}/{len(runs)} solved  best_e={best}  avg {_format_time(average)}")


def _solver_config(args: argparse.Namespace) -> SolverConfig:
    return SolverConfig(
        tabu_steps=args.tabu_steps,
        tabu_tenure=args.tabu_tenure,
        tabu_decay=args.tabu_decay,
        tabu_noise=args.tabu_noise,
        geo_weight=args.geo_weight,
        targeted_escape=not args.no_targeted_escape,
        escape_quench_steps=args.escape_quench_steps,
        tabu=not args.no_tabu,
        kick=not args.no_kick,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    construction = parser.add_argument_group("construction")
    construction.add_argument("--strategy", choices=STRATEGIES, default="gs4")
    construction.add_argument("--order", type=int, default=668)
    construction.add_argument("--start-kind", choices=START_KINDS, default="random")
    construction.add_argument(
        "--sweep",
        nargs="+",
        metavar=("STRATEGY", "N"),
        help="Run one strategy for one or more sequence lengths n",
    )

    execution = parser.add_argument_group("execution")
    execution.add_argument("--steps", type=int, default=200_000)
    execution.add_argument("--seed", type=int, default=42)
    execution.add_argument("--runs", type=int, default=1)
    execution.add_argument("--seeds", type=int, default=10)
    execution.add_argument("--workers", type=int, default=1)
    execution.add_argument("--output", type=Path, default=Path("data/hadamard.db"))
    execution.add_argument("--no-output", action="store_true")
    execution.add_argument("--no-verify", action="store_true", help="Skip the post-sweep DB audit")
    execution.add_argument("--check", type=Path, help="Audit an existing SQLite database and exit")

    solver = parser.add_argument_group("solver")
    solver.add_argument("--tabu-steps", type=int, default=SolverConfig.tabu_steps)
    solver.add_argument("--tabu-tenure", type=float, default=SolverConfig.tabu_tenure)
    solver.add_argument("--tabu-decay", type=float, default=SolverConfig.tabu_decay)
    solver.add_argument("--tabu-noise", type=float, default=SolverConfig.tabu_noise)
    solver.add_argument("--geo-weight", type=float, default=SolverConfig.geo_weight)
    solver.add_argument("--escape-quench-steps", type=int, default=SolverConfig.escape_quench_steps)
    solver.add_argument("--no-targeted-escape", action="store_true")
    solver.add_argument("--no-tabu", action="store_true")
    solver.add_argument("--no-kick", action="store_true")
    return parser


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    if args.steps < 1 or args.runs < 1 or args.seeds < 1 or args.workers < 1:
        parser.error("steps, runs, seeds, and workers must be positive")
    if args.check is not None:
        report = audit_database(args.check)
        for failure in report.failures:
            print(f"  FAIL {failure}")
        print(f"{report.valid}/{report.checked} valid")
        return 0 if report.ok else 1

    try:
        config = _solver_config(args)
        if args.sweep:
            strategy = args.sweep[0]
            if strategy not in STRATEGIES:
                parser.error(f"strategy must be one of {STRATEGIES}")
            ns = [int(value) for value in args.sweep[1:]]
            if not ns:
                parser.error("--sweep requires at least one n")
            tasks = [
                (strategy, n, args.steps, args.seed + offset, config, args.start_kind)
                for n in ns
                for offset in range(args.seeds)
            ]
        else:
            n = n_from_order(args.strategy, args.order)
            tasks = [
                (
                    args.strategy,
                    n,
                    args.steps,
                    args.seed + offset,
                    config,
                    args.start_kind,
                )
                for offset in range(args.runs)
            ]
    except ValueError as error:
        parser.error(str(error))

    results = _execute_all(tasks, args.workers)
    _print_summary(results)
    if not results:
        return 1
    if not args.no_output:
        for result in results:
            save_run(args.output, result)
    if args.sweep and not args.no_output and not args.no_verify:
        report = audit_database(args.output)
        print(f"DB audit: {report.valid}/{report.checked} valid")
        if not report.ok:
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
