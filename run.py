"""CLI for GS4 search, construction, persistence, and database audits."""

from __future__ import annotations

import argparse
import signal
import time
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from pathlib import Path

from src.generator import (
    START_KINDS,
    STRATEGIES,
    StartConstruction,
    n_from_order,
    start_construction,
)
from src.models import RunResult
from src.output import save_runs
from src.pipeline import execute
from src.solver import SolverConfig
from src.verify import audit_database


def _format_time(seconds: float) -> str:
    if seconds < 0.001:
        return f"{seconds * 1_000_000:.0f}us"
    if seconds < 1.0:
        return f"{seconds * 1000:.0f}ms"
    return f"{seconds:.1f}s"


def _print_status(status: str, *, final: bool) -> None:
    print(f"\r{status:<79}", end="\n" if final else "", flush=True)


def _audit_with_progress(path: Path):
    started = time.perf_counter()
    last_update = started

    def progress(checked: int, total: int) -> None:
        nonlocal last_update
        now = time.perf_counter()
        if now - last_update >= 0.5 or checked == total:
            rate = checked / max(now - started, 1e-9)
            eta = (total - checked) / rate
            _print_status(
                f"DB audit [{checked}/{total}] {rate:.1f} rows/s ETA {_format_time(eta)}",
                final=checked == total,
            )
            last_update = now

    return audit_database(path, progress)


def _print_run_plan(
    tasks: list[tuple[str, int, int, int, SolverConfig, StartConstruction]],
    workers: int,
    output: Path | None,
) -> None:
    strategy, _, budget, _, config, start = tasks[0]
    ns = sorted({task[1] for task in tasks})
    seeds = sorted({task[3] for task in tasks})
    n_label = ",".join(str(n) for n in ns)
    order_label = ",".join(str(4 * n) for n in ns)
    seed_label = str(seeds[0]) if len(seeds) == 1 else f"{seeds[0]}..{seeds[-1]}"
    tabu = str(config.tabu_steps) if config.tabu else "off"
    print(
        f"Search: strategy={strategy} n={n_label} order={order_label} "
        f"runs={len(tasks)} seeds={seed_label}"
    )
    print(
        f"  budget={budget:,} candidate evaluations/run workers={workers} start={start.name} "
        f"tabu={tabu} kick={'on' if config.kick else 'off'} "
        f"targeted={'on' if config.targeted_escape else 'off'}"
    )
    print(f"  output={output if output is not None else 'disabled'}\n")


def _save_all(path: Path, results: list[RunResult]) -> None:
    noun = "run" if len(results) == 1 else "runs"
    print(f"\nSaving {len(results)} {noun} to {path}...", flush=True)
    started = time.perf_counter()
    last_update = started

    def progress(saved: int, total: int) -> None:
        nonlocal last_update
        now = time.perf_counter()
        if now - last_update >= 0.5 or saved == total:
            rate = saved / max(now - started, 1e-9)
            eta = (total - saved) / rate
            _print_status(
                f"DB write [{saved}/{total}] {rate:.1f} rows/s ETA {_format_time(eta)}",
                final=saved == total,
            )
            last_update = now

    save_runs(path, results, progress)


def _ignore_sigint() -> None:
    signal.signal(signal.SIGINT, signal.SIG_IGN)


def _execute_single(
    strategy: str,
    n: int,
    candidate_budget: int,
    seed: int,
    solver_config: SolverConfig,
    start: StartConstruction,
) -> RunResult:
    return execute(strategy, n, candidate_budget, seed, solver_config, start)


def _execute_all(
    tasks: list[tuple[str, int, int, int, SolverConfig, StartConstruction]], workers: int
) -> list[RunResult]:
    if workers > 1:
        total = len(tasks)
        ordered: list[RunResult | None] = [None] * total
        failures: list[tuple[int, Exception]] = []
        started = time.perf_counter()
        last_update = started
        future_indexes: dict[Future[RunResult], int] = {}
        with ProcessPoolExecutor(
            max_workers=min(workers, len(tasks)), initializer=_ignore_sigint
        ) as pool:
            try:
                for index, task in enumerate(tasks):
                    future_indexes[pool.submit(_execute_single, *task)] = index
                for completed, future in enumerate(as_completed(future_indexes), 1):
                    index = future_indexes[future]
                    try:
                        ordered[index] = future.result()
                    except Exception as error:
                        failures.append((index, error))

                    now = time.perf_counter()
                    if now - last_update >= 0.5 or completed == total:
                        results = [result for result in ordered if result is not None]
                        solved = sum(result.solved for result in results)
                        best_q = min(
                            (result.energy // (64 * result.n) for result in results), default=None
                        )
                        rate = completed / max(now - started, 1e-9)
                        eta = (total - completed) / rate
                        status = (
                            f"[{completed}/{total}] solved={solved} failed={len(failures)} "
                            f"bestQ={best_q if best_q is not None else '-'} "
                            f"{rate:.1f} runs/s ETA {_format_time(eta)}"
                        )
                        _print_status(status, final=completed == total)
                        last_update = now
            except KeyboardInterrupt:
                for future in future_indexes:
                    future.cancel()
                pool.shutdown(wait=True, cancel_futures=True)
                raise

        for index, error in failures:
            print(f"seed={tasks[index][3]} FAILED: {error}")
        return [result for result in ordered if result is not None]

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
    noun = "run" if len(results) == 1 else "runs"
    print(
        f"\n{len(results)} {noun}, {solved} solved, "
        f"{_format_time(total_seconds)} accumulated runtime"
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
        escape_quench_budget=args.escape_quench_budget,
        tabu=not args.no_tabu,
        kick=not args.no_kick,
        trace_phases=args.trace_phases,
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
    execution.add_argument("--candidate-budget", type=int, default=100_000_000)
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
    solver.add_argument(
        "--escape-quench-budget", type=int, default=SolverConfig.escape_quench_budget
    )
    solver.add_argument("--no-targeted-escape", action="store_true")
    solver.add_argument("--no-tabu", action="store_true")
    solver.add_argument("--no-kick", action="store_true")
    solver.add_argument("--trace-phases", action="store_true")
    return parser


def main() -> int:
    parser = _parser()
    args = parser.parse_args()
    if args.candidate_budget < 1 or args.runs < 1 or args.seeds < 1 or args.workers < 1:
        parser.error("candidate-budget, runs, seeds, and workers must be positive")
    if args.check is not None:
        print(f"Auditing database {args.check}...")
        report = _audit_with_progress(args.check)
        for failure in report.failures:
            print(f"  FAIL {failure}")
        print(f"{report.valid}/{report.checked} valid, {report.quarantined} quarantined")
        return 0 if report.ok else 1

    try:
        config = _solver_config(args)
        start = start_construction(args.start_kind)
        if args.sweep:
            strategy = args.sweep[0]
            if strategy not in STRATEGIES:
                parser.error(f"strategy must be one of {STRATEGIES}")
            ns = [int(value) for value in args.sweep[1:]]
            if not ns:
                parser.error("--sweep requires at least one n")
            tasks = [
                (strategy, n, args.candidate_budget, args.seed + offset, config, start)
                for n in ns
                for offset in range(args.seeds)
            ]
        else:
            n = n_from_order(args.strategy, args.order)
            tasks = [
                (
                    args.strategy,
                    n,
                    args.candidate_budget,
                    args.seed + offset,
                    config,
                    start,
                )
                for offset in range(args.runs)
            ]
    except ValueError as error:
        parser.error(str(error))

    _print_run_plan(tasks, args.workers, None if args.no_output else args.output)
    results = _execute_all(tasks, args.workers)
    _print_summary(results)
    if not results:
        return 1
    if not args.no_output:
        _save_all(args.output, results)
    else:
        print("\nOutput disabled; results were not saved.")
    if args.sweep and not args.no_output and not args.no_verify:
        print(f"\nAuditing database {args.output}...")
        report = _audit_with_progress(args.output)
        print(f"DB audit: {report.valid}/{report.checked} valid, {report.quarantined} quarantined")
        if not report.ok:
            print("Database audit failed, see above for details.")
            return 1
    elif args.sweep and not args.no_output:
        print("Database audit skipped (--no-verify).")
    print("\nAll done.")
    return 0


if __name__ == "__main__":
    try:
        exit_code = main()
    except KeyboardInterrupt:
        print("\nInterrupted; pending work stopped.")
        exit_code = 130
    raise SystemExit(exit_code)
