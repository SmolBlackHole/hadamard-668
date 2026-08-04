"""Compare all current Hadamard search strategies across supported orders."""
from __future__ import annotations
from strategies.turyn_steepest import TurynSteepestSearch
from strategies.pocs import TurynPocsSearch
from strategies.spectral import SpectralSearch
from strategies.repair import RepairSearch
from strategies.montecarlo import MonteCarloSearch
from strategies.ising import IsingSearch
from strategies.genetic import GeneticSearch
from strategies.circulant import TurynGreedySearch
from strategies.base import Pipeline
from strategies.annealing import TurynAnnealingSearch
from gpu import correlation_histogram, gram_matrix, xp
import argparse
import contextlib
import io
import json
import multiprocessing
import os
from queue import Empty
import subprocess
import time
from tqdm import tqdm
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))


ORDERS = (4, 8, 12, 16, 20, 668)
SIEVE_NS = (2, 3, 4, 5, 6, 7, 8, 9, 36, 56)
STEPS_SMALL = 2_000
STEPS_BIG = 5_000
SEED = 42
TIMEOUT_SECONDS = 60
GPU_COMPARE_STRATEGIES = ("repair", "ising", "spectral")


def _turyn_n(order: int) -> int:
    if order % 4:
        raise ValueError("Turyn orders are divisible by four")
    n = (order // 4 + 1) // 3
    if n < 2 or order != 4 * (3 * n - 1):
        raise ValueError("order has no TT(n) construction")
    return n


def _turyn_greedy(order: int) -> TurynGreedySearch:
    return TurynGreedySearch(n=_turyn_n(order))


def _turyn_annealing(order: int) -> TurynAnnealingSearch:
    return TurynAnnealingSearch(n=_turyn_n(order))


def _turyn_pocs(order: int) -> TurynPocsSearch:
    return TurynPocsSearch(n=_turyn_n(order))


def _turyn_steepest(order: int) -> TurynSteepestSearch:
    return TurynSteepestSearch(n=_turyn_n(order))


def sieve_strategies(sieve: bool):
    """Return all Turyn solvers with identical sieve configuration."""
    return (
        ("TurynGreedy", lambda n: TurynGreedySearch(n=n, sieve=sieve)),
        ("TurynAnnealing", lambda n: TurynAnnealingSearch(n=n, sieve=sieve)),
        ("TurynPOCS", lambda n: TurynPocsSearch(n=n, sieve=sieve)),
        ("TurynSteepest", lambda n: TurynSteepestSearch(n=n, sieve=sieve)),
        ("MonteCarlo", lambda n: MonteCarloSearch(
            order=4 * (3 * n - 1), sieve=sieve)),
        ("Spectral", lambda n: SpectralSearch(
            ORDER=4 * (3 * n - 1), inner_steps=5, sieve=sieve)),
        ("Genetic", lambda n: GeneticSearch(order=4 * (3 * n - 1), sieve=sieve)),
        ("Ising", lambda n: IsingSearch(order=4 * (3 * n - 1), sieve=sieve)),
    )


def _half_steps(order: int) -> int:
    return STEPS_BIG // 2 if order == 668 else STEPS_SMALL // 2


def steps_for_order(order: int) -> int:
    return STEPS_BIG if order == 668 else STEPS_SMALL


def benchmark_groups(*, include_all: bool = False):
    core = (
        ("Individual strategies", (
            ("TurynGreedy", _turyn_greedy),
            ("TurynAnnealing", _turyn_annealing),
            ("TurynPOCS", _turyn_pocs),
            ("TurynSteepest", _turyn_steepest),
            ("MonteCarlo", lambda order: MonteCarloSearch(order=order)),
            ("RepairSearch", lambda order: RepairSearch(order=order)),
            ("Spectral", lambda order: SpectralSearch(ORDER=order, inner_steps=5)),
        )),
    )
    if not include_all:
        return core
    experimental = (
        ("Experimental strategies", (
            ("Genetic", lambda order: GeneticSearch(order=order)),
            ("Ising", lambda order: IsingSearch(order=order)),
        )),
        ("Pipelines", (
            ("TurynGreedy->Repair", lambda o: Pipeline(
                [(_turyn_greedy(o), _half_steps(o)), (RepairSearch(order=o), _half_steps(o))])),
            ("TurynAnnealing->Repair", lambda o: Pipeline(
                [(_turyn_annealing(o), _half_steps(o)), (RepairSearch(order=o), _half_steps(o))])),
        )),
    )
    return core + experimental


def _benchmark_worker(strategy, steps: int, seed: int, results) -> None:
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            matrix, metrics, algorithm_seconds = strategy.search(
                steps=steps, seed=seed)
        results.put({"status": "ok", "backend": getattr(
            strategy, "compute_backend", xp.__name__),
            "algorithm_seconds": algorithm_seconds,
            "metrics": metrics,
            "correlation_histogram": correlation_histogram(gram_matrix(matrix))})
    except BaseException as error:
        results.put({"status": "error", "message": str(error)[:120]})


def run_one(factory, order: int, steps: int, seed: int, timeout_seconds: float = TIMEOUT_SECONDS) -> dict:
    try:
        strategy = factory(order)
    except Exception as error:
        return {"status": "error", "message": f"initialisation: {error}"}
    context = multiprocessing.get_context("spawn")
    results = context.Queue()
    process = context.Process(target=_benchmark_worker,
                              args=(strategy, steps, seed, results))
    started = time.perf_counter()
    process.start()
    process.join(timeout_seconds)
    wall = time.perf_counter() - started
    if process.is_alive():
        process.terminate()
        process.join()
        return {"status": "timeout", "wall_seconds": wall}
    try:
        payload = results.get(timeout=1)
    except Empty:
        return {"status": "error", "message": f"worker exit={process.exitcode}"}
    if payload["status"] == "error":
        payload["wall_seconds"] = wall
        return payload
    payload["wall_seconds"] = wall
    return payload


def _cell(result: dict, order: int) -> str:
    if result["status"] == "na":
        return "N/A"
    if result["status"] == "sieved_out":
        return "SIEVED OUT"
    if result["status"] == "timeout":
        return "TIMEOUT"
    if result["status"] == "error":
        return "ERROR"
    metrics = result["metrics"]
    if metrics["energy"] == 0:
        return f"YES; algo={result['algorithm_seconds']:.1f}s"
    total = order * (order - 1) // 2
    rms = (metrics["energy"] / total) ** 0.5
    return (f"OK; e={metrics['energy']}; rms={rms:.2f}; "
            f"orth={metrics['orthogonal_pairs']}/{total}; algo={result['algorithm_seconds']:.1f}s")


def _render_table(title: str, rows: list[tuple[str, list[str]]]) -> list[str]:
    header = "| Strategy | " + \
        " | ".join(str(order) for order in ORDERS) + " |"
    divider = "|---|" + "---|" * len(ORDERS)
    return [f"## {title}", "", header, divider,
            *(f"| {name} | " + " | ".join(values) + " |" for name, values in rows), ""]


def main(timeout_seconds: float = TIMEOUT_SECONDS, *, include_all: bool = False) -> None:
    groups = benchmark_groups(include_all=include_all)
    total_cases = sum(len(strategies) * len(ORDERS)
                      for _, strategies in groups)
    rendered: list[tuple[str, list[tuple[str, list[str]]]]] = []
    case_results: list[dict[str, object]] = []
    print(
        f"Hadamard benchmark: {total_cases} cases | seed={SEED} | timeout={timeout_seconds:g}s")
    with tqdm(total=total_cases, desc="Benchmark", unit="case", dynamic_ncols=True) as progress:
        for group, strategies in groups:
            table_rows = []
            for name, factory in strategies:
                cells = []
                for order in ORDERS:
                    steps = steps_for_order(order)
                    progress.set_postfix_str(
                        f"{group}: {name}, n={order}, steps={steps}")
                    turyn_strategy = name.startswith("Turyn") or name in {
                        "MonteCarlo", "Ising", "Spectral", "Genetic",
                    }
                    result = ({"status": "na"} if turyn_strategy and (
                        order != 4 * (3 * ((order // 4 + 1) // 3) - 1)
                        or ((order // 4 + 1) // 3) < 2
                    ) else run_one(factory, order, steps, SEED, timeout_seconds))
                    case_results.append({
                        "group": group,
                        "strategy": name,
                        "order": order,
                        "steps": steps,
                        "seed": SEED,
                        **result,
                    })
                    cells.append(_cell(result, order))
                    if result["status"] == "timeout":
                        tqdm.write(f"TIMEOUT  {group} | {name} | n={order}")
                    elif result["status"] == "error":
                        tqdm.write(
                            f"ERROR    {group} | {name} | n={order} | {result['message']}")
                    progress.update()
                table_rows.append((name, cells))
            rendered.append((group, table_rows))
    budgets = ", ".join(
        f"n={order}: {steps_for_order(order)}" for order in ORDERS)
    lines = ["# Hadamard Benchmark", "", f"Backend: {xp.__name__}", f"Seed: {SEED}",
             f"Step budgets: {budgets}",
             f"Timeout per case: {timeout_seconds:g} seconds", "",
             "Cell format: `status; e=energy; rms=root-mean-square correlation; orth=orthogonal pairs; algo=search seconds`.", ""]
    for group, rows in rendered:
        lines.extend(_render_table(group, rows))
    report = "\n".join(lines).rstrip()
    Path("benchmark_results.md").write_text(report + "\n", encoding="utf-8")
    Path("benchmark_results.json").write_text(json.dumps({
        "backend": xp.__name__,
        "seed": SEED,
        "timeout_seconds": timeout_seconds,
        "include_all": include_all,
        "cases": case_results,
    }, indent=2) + "\n", encoding="utf-8")
    print("\n" + report)
    print("Saved: benchmark_results.md and benchmark_results.json")


def sieve_compare(steps: int, timeout_seconds: float, ns: tuple[int, ...] = SIEVE_NS) -> None:
    """Compare random and sieved initialization across TT(n) orders."""
    def run_case(factory, n: int) -> dict:
        started = time.perf_counter()
        try:
            strategy = factory(n)
            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                matrix, metrics, algorithm_seconds = strategy.search(
                    steps, SEED)
            wall_seconds = time.perf_counter() - started
            if wall_seconds > timeout_seconds:
                return {"status": "timeout", "wall_seconds": wall_seconds}
            return {
                "status": "ok", "backend": getattr(strategy, "compute_backend", xp.__name__),
                "algorithm_seconds": algorithm_seconds, "wall_seconds": wall_seconds,
                "metrics": metrics,
                "correlation_histogram": correlation_histogram(gram_matrix(matrix)),
            }
        except ValueError as error:
            if "no compatible row-sum pattern" in str(error):
                return {"status": "sieved_out"}
            return {"status": "error", "message": str(error)[:120]}
        except BaseException as error:
            return {"status": "error", "message": str(error)[:120]}

    records: list[dict[str, object]] = []
    rows = ["# Turyn Sieve Comparison", "", f"Seed: {SEED}",
            f"Steps per run: {steps}", f"Timeout per run: {timeout_seconds:g}s", "",
            "| solver | n | order | random start | sieved start |",
            "|---|---:|---:|---|---|"]
    random_strategies = dict(sieve_strategies(False))
    sieved_strategies = dict(sieve_strategies(True))
    for name, factory in random_strategies.items():
        for n in ns:
            order = 4 * (3 * n - 1)
            random_result = run_case(factory, n)
            sieve_factory = sieved_strategies[name]
            sieve_result = run_case(sieve_factory, n)
            records.extend((
                {"solver": name, "n": n, "order": order,
                    "initialization": "random", **random_result},
                {"solver": name, "n": n, "order": order,
                    "initialization": "sieve", **sieve_result},
            ))
            rows.append(
                f"| {name} | {n} | {order} | {_cell(random_result, order)} | {_cell(sieve_result, order)} |")
    report = "\n".join(rows) + "\n"
    Path("benchmark_results.md").write_text(report, encoding="utf-8")
    Path("benchmark_results.json").write_text(json.dumps({
        "kind": "sieve_compare", "backend": xp.__name__, "seed": SEED,
        "steps": steps, "timeout_seconds": timeout_seconds, "cases": records,
    }, indent=2) + "\n", encoding="utf-8")
    print(report)
    print("Saved: benchmark_results.md and benchmark_results.json")


def _gpu_strategy(name: str):
    factories = {
        "repair": RepairSearch, "ising": IsingSearch,
        "spectral": lambda: SpectralSearch(inner_steps=1),
    }
    return factories[name]()


def _gpu_worker(name: str, steps: int) -> None:
    strategy = _gpu_strategy(name)
    started = time.perf_counter()
    _, metrics, _ = strategy.search(steps=steps, seed=SEED)
    print(json.dumps({"strategy": name, "backend": getattr(
        strategy, "compute_backend", xp.__name__),
        "seconds": round(time.perf_counter() - started, 3), "energy": metrics["energy"]}))


def gpu_compare(steps: int) -> None:
    rows = ["# GPU comparison", "",
            "| strategy | numpy | cupy |", "|---|---:|---:|"]
    script = str(Path(__file__).resolve())
    for name in GPU_COMPARE_STRATEGIES:
        cells = []
        for backend in ("numpy", "cupy"):
            env = os.environ.copy()
            env["HADAMARD_BACKEND"] = backend
            try:
                result = subprocess.run([sys.executable, script, "--gpu-worker", name, "--steps", str(
                    steps)], capture_output=True, text=True, env=env, timeout=TIMEOUT_SECONDS, check=True)
                payload = json.loads(result.stdout.strip().splitlines()[-1])
                cells.append(
                    f"{payload['seconds']:.3f}s (e={payload['energy']})")
            except subprocess.TimeoutExpired:
                cells.append("TIMEOUT (>60s)")
            except (subprocess.CalledProcessError, json.JSONDecodeError) as error:
                cells.append(f"ERROR ({type(error).__name__})")
        rows.append(f"| {name} | {cells[0]} | {cells[1]} |")
    Path("benchmark_gpu_compare.md").write_text(
        "\n".join(rows) + "\n", encoding="utf-8")
    print("Saved: benchmark_gpu_compare.md")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timeout", type=float, default=TIMEOUT_SECONDS)
    parser.add_argument("--all", action="store_true",
                        help="include experimental strategies and pipelines")
    parser.add_argument("--gpu-compare", action="store_true")
    parser.add_argument("--sieve-compare", action="store_true",
                        help="compare random and sieved TT(n) initialization")
    parser.add_argument("--sieve-n", type=int, nargs="+", default=SIEVE_NS,
                        help="TT(n) values for --sieve-compare")
    parser.add_argument("--gpu-worker", choices=GPU_COMPARE_STRATEGIES)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()
    if args.gpu_worker:
        _gpu_worker(args.gpu_worker, args.steps)
    elif args.gpu_compare:
        gpu_compare(args.steps)
    elif args.sieve_compare:
        sieve_compare(args.steps, args.timeout, tuple(args.sieve_n))
    else:
        main(args.timeout, include_all=args.all)
