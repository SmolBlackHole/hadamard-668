"""Compare all current Hadamard search strategies across supported orders."""
from __future__ import annotations
from strategies.walsh import WalshSearch
from strategies.spectral import SpectralSearch
from strategies.rowwise import RowwiseSearch
from strategies.repair import RepairSearch
from strategies.montecarlo import MonteCarloSearch
from strategies.ising import IsingSearch
from strategies.gold import GoldSearch
from strategies.genetic import GeneticSearch
from strategies.direct import DirectSearch
from strategies.diffset import DiffsetSearch
from strategies.circulant import CirculantSearch
from strategies.ca import CASearch
from strategies.baumert import BaumertHallSearch
from strategies.base import Pipeline
from strategies.annealing import AnnealingSearch

import argparse
import contextlib
import io
import json
import multiprocessing
import os
from pathlib import Path
from queue import Empty
import subprocess
import sys
import time

from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent / "src"))


ORDERS = (4, 8, 12, 16, 20, 668)
STEPS_SMALL = 2_000
STEPS_BIG = 5_000
SEED = 42
TIMEOUT_SECONDS = 60


def _circulant(order: int) -> CirculantSearch:
    k = order // 4
    half = (k + 1) // 2
    return CirculantSearch(ORDER=order, K=k, HALF=half) if order <= 20 else CirculantSearch()


def _annealing(order: int) -> AnnealingSearch:
    k = order // 4
    half = (k + 1) // 2
    return AnnealingSearch(ORDER=order, K=k, HALF=half) if order <= 20 else AnnealingSearch()


def _baumert(order: int) -> BaumertHallSearch:
    t = order // 4
    half = (t + 1) // 2
    return BaumertHallSearch(ORDER=order, T=t, HALF=half) if order <= 20 else BaumertHallSearch()


def _half_steps(order: int) -> int:
    return STEPS_BIG // 2 if order == 668 else STEPS_SMALL // 2


def benchmark_groups():
    return (
        ("Individual strategies", (
            ("Circulant", _circulant),
            ("Hybrid", lambda o: CirculantSearch(constructions="all", ORDER=o, K=o//4,
             HALF=(o//4+1)//2) if o <= 20 else CirculantSearch(constructions="all")),
            ("Annealing", _annealing),
            ("BaumertHall", _baumert),
            ("Diffset", lambda order: DiffsetSearch(order=order)),
            ("MonteCarlo", lambda order: MonteCarloSearch(order=order)),
            ("DirectSearch", lambda order: DirectSearch(order=order)),
            ("RepairSearch", lambda order: RepairSearch(order=order)),
            ("Ising", lambda order: IsingSearch(order=order)),
            ("Rowwise", lambda order: RowwiseSearch(order=order)),
            ("Walsh", lambda order: WalshSearch(order=order)),
            ("Genetic", lambda order: GeneticSearch(order=order)),
            ("Gold/LFSR", lambda order: GoldSearch(order=order)),
            ("CA spectral", lambda order: CASearch(order=order, mode="spectral")),
            ("Spectral", lambda order: SpectralSearch(ORDER=order, inner_steps=5)),
        )),
        ("Pipelines", (
            ("Circulant→Annealing", lambda o: Pipeline(
                [(_circulant(o), _half_steps(o)), (_annealing(o), _half_steps(o))])),
            ("Circulant→Repair", lambda o: Pipeline(
                [(_circulant(o), _half_steps(o)), (RepairSearch(order=o), _half_steps(o))])),
            ("Annealing→Repair", lambda o: Pipeline(
                [(_annealing(o), _half_steps(o)), (RepairSearch(order=o), _half_steps(o))])),
            ("Circulant→Annealing→Repair", lambda o: Pipeline([(_circulant(o), _half_steps(
                o)//2), (_annealing(o), _half_steps(o)//2), (RepairSearch(order=o), _half_steps(o))])),
        )),
    )


def _benchmark_worker(strategy, steps: int, seed: int, results) -> None:
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            started = time.perf_counter()
            _, metrics, _ = strategy.search(steps=steps, seed=seed)
        results.put(
            {"status": "ok", "seconds": time.perf_counter() - started, "metrics": metrics})
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
        return {"status": "timeout", "seconds": wall}
    try:
        payload = results.get(timeout=1)
    except Empty:
        return {"status": "error", "message": f"worker exit={process.exitcode}"}
    if payload["status"] == "error":
        return payload
    payload["seconds"] = wall
    return payload


def _cell(result: dict, order: int) -> str:
    if result["status"] == "na":
        return "N/A"
    if result["status"] == "timeout":
        return "TIMEOUT"
    if result["status"] == "error":
        return "ERROR"
    metrics = result["metrics"]
    if metrics["energy"] == 0:
        return f"YES {result['seconds']:.1f}s"
    total = order * (order - 1) // 2
    return f"e={metrics['energy']} | {metrics['orthogonal_pairs']}/{total} | {result['seconds']:.1f}s"


def _render_table(title: str, rows: list[tuple[str, list[str]]]) -> list[str]:
    header = "| Strategy | " + \
        " | ".join(str(order) for order in ORDERS) + " |"
    divider = "|---|" + "---|" * len(ORDERS)
    return [f"## {title}", "", header, divider,
            *(f"| {name} | " + " | ".join(values) + " |" for name, values in rows), ""]


def main(timeout_seconds: float = TIMEOUT_SECONDS) -> None:
    groups = benchmark_groups()
    total_cases = sum(len(strategies) * len(ORDERS)
                      for _, strategies in groups)
    rendered: list[tuple[str, list[tuple[str, list[str]]]]] = []
    print(
        f"Hadamard benchmark: {total_cases} cases | seed={SEED} | timeout={timeout_seconds:g}s")
    with tqdm(total=total_cases, desc="Benchmark", unit="case", dynamic_ncols=True) as progress:
        for group, strategies in groups:
            table_rows = []
            for name, factory in strategies:
                cells = []
                for order in ORDERS:
                    steps = STEPS_BIG if order == 668 else STEPS_SMALL
                    progress.set_postfix_str(
                        f"{group}: {name}, n={order}, steps={steps}")
                    result = ({"status": "na"} if name == "BaumertHall" and (order // 4) % 2 == 0
                              else run_one(factory, order, steps, SEED, timeout_seconds))
                    cells.append(_cell(result, order))
                    if result["status"] == "timeout":
                        tqdm.write(f"TIMEOUT  {group} | {name} | n={order}")
                    elif result["status"] == "error":
                        tqdm.write(
                            f"ERROR    {group} | {name} | n={order} | {result['message']}")
                    progress.update()
                table_rows.append((name, cells))
            rendered.append((group, table_rows))
    lines = ["# Hadamard Benchmark", "", f"Seed: {SEED}",
             f"Timeout per case: {timeout_seconds:g} seconds", "",
             "Cell format: `e=energy | orthogonal_pairs/total_pairs | seconds`.", ""]
    for group, rows in rendered:
        lines.extend(_render_table(group, rows))
    report = "\n".join(lines)
    Path("benchmark_results.md").write_text(report + "\n", encoding="utf-8")
    print("\n" + report)
    print("Saved: benchmark_results.md")


def _gpu_strategy(name: str):
    factories = {
        "repair": RepairSearch, "direct": DirectSearch, "ising": IsingSearch,
        "spectral": lambda: SpectralSearch(inner_steps=1), "ca": CASearch,
        "walsh": WalshSearch,
    }
    return factories[name]()


def _gpu_worker(name: str, steps: int) -> None:
    strategy = _gpu_strategy(name)
    started = time.perf_counter()
    _, metrics, _ = strategy.search(steps=steps, seed=SEED)
    print(json.dumps({"strategy": name, "backend": os.environ.get("HADAMARD_BACKEND", "auto"),
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
    parser.add_argument("--gpu-compare", action="store_true")
    parser.add_argument("--gpu-worker", choices=GPU_COMPARE_STRATEGIES)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()
    if args.gpu_worker:
        _gpu_worker(args.gpu_worker, args.steps)
    elif args.gpu_compare:
        gpu_compare(args.steps)
    else:
        main(args.timeout)
