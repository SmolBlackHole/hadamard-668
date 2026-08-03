"""Contracts for isolated benchmark execution."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import benchmark
from benchmark import ORDERS, _cell, benchmark_groups, run_one, steps_for_order
from strategies.circulant import TurynGreedySearch


def test_benchmark_worker_returns_a_completed_result() -> None:
    result = run_one(
        lambda order: TurynGreedySearch(n=8), 92, 1, 0, 5)
    assert result["status"] == "ok"
    assert result["metrics"]["energy"] >= 0
    assert result["algorithm_seconds"] >= 0
    assert result["wall_seconds"] >= result["algorithm_seconds"]


def test_benchmark_worker_reports_a_timeout() -> None:
    result = run_one(lambda order: TurynGreedySearch(), 668, 1, 0, 0.01)
    assert result["status"] == "timeout"


def test_benchmark_has_seven_strategies_for_every_order() -> None:
    groups = benchmark_groups()
    assert len(groups) == 1
    assert len(groups[0][1]) == 7
    assert len(groups[0][1]) * len(ORDERS) == 42
    assert [steps_for_order(order) for order in ORDERS] == [
        2_000, 2_000, 2_000, 2_000, 2_000, 5_000]


def test_full_benchmark_adds_experimental_strategies_and_pipelines() -> None:
    groups = benchmark_groups(include_all=True)
    assert [name for name, _ in groups] == [
        "Individual strategies", "Experimental strategies", "Pipelines"]
    assert sum(len(strategies) for _, strategies in groups) > 7


def test_benchmark_help_works_without_pythonpath() -> None:
    root = Path(__file__).parents[1]
    result = subprocess.run(
        [sys.executable, "benchmark.py", "--help"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "--all" in result.stdout


def test_turyn_inapplicable_orders_render_as_na() -> None:
    assert _cell({"status": "na"}, 8) == "N/A"


def test_benchmark_cell_has_unambiguous_metrics() -> None:
    result = {
        "status": "ok",
        "algorithm_seconds": 1.25,
        "metrics": {"energy": 16, "orthogonal_pairs": 3},
    }
    cell = _cell(result, 4)
    assert cell == "OK; e=16; rms=1.63; orth=3/6; algo=1.2s"
    assert " | " not in cell


def test_benchmark_writes_one_json_record_per_case(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(benchmark, "ORDERS", (4, 8))
    monkeypatch.setattr(
        benchmark,
        "benchmark_groups",
        lambda **_kwargs: (("Test", (("Other", lambda _order: None),)),),
    )
    monkeypatch.setattr(
        benchmark,
        "run_one",
        lambda *_args: {"status": "ok", "backend": "numpy",
                        "algorithm_seconds": 0.1, "wall_seconds": 0.2,
                        "correlation_histogram": {"0": 6},
                        "metrics": {"energy": 0, "orthogonal_pairs": 6,
                                    "max_abs_correlation": 0}},
    )

    benchmark.main(timeout_seconds=1, include_all=True)

    report = json.loads((tmp_path / "benchmark_results.json").read_text())
    assert report["include_all"]
    assert len(report["cases"]) == 2
    assert report["cases"][0]["metrics"]["energy"] == 0
    assert report["cases"][0]["algorithm_seconds"] == 0.1
    assert report["cases"][0]["correlation_histogram"] == {"0": 6}
    assert report["cases"][1]["status"] == "ok"
