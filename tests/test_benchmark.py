"""Contracts for isolated benchmark execution."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from benchmark import ORDERS, _cell, benchmark_groups, run_one, steps_for_order
from strategies.circulant import CirculantSearch


def test_benchmark_worker_returns_a_completed_result() -> None:
    result = run_one(
        lambda order: CirculantSearch(ORDER=order, K=order // 4, HALF=(order // 4 + 1) // 2),
        4, 1, 0, 5)
    assert result["status"] == "ok"
    assert result["metrics"]["energy"] == 0


def test_benchmark_worker_reports_a_timeout() -> None:
    result = run_one(lambda order: CirculantSearch(), 668, 1, 0, 0.01)
    assert result["status"] == "timeout"


def test_benchmark_has_eight_strategies_for_every_order() -> None:
    groups = benchmark_groups()
    assert len(groups) == 1
    assert len(groups[0][1]) == 8
    assert len(groups[0][1]) * len(ORDERS) == 48
    assert [steps_for_order(order) for order in ORDERS] == [
        2_000, 2_000, 2_000, 2_000, 2_000, 5_000]


def test_full_benchmark_adds_experimental_strategies_and_pipelines() -> None:
    groups = benchmark_groups(include_all=True)
    assert [name for name, _ in groups] == [
        "Individual strategies", "Experimental strategies", "Pipelines"]
    assert sum(len(strategies) for _, strategies in groups) > 8


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


def test_baumert_inapplicable_orders_render_as_na() -> None:
    assert _cell({"status": "na"}, 8) == "N/A"


def test_benchmark_cell_has_unambiguous_metrics() -> None:
    result = {
        "status": "ok",
        "seconds": 1.25,
        "metrics": {"energy": 16, "orthogonal_pairs": 3},
    }
    cell = _cell(result, 4)
    assert cell == "OK; e=16; rms=1.63; orth=3/6; t=1.2s"
    assert " | " not in cell
