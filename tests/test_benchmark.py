"""Contracts for isolated benchmark execution."""
from __future__ import annotations

from benchmark import ORDERS, _cell, benchmark_groups, run_one
from strategies.walsh import WalshSearch


def test_benchmark_worker_returns_a_completed_result() -> None:
    result = run_one(lambda order: WalshSearch(order), 4, 1, 0, 5)
    assert result["status"] == "ok"
    assert result["metrics"]["energy"] == 0


def test_benchmark_worker_reports_a_timeout() -> None:
    result = run_one(lambda order: WalshSearch(order), 668, 1, 0, 0.01)
    assert result["status"] == "timeout"


def test_benchmark_has_nine_strategies_for_every_order() -> None:
    groups = benchmark_groups()
    assert len(groups) == 1
    assert len(groups[0][1]) == 9
    assert len(groups[0][1]) * len(ORDERS) == 54


def test_baumert_inapplicable_orders_render_as_na() -> None:
    assert _cell({"status": "na"}, 8) == "N/A"
