"""Smoke tests against known matrices and the search-engine API."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from constructions import autocorrelation_energy, build_williamson, symmetric_circulant
from gpu import check_orthogonality
from strategies.annealing import AnnealingSearch
from strategies.base import Pipeline
from strategies.circulant import CirculantSearch
from strategies.repair import RepairSearch
from strategies.rowwise import RowwiseSearch
from verifier.known import get_known, paley, sylvester


@pytest.mark.parametrize("order", [1, 2, 4, 8, 12, 16, 20])
def test_known_matrices_are_orthogonal(order: int) -> None:
    matrix = get_known(order)
    metrics = check_orthogonality(matrix)
    assert matrix.shape == (order, order)
    assert metrics["energy"] == 0
    assert metrics["orthogonal_pairs"] == order * (order - 1) // 2


@pytest.mark.parametrize(("half", "expected"), [
    ([1], [1]),
    ([1, -1], [1, -1, -1]),
    ([1, -1, 1], [1, -1, 1, 1, -1]),
    ([1, -1, 1, -1], [1, -1, 1, -1, -1, 1, -1]),
])
def test_symmetric_circulant_examples(
    half: list[int], expected: list[int],
) -> None:
    assert np.array_equal(
        symmetric_circulant(np.array(half, dtype=np.int8)),
        np.array(expected, dtype=np.int8),
    )


def test_williamson_builds_order_four_hadamard() -> None:
    matrix = build_williamson(*(np.ones(1, dtype=np.int8) for _ in range(4)))
    assert check_orthogonality(matrix) == {
        "energy": 0, "orthogonal_pairs": 6, "max_abs_correlation": 0}


def test_autocorrelation_energy_of_singletons_is_zero() -> None:
    sequence = np.array([1], dtype=np.int8)
    assert autocorrelation_energy(
        (sequence, sequence, sequence, sequence)) == 0


def test_run_help_works_without_pythonpath() -> None:
    environment = os.environ.copy()
    environment.pop("PYTHONPATH", None)
    result = subprocess.run(
        [sys.executable, "run.py", "--help"],
        cwd=Path(__file__).parents[1],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("order", [4, 8, 12])
def test_rowwise_solves_small_known_orders(order: int) -> None:
    _, metrics, _ = RowwiseSearch(order).search(steps=1_000, seed=0)
    assert metrics["energy"] == 0


def test_circulant_search_solves_order_four() -> None:
    _, metrics, _ = CirculantSearch(
        ORDER=4, K=1, HALF=1).search(steps=100, seed=0)
    assert metrics["energy"] == 0


def test_real_pipeline_runs_all_stages() -> None:
    stages = [
        (CirculantSearch(ORDER=12, K=3, HALF=2), 0),
        (AnnealingSearch(ORDER=12, K=3, HALF=2), 0),
        (RepairSearch(order=12), 0),
    ]
    _, metrics, _ = Pipeline(stages).search(steps=0, seed=0)
    assert metrics["energy"] >= 0


@pytest.mark.parametrize("matrix", [sylvester(4), paley(8), paley(12)])
def test_known_generators_only_emit_signs(matrix: np.ndarray) -> None:
    assert np.all(np.isin(matrix, (-1, 1)))
