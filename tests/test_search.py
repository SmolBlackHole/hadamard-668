"""Smoke tests against known matrices and the search-engine API."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from builders import build_goethals_seidel
from constructions import get_known, sylvester
from gpu import check_orthogonality
from run import derive_seeds, select_best_run, worker_count
from strategies.base import Pipeline, Result, SearchStrategy
from strategies.base import RunResult as RR
from strategies.kflip import KFlipRepair


@pytest.mark.parametrize("order", [1, 2, 4, 8, 12, 16, 20])
def test_known_matrices_are_orthogonal(order: int) -> None:
    matrix = get_known(order)
    metrics = check_orthogonality(matrix)
    assert matrix.shape == (order, order)
    assert metrics["energy"] == 0
    assert metrics["orthogonal_pairs"] == order * (order - 1) // 2


def test_goethals_seidel_builds_order_four_hadamard() -> None:
    matrix = build_goethals_seidel(
        *(np.ones(1, dtype=np.int8) for _ in range(4)))
    assert check_orthogonality(matrix) == {
        "energy": 0,
        "orthogonal_pairs": 6,
        "max_abs_correlation": 0,
    }


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


def test_run_writes_each_parallel_seed_separately(tmp_path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "run.py",
            "--strategy",
            "kflip",
            "--steps",
            "0",
            "--seed",
            "7",
            "--runs",
            "2",
            "--workers",
            "2",
            "--runs-dir",
            str(tmp_path),
        ],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    outputs = sorted(tmp_path.iterdir())
    assert len(outputs) == 2
    assert all((directory / "matrix.csv").is_file() for directory in outputs)
    assert all((directory / "run.json").is_file() for directory in outputs)
    assert "Run summary" in result.stdout
    assert "best seed=" in result.stdout


def test_turyn_search_uses_the_expected_order() -> None:
    matrix, _, _, _ = KFlipRepair(n=8).search(steps=0, seed=0)
    assert matrix.shape == (92, 92)


def test_real_pipeline_runs_all_stages() -> None:
    stages: list[tuple[SearchStrategy, int]] = [
        (KFlipRepair(n=8), 0),
        (KFlipRepair(n=8), 0),
    ]
    _, metrics, _, _ = Pipeline(stages).search(steps=0, seed=0)
    assert metrics["energy"] >= 0


def test_pipeline_short_circuits_only_after_exact_verification() -> None:
    calls: list[int] = []

    class Source(SearchStrategy):
        ORDER = 4

        @property
        def name(self) -> str:
            return "source"

        def search(self, steps: int, seed: int):
            return Result(
                np.ones((4, 4), dtype=np.int8),
                {
                    "energy": 0,
                    "orthogonal_pairs": 6,
                    "max_abs_correlation": 0,
                },
                0.0,
            )

    class Sink(SearchStrategy):
        ORDER = 4

        @property
        def name(self) -> str:
            return "sink"

        def search(self, steps: int, seed: int):
            raise AssertionError("later stages use refine")

        def refine(
            self, matrix: np.ndarray, steps: int, seed: int, sequences: np.ndarray | None = None
        ):
            calls.append(seed)
            candidate = sylvester(4)
            return Result(candidate, check_orthogonality(candidate), 0.0)

    _, metrics, _, _ = Pipeline([(Source(), 1), (Sink(), 1)]).search(0, 10)

    assert calls == [11]
    assert metrics["energy"] == 0


def test_pipeline_rejects_non_refining_followup() -> None:
    with pytest.raises(ValueError, match="cannot refine"):
        from strategies.pocs import PocsSearch
        Pipeline(
            [
                (PocsSearch(n=8), 1),
                (PocsSearch(n=8), 1),
            ]
        )


def test_parallel_run_helpers_are_deterministic_and_gpu_safe() -> None:
    assert derive_seeds(40, 3) == [40, 41, 42]
    assert worker_count("kflip", runs=3, workers=8) == 3
    assert worker_count("kflip:10,kflip:5", runs=3, workers=2) == 2
    assert worker_count("pocs", runs=3, workers=3) == 1
    assert worker_count("spectral_descent", runs=3, workers=3) == 1

    matrix = sylvester(4)
    results = [
        RR(seed=40, matrix=matrix, energy=8,
           orthogonal_pairs=0, max_off_diagonal=0, wall=0.2),
        RR(seed=41, matrix=matrix, energy=0,
           orthogonal_pairs=0, max_off_diagonal=0, wall=0.3),
    ]
    assert select_best_run(results).seed == 41
