"""Tests for Pipeline short-circuit mechanics and stage validation."""

from __future__ import annotations

import numpy as np
import pytest

from gpu import Metrics, check_orthogonality
from strategies.base import Pipeline, Result, SearchStrategy
from strategies.custom import CustomSolver


def test_custom_solver_uses_expected_order() -> None:
    r = CustomSolver(kind="gs4", n=11).search(steps=0, seed=0)
    assert r.matrix.shape == (44, 44)


def test_pipeline_runs_all_stages() -> None:
    class Refining(SearchStrategy):
        ORDER = 32

        @property
        def name(self):
            return "refining"

        def search(self, steps, seed):
            return Result(
                matrix=np.ones((32, 32), dtype=np.int8),
                metrics=Metrics(0, 0, 0),
                elapsed=0.0,
                seed=seed,
            )

        def refine(self, matrix, steps, seed, sequences=None):
            return Result(
                matrix=np.ones((32, 32), dtype=np.int8),
                metrics=Metrics(0, 0, 0),
                elapsed=0.0,
                seed=seed,
            )

    r = Pipeline(
        [
            (Refining(), 0),
            (Refining(), 0),
        ]
    ).search(steps=0, seed=0)
    assert r.matrix.shape == (32, 32)


def test_pipeline_short_circuits_after_exact_verification() -> None:
    calls: list[int] = []

    class Source(SearchStrategy):
        ORDER = 4

        @property
        def name(self) -> str:
            return "source"

        def search(self, steps: int, seed: int):
            return Result(
                matrix=np.ones((4, 4), dtype=np.int8),
                metrics=Metrics(0, 6, 0),
                elapsed=0.0,
                seed=seed,
            )

    class Sink(SearchStrategy):
        ORDER = 4

        @property
        def name(self) -> str:
            return "sink"

        def search(self, steps: int, seed: int):
            raise AssertionError("later stages use refine")

        def refine(self, matrix, steps, seed, sequences=None):
            calls.append(seed)
            candidate = np.array(
                [[1, 1, 1, 1], [1, -1, 1, -1], [1, 1, -1, -1], [1, -1, -1, 1]], dtype=np.int8
            )
            return Result(
                matrix=candidate, metrics=check_orthogonality(candidate), elapsed=0.0, seed=seed
            )

    r = Pipeline([(Source(), 1), (Sink(), 1)]).search(0, 10)
    assert calls == [11]
    assert r.metrics.energy == 0


def test_pipeline_rejects_non_refining_followup() -> None:
    class NoRefine(SearchStrategy):
        ORDER = 8

        @property
        def name(self):
            return "norefine"

        def search(self, steps, seed):
            return Result(np.ones((8, 8), dtype=np.int8), Metrics(0, 0, 0), 0.0, seed)

    with pytest.raises(ValueError, match="cannot refine"):
        Pipeline([(NoRefine(), 1), (NoRefine(), 1)])
