"""Property tests for metrics, gram primitives, and pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from gpu import Metrics, check_orthogonality, gram_matrix, metrics_from_gram
from strategies.base import Pipeline, Result, SearchStrategy


def _reference_metrics(matrix: np.ndarray) -> Metrics:
    gram = matrix.astype(np.int64) @ matrix.astype(np.int64).T
    upper = gram[np.triu_indices(matrix.shape[0], k=1)]
    return Metrics(
        energy=int(np.dot(upper, upper)),
        orthogonal_pairs=int(np.count_nonzero(upper == 0)),
        max_abs_correlation=int(np.abs(upper).max(initial=0)),
    )


def _hadamard2():
    return np.array([[1, 1], [1, -1]], dtype=np.int8)


def _hadamard_sylvester4():
    h2 = _hadamard2()
    return np.block([[h2, h2], [h2, -h2]])


@pytest.mark.parametrize(
    "matrix",
    [
        _hadamard2(),
        _hadamard_sylvester4(),
        np.array([[1, 1, 1], [1, -1, 1], [1, 1, -1]], dtype=np.int8),
    ],
)
def test_metrics_match_independent_reference(matrix):
    assert check_orthogonality(matrix) == _reference_metrics(matrix)


@pytest.mark.parametrize(
    "matrix",
    [
        _hadamard2(),
        _hadamard_sylvester4(),
        np.ones((3, 3), dtype=np.int8),
    ],
)
def test_gram_primitives_match_independent_reference(matrix):
    gram = gram_matrix(matrix)
    assert metrics_from_gram(gram) == _reference_metrics(matrix)


def test_pipeline_passes_matrix_and_incremented_seed():
    class Source(SearchStrategy):
        ORDER = 4

        @property
        def name(self):
            return "source"

        def search(self, steps, seed):
            return Result(matrix=_hadamard2(), metrics=Metrics(1, 0, 2), elapsed=0.1, seed=seed)

    class Sink(SearchStrategy):
        ORDER = 4

        @property
        def name(self):
            return "sink"

        def search(self, steps, seed):
            raise AssertionError

        def refine(self, matrix, steps, seed, sequences=None):
            return Result(matrix=matrix, metrics=Metrics(1, 0, 2), elapsed=0.2, seed=seed)

    r = Pipeline([(Source(), 3), (Sink(), 5)]).search(steps=0, seed=20)
    assert r.metrics.energy == 1
