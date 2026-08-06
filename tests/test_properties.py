"""Property tests for metrics and gram primitives."""

from __future__ import annotations

import numpy as np
import pytest

from metrics import Metrics, check_orthogonality, gram_matrix, metrics_from_gram


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
