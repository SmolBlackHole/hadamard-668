"""Tests for matrix builders and known constructions."""

from __future__ import annotations

import numpy as np
import pytest
from test_constructions_helper import get_known

from builders import build_goethals_seidel, build_turyn
from gpu import check_orthogonality


@pytest.mark.parametrize("order", [1, 2, 4, 8, 12, 16, 20])
def test_known_matrices_are_orthogonal(order: int) -> None:
    matrix = get_known(order)
    metrics = check_orthogonality(matrix)
    assert matrix.shape == (order, order)
    assert metrics.energy == 0
    assert metrics.orthogonal_pairs == order * (order - 1) // 2


def test_goethals_seidel_with_unit_sequences() -> None:
    matrix = build_goethals_seidel(*(np.ones(1, dtype=np.int8) for _ in range(4)))
    m = check_orthogonality(matrix)
    assert m.energy == 0
    assert m.orthogonal_pairs == 6


@pytest.mark.parametrize("n", [1, 3, 5])
def test_goethals_seidel_builds_valid_matrix(n):
    seqs = tuple(
        np.array(np.random.default_rng(i).choice((-1, 1), size=n), dtype=np.int8)
        for i in range(4)
    )
    matrix = build_goethals_seidel(*seqs)
    assert matrix.shape == (4 * n, 4 * n)
    assert matrix.dtype == np.int8
    assert np.all(np.isin(matrix, (-1, 1)))


def test_turyn_type_eight_builds_hadamard():
    tt8 = tuple(
        np.array(s, dtype=np.int8)
        for s in (
            (1, 1, -1, 1, -1, 1, -1, 1),
            (1, -1, -1, -1, -1, -1, -1, 1),
            (1, -1, -1, 1, 1, 1, 1, -1),
            (1, 1, 1, -1, 1, 1, -1),
        )
    )
    matrix = build_turyn(*tt8)
    assert check_orthogonality(matrix).energy == 0
    assert matrix.shape == (92, 92)
