"""Tests for matrix builders."""

from __future__ import annotations

import numpy as np

from builders import build_goethals_seidel, build_negacyclic_gs4
from gpu import check_orthogonality


def test_goethals_seidel_unit_works():
    m = check_orthogonality(build_goethals_seidel(*(np.ones(1, dtype=np.int8) for _ in range(4))))
    assert m.energy == 0
    assert m.orthogonal_pairs == 6


def test_negacyclic_gs4_unit_works():
    m = check_orthogonality(build_negacyclic_gs4(*(np.ones(1, dtype=np.int8) for _ in range(4))))
    assert m.energy == 0
    assert m.orthogonal_pairs == 6
