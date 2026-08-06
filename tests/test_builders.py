"""Tests for matrix builders."""

from __future__ import annotations

import numpy as np

from src.builder import Builder
from src.metrics import check_orthogonality


def test_builder_unit_works() -> None:
    b = Builder(kind="gs4", n=1)
    seqs = np.ones((b.k, 1), dtype=np.int8)
    m = check_orthogonality(b.build(seqs))
    assert m.energy == 0


def test_best_factorization() -> None:
    assert Builder.factorize(15) == [5, 3]
    assert Builder.factorize(7) == [7]  # prime, no split
