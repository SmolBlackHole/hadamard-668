"""Tests for matrix builders."""

from __future__ import annotations

import numpy as np
import pytest

from builder import Builder
from metrics import check_orthogonality


@pytest.mark.parametrize("kind", ["gs4", "golay_2n", "gs4_group"])
def test_builder_unit_works(kind):
    b = Builder(kind=kind, n=1)
    seqs = np.ones((b.k, 1), dtype=np.int8)
    m = check_orthogonality(b.build(seqs))
    assert m.energy == 0


def test_best_factorization() -> None:
    assert Builder.factorize(15) == [5, 3]
    assert Builder.factorize(7) == [7]  # prime, no split
