"""Tests for matrix builders."""

from __future__ import annotations

import numpy as np

from src.builder import build_gs4
from src.verify import verify_candidate


def test_builder_unit_works() -> None:
    sequences = np.ones((4, 1), dtype=np.int8)
    matrix = build_gs4(sequences)
    assert matrix.shape == (4, 4)
    verify_candidate(sequences)


def test_builder_rejects_invalid_shape() -> None:
    with np.testing.assert_raises_regex(ValueError, "shape"):
        build_gs4(np.ones((3, 2), dtype=np.int8))
