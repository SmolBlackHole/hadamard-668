"""Tests for pure-Python Hadamard verification."""

from __future__ import annotations

import pytest

from src.verify import InvalidMatrix, independent_audit


def _hadamard2() -> list[list[int]]:
    return [[1, 1], [1, -1]]


def test_valid_hadamard_passes() -> None:
    independent_audit(_hadamard2())


def test_non_pm1_value_raises() -> None:
    with pytest.raises(InvalidMatrix, match="contains a value"):
        independent_audit([[1, 0], [1, -1]])


def test_non_orthogonal_rows_raises() -> None:
    with pytest.raises(InvalidMatrix, match="dot product"):
        independent_audit([[1, 1], [1, 1]])
