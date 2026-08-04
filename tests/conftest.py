"""Fixtures for the Hadamard test suite."""
from __future__ import annotations

import numpy as np
import pytest

sys_path = __import__("sys").path
from pathlib import Path

sys_path.insert(0, str(Path(__file__).parent / "src"))

from fixtures import (
    TT8_SEQUENCES, TT36_SEQUENCES, TT36_HEX,
    tt8_sequences, tt36_sequences,
    turyn_lengths, turyn_order,
    sylvester, paley,
)


@pytest.fixture(scope="session")
def tt8_arrays():
    """TT(8) sequences + lengths for order 92."""
    return tt8_sequences(), turyn_lengths(8)


@pytest.fixture(scope="session")
def tt36_arrays():
    """TT(36) sequences + lengths for order 428."""
    return tt36_sequences(), turyn_lengths(36)


@pytest.fixture(scope="session")
def sylvester4():
    """Sylvester Hadamard of order 4."""
    return sylvester(4)


@pytest.fixture(scope="session")
def paley12():
    """Paley Hadamard of order 12."""
    return paley(12)
