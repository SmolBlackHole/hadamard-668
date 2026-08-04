"""Pytest fixtures for the Hadamard test suite."""
from __future__ import annotations
from fixtures import tt_sequences, tt_lengths
from constructions import sylvester, paley
import pytest

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


@pytest.fixture(scope="session")
def tt8():
    return tt_sequences(8), tt_lengths(8)


@pytest.fixture(scope="session")
def tt36():
    return tt_sequences(36), tt_lengths(36)


@pytest.fixture(scope="session")
def sylvester4():
    return sylvester(4)


@pytest.fixture(scope="session")
def paley12():
    return paley(12)
