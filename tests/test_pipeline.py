"""Tests for Generator and search logic."""

from __future__ import annotations

import pytest

from src.generator import Generator


def test_generator_uses_expected_order() -> None:
    assert Generator(kind="gs4", n=11).order == 44


def test_generator_from_cli_parses_strategy_names() -> None:
    assert Generator.from_cli("gs4", 40).order == 40


def test_tensor_search_smoke() -> None:
    r = Generator.from_cli("tensor", 240).search(steps=5000, seed=42)
    assert r.matrix.shape == (240, 240)
    assert r.metrics.energy == 0


def test_unknown_strategy_raises() -> None:
    with pytest.raises(ValueError, match="unknown"):
        Generator.from_cli("nonsense", 100)
