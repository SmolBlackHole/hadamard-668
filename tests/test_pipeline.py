"""Tests for Generator and search logic."""

from __future__ import annotations

from generator import Generator


def test_generator_uses_expected_order() -> None:
    r = Generator(kind="gs4", n=11).search(steps=0, seed=0)
    assert r.matrix.shape == (44, 44)


def test_generator_from_cli_parses_strategy_names() -> None:
    assert Generator.from_cli("gs4", 40).ORDER == 40
    assert Generator.from_cli("golay_2n", 40).ORDER == 40
    assert Generator.from_cli("gs4_group", 40).ORDER == 40
    assert Generator.from_cli("tensor", 240).ORDER == 240  # 16*3*5
