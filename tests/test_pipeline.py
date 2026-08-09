"""Tests for Generator and search logic."""

from __future__ import annotations

import numpy as np
import pytest

from src.builder import Builder
from src.constructions import double_gs4, paley_ng_sequences
from src.generator import Generator
from src.metrics import check_orthogonality
from src.tracker import Tracker


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


def test_paley_ng_constructs_hadamard_order_208() -> None:
    result = Generator(kind="paley-ng", n=52).search(steps=1, seed=123)
    assert result.sequences is not None
    tracker = Tracker()
    tracker.build(result.sequences)
    assert tracker.energy() == 0
    assert result.solver_e == 0
    assert result.iterations == 0
    assert result.stats is None
    assert result.matrix.shape == (208, 208)
    assert check_orthogonality(result.matrix).energy == 0


def test_paley_ng_is_deterministic_and_matches_ito_prefix() -> None:
    generator = Generator.from_cli("paley-ng", 208)
    first = generator.search(steps=1, seed=1)
    second = generator.search(steps=999_999, seed=999)
    assert generator.name == "paley-ng"
    assert first.sequences is not None
    assert second.sequences is not None
    assert (first.sequences == second.sequences).all()
    assert first.sequences[0, :20].tolist() == [
        -1,
        -1,
        -1,
        -1,
        1,
        -1,
        1,
        -1,
        -1,
        -1,
        -1,
        1,
        -1,
        1,
        1,
        -1,
        1,
        1,
        -1,
        -1,
    ]
    assert first.sequences[1, :20].tolist() == [
        -1,
        -1,
        1,
        1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        -1,
        1,
        -1,
        1,
        1,
        1,
        -1,
        1,
        -1,
        1,
    ]


@pytest.mark.parametrize("n, message", [(51, "even"), (50, "prime")])
def test_paley_ng_rejects_unsupported_orders(n: int, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        Generator(kind="paley-ng", n=n)


@pytest.mark.parametrize("order", [0, 210])
def test_paley_ng_rejects_invalid_cli_order(order: int) -> None:
    with pytest.raises(ValueError, match="positive order divisible by 4"):
        Generator.from_cli("paley-ng", order)


def test_turyn_double_preserves_gs4_solution() -> None:
    source = paley_ng_sequences(52)
    lifted = double_gs4(source)
    tracker = Tracker()
    tracker.build(lifted)
    assert lifted.dtype == np.int8
    assert lifted.shape == (4, 104)
    assert tracker.energy() == 0
    assert check_orthogonality(Builder(kind="gs4", n=104).build(lifted)).energy == 0


def test_construct_uses_direct_paley_then_turyn() -> None:
    result = Generator(kind="construct", n=104).search(steps=1, seed=999)
    assert result.sequences is not None
    assert result.metrics.energy == 0
    assert result.iterations == 0
    assert result.stats is None
    assert np.array_equal(result.sequences, double_gs4(paley_ng_sequences(52)))


def test_construct_searches_smaller_base_then_lifts() -> None:
    result = Generator(kind="construct", n=14).search(steps=2000, seed=0)
    assert result.sequences is not None
    assert result.metrics.energy == 0
    assert result.iterations > 0
    assert result.stats is not None
    tracker = Tracker()
    tracker.build(result.sequences)
    assert tracker.energy() == 0


def test_construct_fallback_matches_gs4_for_odd_n() -> None:
    direct = Generator(kind="gs4", n=11).search(steps=2000, seed=42)
    fallback = Generator(kind="construct", n=11).search(steps=2000, seed=42)
    assert direct.sequences is not None
    assert fallback.sequences is not None
    assert direct.solver_e == fallback.solver_e
    assert direct.iterations == fallback.iterations
    assert np.array_equal(direct.sequences, fallback.sequences)
