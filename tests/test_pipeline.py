"""Tests for sequence generation and the end-to-end execution pipeline."""

from __future__ import annotations

import numpy as np
import pytest

from src.constructions import double_gs4, paley_ng_sequences
from src.generator import exact_sequences, n_from_order
from src.pipeline import execute
from src.solver import SolverConfig
from src.tracker import Tracker
from src.verify import verify_candidate


def test_order_conversion_and_unknown_strategy() -> None:
    assert n_from_order("gs4", 40) == 10
    with pytest.raises(ValueError, match="strategy"):
        n_from_order("nonsense", 40)
    with pytest.raises(ValueError, match="positive order divisible by 4"):
        n_from_order("gs4", 42)


def test_paley_ng_constructs_verified_order_208() -> None:
    result = execute("paley-ng", 52, 1, 123, SolverConfig())
    assert result.solved
    assert result.order == 208
    assert result.steps == 0
    verify_candidate(result.sequences)


def test_paley_ng_is_deterministic_and_matches_ito_prefix() -> None:
    first = exact_sequences("paley-ng", 52)
    second = exact_sequences("paley-ng", 52)
    assert first is not None and second is not None
    assert np.array_equal(first, second)
    assert first[0, :20].tolist() == [
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


@pytest.mark.parametrize("n, message", [(51, "even"), (50, "prime")])
def test_paley_ng_rejects_unsupported_orders(n: int, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        exact_sequences("paley-ng", n)


def test_turyn_double_preserves_gs4_solution() -> None:
    lifted = double_gs4(paley_ng_sequences(52))
    tracker = Tracker()
    tracker.build(lifted)
    assert lifted.shape == (4, 104)
    assert tracker.energy() == 0
    verify_candidate(lifted)


def test_construct_uses_direct_paley_then_turyn() -> None:
    result = execute("construct", 104, 1, 999, SolverConfig())
    assert result.solved
    assert result.steps == 0
    assert np.array_equal(result.sequences, double_gs4(paley_ng_sequences(52)))


def test_construct_searches_smaller_base_then_lifts() -> None:
    result = execute("construct", 14, 2_000, 0, SolverConfig())
    assert result.solved
    assert result.steps > 0


def test_construct_fallback_matches_gs4_for_odd_n() -> None:
    config = SolverConfig()
    direct = execute("gs4", 11, 2_000, 42, config)
    fallback = execute("construct", 11, 2_000, 42, config)
    assert direct.energy == fallback.energy
    assert direct.steps == fallback.steps
    assert np.array_equal(direct.sequences, fallback.sequences)
