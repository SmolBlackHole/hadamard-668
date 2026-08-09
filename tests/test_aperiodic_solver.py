"""Search and construction tests for weighted aperiodic sequence families."""

from __future__ import annotations

import numpy as np

from src.aperiodic_constructions import base_to_gs4, random_bs, random_tt, tt_to_gs4
from src.aperiodic_solver import AperiodicSolverConfig, search
from src.base_tracker import BaseTracker
from src.builder import Builder
from src.metrics import check_orthogonality
from src.tracker import Tracker

TT_WEIGHTS = (1, 1, 2, 2)


def test_random_bs_is_gauge_normalized_and_satisfies_cheap_constraints() -> None:
    sequences = random_bs(9, 8, np.random.default_rng(42))
    a, b, c, d = sequences
    assert tuple(len(sequence) for sequence in sequences) == (9, 9, 8, 8)
    assert [int(sequence[0]) for sequence in sequences] == [1, 1, 1, 1]
    assert int(a[-1]) == -int(b[-1])
    assert all(np.all(np.abs(sequence) == 1) for sequence in sequences)
    assert sum(int(sequence.sum()) ** 2 for sequence in sequences) == 34
    assert (
        int(a @ np.where(np.arange(9) % 2 == 0, 1, -1)) ** 2
        + int(b @ np.where(np.arange(9) % 2 == 0, 1, -1)) ** 2
        + int(c @ np.where(np.arange(8) % 2 == 0, 1, -1)) ** 2
        + int(d @ np.where(np.arange(8) % 2 == 0, 1, -1)) ** 2
        == 34
    )


def test_known_bs_2_1_embeds_into_valid_hadamard() -> None:
    sequences = tuple(np.asarray(values, dtype=np.int8) for values in ((1, 1), (1, -1), (1,), (1,)))
    aperiodic = BaseTracker()
    aperiodic.build(sequences)
    gs4_sequences = base_to_gs4(sequences)
    gs4 = Tracker()
    gs4.build(gs4_sequences)
    matrix = Builder(kind="gs4", n=3).build(gs4_sequences)

    assert aperiodic.energy() == 0
    assert gs4.energy() == 0
    assert check_orthogonality(matrix).energy == 0


def test_arbitrary_bs_n_plus_1_n_preserves_scaled_q_in_gs4() -> None:
    sequences = random_bs(9, 8, np.random.default_rng(7), sum_filter=False)
    aperiodic = BaseTracker()
    aperiodic.build(sequences)
    gs4 = Tracker()
    gs4.build(base_to_gs4(sequences))

    assert aperiodic.energy() % 4 == 0
    assert gs4.energy() // (64 * 17) == aperiodic.energy() // 4


def test_random_tt_is_gauge_normalized_and_satisfies_top_lag() -> None:
    sequences = random_tt(12, np.random.default_rng(42))
    a, b, c, d = sequences
    assert tuple(len(sequence) for sequence in sequences) == (12, 12, 12, 11)
    assert [int(sequence[0]) for sequence in sequences] == [1, 1, 1, 1]
    assert int(a[0] * a[-1]) == int(b[0] * b[-1])
    assert int(a[0] * a[-1]) == -int(c[0] * c[-1])
    assert np.all(np.abs(d) == 1)
    alternating_n = np.where(np.arange(12) % 2 == 0, 1, -1)
    alternating_d = alternating_n[:-1]
    assert (
        int(a.sum()) ** 2 + int(b.sum()) ** 2 + 2 * int(c.sum()) ** 2 + 2 * int(d.sum()) ** 2 == 70
    )
    assert (
        int(a @ alternating_n) ** 2
        + int(b @ alternating_n) ** 2
        + 2 * int(c @ alternating_n) ** 2
        + 2 * int(d @ alternating_d) ** 2
        == 70
    )


def test_known_tt_2_embeds_into_valid_hadamard() -> None:
    sequences = tuple(
        np.asarray(values, dtype=np.int8) for values in ((-1, -1), (-1, -1), (-1, 1), (-1,))
    )
    aperiodic = BaseTracker()
    aperiodic.build(sequences, weights=TT_WEIGHTS)
    gs4_sequences = tt_to_gs4(sequences)
    gs4 = Tracker()
    gs4.build(gs4_sequences)
    matrix = Builder(kind="gs4", n=5).build(gs4_sequences)

    assert aperiodic.energy() == 0
    assert gs4.energy() == 0
    assert check_orthogonality(matrix).energy == 0


def test_greedy_solves_small_tt_without_tabu() -> None:
    rng = np.random.default_rng(8)
    tracker = BaseTracker()
    tracker.build(random_tt(4, rng, sum_filter=False), weights=TT_WEIGHTS)
    result = search(
        tracker,
        rng,
        steps=1_000,
        config=AperiodicSolverConfig(tabu=False),
    )
    assert result.energy == 0
    assert result.stats.greedy_accepts > 0
    assert result.stats.tabu_walks == 0


def test_tabu_solves_plateau_and_leaves_exact_tracker() -> None:
    rng = np.random.default_rng(4)
    tracker = BaseTracker()
    tracker.build(random_tt(4, rng, sum_filter=False), weights=TT_WEIGHTS)
    result = search(tracker, rng, steps=1_000)

    rebuilt = BaseTracker()
    rebuilt.build(result.sequences, weights=TT_WEIGHTS)
    assert result.energy == 0
    assert result.stats.tabu_walks > 0
    assert result.stats.tabu_improvements > 0
    assert result.stats.evaluations <= 1_000
    assert tracker.energy() == rebuilt.energy()
    assert np.array_equal(tracker.residual(), rebuilt.residual())
    assert np.array_equal(tracker.flip_qs(), rebuilt.flip_qs())
