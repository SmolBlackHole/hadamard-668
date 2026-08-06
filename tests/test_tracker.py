"""Tracker invariants against the full GS4 Gram matrix."""

from __future__ import annotations

from itertools import combinations

import numpy as np
import pytest

from src.builder import Builder
from src.tracker import Tracker


def _full_energy(builder: Builder, seqs: np.ndarray) -> int:
    matrix = builder.build(seqs).astype(np.int64)
    gram = matrix @ matrix.T
    np.fill_diagonal(gram, 0)
    return int(np.sum(gram * gram) // 2)


@pytest.mark.parametrize("n", (3, 4, 5, 6, 7))
def test_single_flip_energies_match_full_gram(n: int) -> None:
    builder = Builder(kind="gs4", n=n)
    seqs = np.random.default_rng(n).choice((-1, 1), size=(builder.k, n)).astype(np.int8)
    tracker = Tracker()
    tracker.build(seqs)

    energies = tracker.flip_energies()
    for index, (s, c) in enumerate((s, c) for s in range(4) for c in range(n)):
        flipped = seqs.copy()
        flipped[s, c] *= -1
        assert tracker.flip(s, c) == energies[index] == _full_energy(builder, flipped)


@pytest.mark.parametrize("n", (3, 5, 7))
def test_combo_energies_cover_odd_middle_lag(n: int) -> None:
    builder = Builder(kind="gs4", n=n)
    seqs = np.random.default_rng(20 + n).choice((-1, 1), size=(builder.k, n)).astype(np.int8)
    tracker = Tracker()
    tracker.build(seqs)
    positions = [(s, c) for s in range(4) for c in range(n)]

    for combo in combinations(positions[: min(8, len(positions))], 2):
        flipped = seqs.copy()
        for s, c in combo:
            flipped[s, c] *= -1
        assert tracker.combo_energy(list(combo)) == _full_energy(builder, flipped)

    combo = [(0, 0), (0, n // 2), (0, n - 1)]
    flipped = seqs.copy()
    for s, c in combo:
        flipped[s, c] *= -1
    assert tracker.combo_energy(combo) == _full_energy(builder, flipped)


@pytest.mark.parametrize("n", (4, 5, 8))
def test_combo_energy_matches_full_gram_for_larger_moves(n: int) -> None:
    builder = Builder(kind="gs4", n=n)
    rng = np.random.default_rng(40 + n)
    seqs = rng.choice((-1, 1), size=(builder.k, n)).astype(np.int8)
    tracker = Tracker()
    tracker.build(seqs)
    positions: list[tuple[int, int]] = [(s, c) for s in range(4) for c in range(n)]

    for width in range(2, 6):
        combo = [positions[int(i)] for i in rng.choice(len(positions), size=width, replace=False)]
        flipped = seqs.copy()
        for s, c in combo:
            flipped[s, c] *= -1
        assert tracker.combo_energy(combo) == _full_energy(builder, flipped)

    assert tracker.combo_energy([(0, 0), (0, 0)]) == tracker.energy()


@pytest.mark.parametrize("n", (4, 5, 8))
def test_pair_batch_matches_combo_energy_in_combination_order(n: int) -> None:
    builder = Builder(kind="gs4", n=n)
    rng = np.random.default_rng(60 + n)
    seqs = rng.choice((-1, 1), size=(builder.k, n)).astype(np.int8)
    tracker = Tracker()
    tracker.build(seqs)
    positions: list[tuple[int, int]] = [(s, c) for s in range(4) for c in range(n)]
    candidates: list[tuple[int, int]] = [
        positions[int(i)]
        for i in rng.choice(len(positions), size=min(10, len(positions)), replace=False)
    ]
    energies = tracker.pair_energies(candidates)

    for i, j in combinations(range(len(candidates)), 2):
        combo = [candidates[i], candidates[j]]
        flipped = seqs.copy()
        for s, c in combo:
            flipped[s, c] *= -1
        assert energies[i, j] == tracker.combo_energy(combo) == _full_energy(builder, flipped)


def test_accept_keeps_cache_equal_to_fresh_build() -> None:
    n = 9
    rng = np.random.default_rng(4)
    seqs = rng.choice((-1, 1), size=(4, n)).astype(np.int8)
    tracker = Tracker()
    tracker.build(seqs)

    for _ in range(12):
        tracker.accept(seqs, int(rng.integers(4)), int(rng.integers(n)))
        fresh = Tracker()
        fresh.build(seqs)
        assert tracker.energy() == fresh.energy()
        assert tracker._norm2 is not None and fresh._norm2 is not None
        assert np.array_equal(tracker._norm2, fresh._norm2)
        assert np.array_equal(tracker.flip_energies(), fresh.flip_energies())


@pytest.mark.parametrize("n", (4, 5, 8))
def test_flip_batch_matches_single_flip_energies(n: int) -> None:
    builder = Builder(kind="gs4", n=n)
    seqs = np.random.default_rng(80 + n).choice((-1, 1), size=(builder.k, n)).astype(np.int8)
    tracker = Tracker()
    tracker.build(seqs)

    expected = tracker.flip_energies()
    actual = np.concatenate(
        (tracker.flip_batch(0, n), tracker.flip_batch(n, 3 * n), tracker.flip_batch(3 * n, 4 * n))
    )
    assert np.array_equal(actual, expected)
