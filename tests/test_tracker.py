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


@pytest.mark.parametrize("n", (6, 8, 38, 52))
def test_even_tracker_omits_dead_midpoint_lag(n: int) -> None:
    builder = Builder(kind="gs4", n=n)
    seqs = np.random.default_rng(100 + n).choice((-1, 1), size=(builder.k, n)).astype(np.int8)
    tracker = Tracker()
    tracker.build(seqs)

    residual = Tracker._compute_residual(seqs)
    width = (n - 1) // 2

    assert residual[n // 2] == 0
    assert tracker._u is not None and tracker._u.shape == (width,)
    assert tracker._delta is not None and tracker._delta.shape == (4 * n, width)
    assert tracker.energy() == _full_energy(builder, seqs)


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


@pytest.mark.parametrize("n", (38, 49, 50, 51, 52))
def test_combo_energy_matches_fresh_tracker_for_wide_moves(n: int) -> None:
    rng = np.random.default_rng(668 + n)
    sequences = rng.choice((-1, 1), size=(4, n)).astype(np.int8)
    tracker = Tracker()
    tracker.build(sequences)
    positions = np.asarray([(s, c) for s in range(4) for c in range(n)], dtype=np.intp)

    for width in (2, 3, 7, 11, 16, 20):
        chosen = positions[rng.choice(len(positions), size=width, replace=False)]
        combo = [(int(s), int(c)) for s, c in chosen]
        flipped = sequences.copy()
        flipped[chosen[:, 0], chosen[:, 1]] *= -1
        fresh = Tracker()
        fresh.build(flipped)
        assert tracker.combo_energy(combo) == fresh.energy()


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
    assert np.array_equal(tracker.flip_qs() * (64 * n), expected)


@pytest.mark.parametrize("n", (6, 10, 38, 52))
def test_even_half_length_fold_matches_naf_residual(n: int) -> None:
    rng = np.random.default_rng(200 + n)
    seqs = rng.choice((-1, 1), size=(4, n)).astype(np.int8)
    residual = Tracker._compute_residual(seqs)
    h = n // 2
    omega = np.exp(1j * np.pi / n)

    q = seqs[:, :h] + 1j * seqs[:, h:]
    y = q * omega ** np.arange(h)
    offsets = (np.arange(h)[:, None] + np.arange(h)[None, :]) % h
    correlation = np.sum(y[:, :, None] * np.conj(y[:, offsets]), axis=(0, 1))
    expected = np.asarray(
        [omega ** (-t) * (residual[t] + 1j * residual[h - t]) for t in range(1, h)]
    )

    assert np.allclose(correlation[1:], expected, atol=1e-11)
    tracker = Tracker()
    tracker.build(seqs)
    assert np.isclose(tracker._q, np.sum(np.abs(correlation[1:]) ** 2) / 32)


def test_delta_gram_does_not_determine_solving_single_rows() -> None:
    no_solving_single = np.asarray(
        (
            (-1, -1, -1, -1, -1),
            (-1, -1, -1, -1, 1),
            (1, 1, -1, 1, 1),
            (1, -1, 1, -1, 1),
        ),
        dtype=np.int8,
    )
    six_solving_singles = np.asarray(
        (
            (-1, -1, -1, -1, -1),
            (-1, -1, -1, 1, -1),
            (-1, -1, -1, 1, -1),
            (-1, -1, -1, 1, -1),
        ),
        dtype=np.int8,
    )
    trackers = [Tracker(), Tracker()]
    trackers[0].build(no_solving_single)
    trackers[1].build(six_solving_singles)

    u0, u1 = trackers[0]._u, trackers[1]._u
    delta0, delta1 = trackers[0]._delta, trackers[1]._delta
    assert u0 is not None and u1 is not None
    assert delta0 is not None and delta1 is not None
    assert np.array_equal(u0, u1)
    assert np.array_equal(delta0.T @ delta0, delta1.T @ delta1)
    assert np.count_nonzero(trackers[0].flip_qs() == 0) == 0
    assert np.count_nonzero(trackers[1].flip_qs() == 0) == 6
