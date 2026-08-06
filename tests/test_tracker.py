"""Tests for GramTracker invariants."""

from __future__ import annotations

import numpy as np

from builder import Builder
from tracker import GramTracker


def test_flip_restores_sequences() -> None:
    b = Builder(kind="gs4", n=4)
    seqs = np.random.default_rng(1).choice((-1, 1), size=(b.k, 4)).astype(np.int8)
    original = seqs.copy()

    tracker = GramTracker(b.build)
    tracker.build(seqs)
    tracker.flip(seqs, 0, 2)

    assert np.array_equal(seqs, original)


def test_flip_batch_restores_sequences() -> None:
    b = Builder(kind="golay_2n", n=6)
    seqs = np.random.default_rng(2).choice((-1, 1), size=(b.k, 6)).astype(np.int8)
    original = seqs.copy()

    tracker = GramTracker(b.build)
    tracker.build(seqs)
    flips = [(0, 0), (1, 3), (0, 5)]
    tracker.flip_batch(seqs, flips)

    assert np.array_equal(seqs, original)


def test_delta_energy_matches_full_recompute() -> None:
    b = Builder(kind="gs4", n=5)
    seqs = np.random.default_rng(3).choice((-1, 1), size=(b.k, 5)).astype(np.int8)
    tracker = GramTracker(b.build)
    tracker.build(seqs)

    for _ in range(10):
        s = np.random.randint(0, b.k)
        c = np.random.randint(0, b.n)
        seqs_copy = seqs.copy()
        seqs_copy[s, c] *= -1
        full_M = b.build(seqs_copy)
        _, full_e = tracker._gram_energy(full_M)
        delta_e = tracker._delta_energy(full_M)
        assert full_e == delta_e


def test_accept_many_equals_fresh_build() -> None:
    b = Builder(kind="golay_2n", n=6)
    seqs = np.random.default_rng(4).choice((-1, 1), size=(b.k, 6)).astype(np.int8)
    tracker = GramTracker(b.build)
    tracker.build(seqs)

    flips = [(0, 1), (1, 4)]
    tracker.accept_many(seqs, flips)
    e_after_accept = tracker.energy()

    tracker2 = GramTracker(b.build)
    tracker2.build(seqs)
    e_fresh = tracker2.energy()

    assert e_after_accept == e_fresh
