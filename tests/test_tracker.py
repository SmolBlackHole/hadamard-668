"""Tests for Tracker invariants."""

from __future__ import annotations

import numpy as np

from builder import Builder
from tracker import Tracker


def test_flip_restores_sequences() -> None:
    b = Builder(kind="gs4", n=4)
    seqs = np.random.default_rng(1).choice((-1, 1), size=(b.k, 4)).astype(np.int8)
    original = seqs.copy()

    tracker = Tracker()
    tracker.build(seqs)
    tracker.flip(seqs, 0, 2)

    assert np.array_equal(seqs, original)


def test_delta_energy_matches_full_recompute() -> None:
    b = Builder(kind="gs4", n=5)
    seqs = np.random.default_rng(3).choice((-1, 1), size=(b.k, 5)).astype(np.int8)
    tracker = Tracker()
    tracker.build(seqs)

    for _ in range(10):
        s = np.random.randint(0, b.k)
        c = np.random.randint(0, b.n)
        # flip energy via tracker
        ef = tracker.flip(seqs, s, c)
        # fresh compute via builder + gram
        seqs_copy = seqs.copy()
        seqs_copy[s, c] *= -1
        M = b.build(seqs_copy).astype(np.int64)
        G = M @ M.T
        np.fill_diagonal(G, 0)
        full_e = int(np.sum(G * G)) // 2
        assert full_e == ef, f"Mismatch: {full_e} vs {ef}"


def test_accept_equals_fresh_build() -> None:
    b = Builder(kind="gs4", n=6)
    seqs = np.random.default_rng(4).choice((-1, 1), size=(b.k, 6)).astype(np.int8)
    tracker = Tracker()
    tracker.build(seqs)

    tracker.accept(seqs, 0, 1)
    e_after = tracker.energy()

    tracker2 = Tracker()
    tracker2.build(seqs)
    e_fresh = tracker2.energy()

    assert e_after == e_fresh
