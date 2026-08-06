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
    tracker.build(seqs, band_rows=b.band_rows, band_cols=b.band_cols)
    tracker.flip(seqs, 0, 2)

    assert np.array_equal(seqs, original)


def test_delta_energy_matches_full_recompute() -> None:
    b = Builder(kind="gs4", n=5)
    seqs = np.random.default_rng(3).choice((-1, 1), size=(b.k, 5)).astype(np.int8)
    tracker = GramTracker(b.build)
    tracker.build(seqs, band_rows=b.band_rows, band_cols=b.band_cols)

    for _ in range(10):
        s = np.random.randint(0, b.k)
        c = np.random.randint(0, b.n)
        seqs_copy = seqs.copy()
        seqs_copy[s, c] *= -1
        full_M = b.build(seqs_copy)
        _, full_e = tracker._gram_energy(full_M)

        assert tracker.M is not None
        dM = np.subtract(full_M, tracker.M, dtype=np.int16)
        rn, cn = np.nonzero(dM)
        delta_e = tracker._compute_delta(rn.astype(np.int16), cn.astype(np.int16))
        assert full_e == delta_e


def test_accept_equals_fresh_build() -> None:
    b = Builder(kind="golay_2n", n=6)
    seqs = np.random.default_rng(4).choice((-1, 1), size=(b.k, 6)).astype(np.int8)
    tracker = GramTracker(b.build)
    tracker.build(seqs, band_rows=b.band_rows, band_cols=b.band_cols)

    tracker.accept(seqs, 0, 1)
    e_after_accept = tracker.energy()

    tracker2 = GramTracker(b.build)
    tracker2.build(seqs)
    e_fresh = tracker2.energy()

    assert e_after_accept == e_fresh
