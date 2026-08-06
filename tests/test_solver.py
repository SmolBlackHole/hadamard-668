"""Tests for KFlip solver."""

from __future__ import annotations

import numpy as np

from builder import Builder
from solver import search as ils_search
from tracker import Tracker


def test_solver_finds_solution_small_n() -> None:
    b = Builder(kind="gs4", n=5)
    seqs = np.random.default_rng(42).choice((-1, 1), size=(b.k, 5)).astype(np.int8)
    tracker = Tracker()
    tracker.build(seqs)
    best_seq, best_e, _iters, _stats = ils_search(
        seqs, tracker, np.random.default_rng(42), steps=5000
    )

    assert best_e == 0
    matrix = b.build(best_seq)
    G = matrix.astype(np.int64) @ matrix.astype(np.int64).T
    assert np.array_equal(G, 20 * np.eye(20, dtype=np.int64))


def test_solver_budget_not_exceeded() -> None:
    b = Builder(kind="gs4", n=3)
    seqs = np.ones((b.k, 3), dtype=np.int8)
    tracker = Tracker()
    tracker.build(seqs)

    budget = 1000
    _best_seq, _best_e, consumed, _stats = ils_search(
        seqs, tracker, np.random.default_rng(99), steps=budget
    )
    assert consumed <= budget
    assert consumed >= 1


def test_solver_best_e_never_increases() -> None:
    from solver import _update_best

    best_seq = np.ones((4, 3), dtype=np.int8)
    cur_seq = best_seq.copy()

    _, e1 = _update_best(cur_seq, 10, best_seq, 20)
    assert e1 == 10

    _, e2 = _update_best(cur_seq, 30, best_seq, 10)
    assert e2 == 10

    _, e3 = _update_best(cur_seq, 5, best_seq, 5)
    assert e3 == 5
