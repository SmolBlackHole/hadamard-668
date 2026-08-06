"""Tests for KFlip solver."""

from __future__ import annotations

import numpy as np

from builder import Builder
from solver import search as ils_search
from tracker import GramTracker


def test_solver_finds_solution_small_n() -> None:
    b = Builder(kind="gs4", n=5)
    seqs = np.random.default_rng(42).choice((-1, 1), size=(b.k, 5)).astype(np.int8)
    tracker = GramTracker(b.build)
    best_seq, best_e, _iters = ils_search(seqs, tracker, np.random.default_rng(42), steps=5000)

    assert best_e == 0
    matrix = b.build(best_seq)
    G = matrix.astype(np.int64) @ matrix.astype(np.int64).T
    assert np.array_equal(G, 20 * np.eye(20, dtype=np.int64))


def test_solver_golay_2n_kick_recovery() -> None:
    b = Builder(kind="golay_2n", n=6)
    seqs = np.random.default_rng(7).choice((-1, 1), size=(b.k, 6)).astype(np.int8)
    tracker = GramTracker(b.build)
    best_seq, best_e, _iters = ils_search(seqs, tracker, np.random.default_rng(7), steps=1000)

    assert best_e == 0
    matrix = b.build(best_seq)
    G = matrix.astype(np.int64) @ matrix.astype(np.int64).T
    assert np.array_equal(G, 12 * np.eye(12, dtype=np.int64))


def test_solver_budget_not_exceeded() -> None:
    """The solver must not consume more flip evaluations than given steps."""
    b = Builder(kind="gs4", n=3)
    seqs = np.ones((b.k, 3), dtype=np.int8)
    tracker = GramTracker(b.build)

    budget = 1000
    _best_seq, _best_e, consumed = ils_search(
        seqs, tracker, np.random.default_rng(99), steps=budget
    )
    # consumed = total_budget - remaining, from solver's return value
    # build costs 1 step, the rest are flip evaluations
    assert consumed <= budget
    assert consumed >= 1  # at least one build


def test_solver_best_e_never_increases() -> None:
    """best_e from _update_best must be monotonically non-increasing."""
    from solver import _update_best

    best_seq = np.ones((4, 3), dtype=np.int8)
    cur_seq = best_seq.copy()

    # energy going down → new best
    _, e1 = _update_best(cur_seq, 10, best_seq, 20)
    assert e1 == 10

    # energy going up → old best kept
    _, e2 = _update_best(cur_seq, 30, best_seq, 10)
    assert e2 == 10  # best unchanged

    # same energy → old best kept
    _, e3 = _update_best(cur_seq, 5, best_seq, 5)
    assert e3 == 5


def test_all_kinds_validate() -> None:
    """Every builder kind reaches energy 0 with enough steps (real search)."""
    configs = [("gs4", 5, 5000), ("golay_2n", 6, 1000), ("gs4_group", 6, 3000)]
    for kind, n_val, budget in configs:
        b = Builder(kind=kind, n=n_val)
        seqs = np.random.default_rng(42).choice((-1, 1), size=(b.k, n_val)).astype(np.int8)
        tracker = GramTracker(b.build)
        best_seq, best_e, _iters = ils_search(
            seqs, tracker, np.random.default_rng(42), steps=budget
        )
        assert best_e == 0, f"kind={kind} best_e={best_e}"
        matrix = b.build(best_seq)
        G = matrix.astype(np.int64) @ matrix.astype(np.int64).T
        assert np.array_equal(G, b.order * np.eye(b.order, dtype=np.int64)), (
            f"kind={kind} matrix not orthogonal"
        )
