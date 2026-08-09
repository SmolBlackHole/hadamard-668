"""Tests for KFlip solver."""

from __future__ import annotations

from itertools import product
from typing import cast

import numpy as np
import pytest

from src.builder import Builder
from src.solver import (
    SearchStats,
    SolverConfig,
    _legacy_escape_candidates,
    _support_lag625_candidates,
    _tabu_walk,
)
from src.solver import search as ils_search
from src.tracker import Tracker


class _ScriptedTracker:
    """Minimal tracker that makes solver phase transitions deterministic."""

    def __init__(self, initial_energy: int = 10) -> None:
        self._energy = initial_energy
        self.accepted: list[tuple[int, int]] = []

    def build(self, seqs: np.ndarray) -> None:
        del seqs

    def energy(self) -> int:
        return self._energy

    def flip_batch(self, start: int, stop: int) -> np.ndarray:
        return np.full(stop - start, self._energy, dtype=np.int64)

    def accept(self, seqs: np.ndarray, s: int, c: int) -> int:
        del seqs
        self.accepted.append((s, c))
        self._energy += 1
        return self._energy


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
    from src.solver import _update_best

    best_seq = np.ones((4, 3), dtype=np.int8)
    cur_seq = best_seq.copy()

    _, e1 = _update_best(cur_seq, 10, best_seq, 20)
    assert e1 == 10

    _, e2 = _update_best(cur_seq, 30, best_seq, 10)
    assert e2 == 10

    _, e3 = _update_best(cur_seq, 5, best_seq, 5)
    assert e3 == 5


def test_solver_kicks_after_failed_single_scan() -> None:
    tracker = _ScriptedTracker()
    seqs = np.ones((4, 2), dtype=np.int8)

    _best, _energy, _used, stats = ils_search(
        seqs,
        cast(Tracker, tracker),
        np.random.default_rng(1),
        steps=10,
        config=SolverConfig(tabu=False),
    )

    assert stats.kicks == 1
    assert len(tracker.accepted) == 2


def test_legacy_escape_candidates_use_dominant_lag_and_first_five_cartesian_order() -> None:
    n = 52
    u = np.zeros(25, dtype=np.int32)
    u[7] = -9
    delta = np.zeros((4 * n, 25), dtype=np.int8)
    columns = (
        [1, 4, 8, 12, 15, 20],
        [2, 5, 9, 13, 16, 21],
        [3, 6, 10, 14, 17, 22],
        [7, 11, 18, 24, 31, 40],
    )
    for s, values in enumerate(columns):
        delta[s * n + np.asarray(values), 7] = 9

    k, candidates = _legacy_escape_candidates(delta, u, n)

    assert k == 7
    assert len(candidates) == 625
    assert candidates[:3] == [(1, 2, 3, 7), (1, 2, 3, 11), (1, 2, 3, 18)]
    assert candidates[-1] == (15, 16, 17, 31)


def _support_lag_fixture() -> tuple[np.ndarray, np.ndarray]:
    n = 24
    u = np.zeros(11, dtype=np.int32)
    u[[2, 7]] = [1, -1]
    delta = np.zeros((4 * n, 11), dtype=np.int8)
    for s in range(4):
        delta[s * n + np.arange(5), 2] = -1
        delta[s * n + np.arange(10, 15), 7] = 1
    return delta, u


def test_support_lag625_is_deterministic_bounded_and_covers_support_lags() -> None:
    delta, u = _support_lag_fixture()

    candidates = _support_lag625_candidates(delta, u, 24)

    assert candidates == _support_lag625_candidates(delta, u, 24)
    assert len(candidates) == 625
    assert len(candidates) == len(set(candidates))
    first_lag = set(product(range(5), repeat=4))
    second_lag = set(product(range(10, 15), repeat=4))
    assert any(candidate in first_lag for candidate in candidates)
    assert any(candidate in second_lag for candidate in candidates)
    assert (0, 0, 0, 0) in candidates
    assert (4, 4, 4, 4) in candidates
    assert (10, 10, 10, 10) in candidates
    assert (14, 14, 14, 14) in candidates


def test_support_lag625_deduplicates_and_backfills() -> None:
    n = 8
    u = np.zeros(3, dtype=np.int32)
    u[[0, 1]] = 1
    delta = np.zeros((4 * n, 3), dtype=np.int8)
    for s in range(4):
        rows = s * n + np.arange(5)
        delta[rows, 0] = -1
        delta[rows, 1] = -1

    candidates = _support_lag625_candidates(delta, u, n)

    assert len(candidates) == 625
    assert len(candidates) == len(set(candidates))


def test_default_targeted_escape_uses_legacy_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    from src import solver

    seqs = np.ones((4, 3), dtype=np.int8)
    tracker = Tracker()
    tracker.build(seqs)

    def legacy_candidates(*_args: object) -> tuple[int, list[tuple[int, int, int, int]]]:
        return 0, [(0, 0, 0, 0)]

    monkeypatch.setattr(solver, "_legacy_escape_candidates", legacy_candidates)

    def fake_search(
        cand: np.ndarray, empty_tracker: Tracker, *_args: object, **_kwargs: object
    ) -> tuple[np.ndarray, int, int, SearchStats]:
        assert empty_tracker._u is None
        return cand, 1, 1, SearchStats()

    monkeypatch.setattr(solver, "search", fake_search)
    result = solver._targeted_kick(
        seqs,
        tracker,
        np.random.default_rng(1),
        1,
        100,
        seqs.copy(),
        tracker.energy(),
        SolverConfig(),
        SearchStats(),
    )

    assert result[0] is False


@pytest.mark.parametrize(
    ("config", "outer_budget", "expected_steps"),
    [
        (SolverConfig(), 100_000, 10_000),
        (SolverConfig(escape_quench_steps=17), 100_000, 17),
        (SolverConfig(escape_quench_steps=17), 12, 3),
        (SolverConfig(escape_quench_steps=17), 3, 1),
    ],
)
def test_targeted_escape_passes_limited_quench_budget(
    monkeypatch: pytest.MonkeyPatch,
    config: SolverConfig,
    outer_budget: int,
    expected_steps: int,
) -> None:
    from src import solver

    seqs = np.ones((4, 3), dtype=np.int8)
    tracker = Tracker()
    tracker.build(seqs)
    seen_steps: list[int] = []

    def legacy_candidates(*_args: object) -> tuple[int, list[tuple[int, int, int, int]]]:
        return 0, [(0, 0, 0, 0)]

    monkeypatch.setattr(solver, "_legacy_escape_candidates", legacy_candidates)

    def fake_search(
        cand: np.ndarray, _tracker: Tracker, *_args: object, **kwargs: object
    ) -> tuple[np.ndarray, int, int, SearchStats]:
        seen_steps.append(cast(int, kwargs["steps"]))
        return cand, 1, 1, SearchStats()

    monkeypatch.setattr(solver, "search", fake_search)
    solver._targeted_kick(
        seqs,
        tracker,
        np.random.default_rng(1),
        1,
        outer_budget,
        seqs.copy(),
        tracker.energy(),
        config,
        SearchStats(),
    )

    assert seen_steps == [expected_steps]


def test_targeted_escape_rejects_non_positive_quench_budget() -> None:
    from src import solver

    seqs = np.ones((4, 3), dtype=np.int8)
    tracker = Tracker()
    tracker.build(seqs)

    with pytest.raises(ValueError, match="escape_quench_steps must be positive"):
        solver._targeted_kick(
            seqs,
            tracker,
            np.random.default_rng(1),
            1,
            100,
            seqs.copy(),
            tracker.energy(),
            SolverConfig(escape_quench_steps=0),
            SearchStats(),
        )


def test_solver_stops_when_single_scan_exhausts_budget() -> None:
    tracker = _ScriptedTracker()
    seqs = np.ones((4, 2), dtype=np.int8)

    _best, _energy, used, stats = ils_search(
        seqs, cast(Tracker, tracker), np.random.default_rng(1), steps=8
    )

    assert used == 7
    assert stats.kicks == 0
    assert tracker.accepted == []


def test_search_stats_display_flushes_streak() -> None:
    stats = SearchStats()
    stats._hit_single()
    stats._hit_single()
    stats.energy_saved_singles = 2_000

    assert "S=2(2k)" in stats.display()
    assert "strk=2.0/2" in stats.display()


def test_tabu_walk_replays_its_best_state() -> None:
    seqs = np.random.default_rng(0).choice((-1, 1), size=(4, 4)).astype(np.int8)
    tracker = Tracker()
    tracker.build(seqs)
    before = tracker.energy()
    expected_index = int(np.argmin(tracker.flip_energies()))
    expected_energy = int(tracker.flip_energies()[expected_index])
    expected = seqs.copy()
    expected[expected_index // 4, expected_index % 4] *= -1

    result, evaluations = _tabu_walk(
        seqs,
        tracker,
        before,
        np.random.default_rng(1),
        SolverConfig(tabu=True, tabu_steps=1, tabu_noise=0.0),
    )

    assert evaluations == 1
    assert result == expected_energy
    assert np.array_equal(seqs, expected)
    assert tracker.energy() == expected_energy


def test_tabu_walk_keeps_main_state_when_it_finds_no_improvement() -> None:
    seqs = np.random.default_rng(0).choice((-1, 1), size=(4, 3)).astype(np.int8)
    tracker = Tracker()
    tracker.build(seqs)
    index = int(np.argmin(tracker.flip_energies()))
    seqs[index // 3, index % 3] *= -1
    tracker.build(seqs)
    assert tracker.energy() == 0


def test_tabu_walk_adopts_an_exact_tracker_snapshot() -> None:
    seqs = np.random.default_rng(8).choice((-1, 1), size=(4, 5)).astype(np.int8)
    tracker = Tracker()
    tracker.build(seqs)

    result, _ = _tabu_walk(
        seqs,
        tracker,
        tracker.energy(),
        np.random.default_rng(2),
        SolverConfig(tabu=True, tabu_steps=10),
    )

    assert result is not None
    assert result == tracker.energy()
    rebuilt = Tracker()
    rebuilt.build(seqs)
    assert (
        tracker._u is not None
        and rebuilt._u is not None
        and tracker._delta is not None
        and rebuilt._delta is not None
        and tracker._norm2 is not None
        and rebuilt._norm2 is not None
    )
    assert np.array_equal(tracker._u, rebuilt._u)
    assert np.array_equal(tracker._delta, rebuilt._delta)
    assert np.array_equal(tracker._norm2, rebuilt._norm2)
    assert np.array_equal(tracker.flip_qs(), rebuilt.flip_qs())
    original = seqs.copy()

    result, evaluations = _tabu_walk(
        seqs,
        tracker,
        0,
        np.random.default_rng(1),
        SolverConfig(tabu=True, tabu_steps=3, tabu_noise=0.0),
    )

    assert result is None
    assert evaluations == 3
    assert np.array_equal(seqs, original)
    assert tracker.energy() == 0


def test_tabu_walk_does_not_rebuild_a_temporary_tracker(monkeypatch: pytest.MonkeyPatch) -> None:
    seqs = np.random.default_rng(8).choice((-1, 1), size=(4, 5)).astype(np.int8)
    tracker = Tracker()
    tracker.build(seqs)

    def unexpected_build(*_args: object) -> None:
        raise AssertionError("Tabu walk must reuse the main tracker cache")

    monkeypatch.setattr(Tracker, "build", unexpected_build)
    result, evaluations = _tabu_walk(
        seqs,
        tracker,
        tracker.energy(),
        np.random.default_rng(2),
        SolverConfig(tabu=True, tabu_steps=10),
    )

    assert result == tracker.energy()
    assert evaluations == 1


def test_solver_kicks_after_an_unsuccessful_tabu_walk(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    def failed_tabu(*_args: object) -> tuple[None, int]:
        calls.append(1)
        return None, 3

    monkeypatch.setattr("src.solver._tabu_walk", failed_tabu)
    tracker = _ScriptedTracker()
    seqs = np.ones((4, 2), dtype=np.int8)

    _best, _energy, _used, stats = ils_search(
        seqs,
        cast(Tracker, tracker),
        np.random.default_rng(1),
        steps=10,
        config=SolverConfig(tabu=True),
    )

    assert calls == [1]
    assert stats.tabu_walks == 1
    assert stats.tabu_hits == 0
    assert stats.tabu_evals == 3
    assert stats.kicks == 1


def test_solver_counts_tabu_outcomes_by_start_q(monkeypatch: pytest.MonkeyPatch) -> None:
    def successful_tabu(*_args: object) -> tuple[int, int]:
        return 0, 3

    monkeypatch.setattr("src.solver._tabu_walk", successful_tabu)
    tracker = _ScriptedTracker(initial_energy=256)
    seqs = np.ones((4, 2), dtype=np.int8)

    _best, energy, _used, stats = ils_search(
        seqs,
        cast(Tracker, tracker),
        np.random.default_rng(1),
        steps=10,
        config=SolverConfig(kick=False, tabu=True),
    )

    assert energy == 0
    assert stats.tabu_walks_q2 == 1
    assert stats.tabu_hits_q2 == 1
    assert stats.tabu_walks_q1 == 0
    assert stats.tabu_hits_q1 == 0
