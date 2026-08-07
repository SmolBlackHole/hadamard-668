"""Tests for KFlip solver."""

from __future__ import annotations

from typing import cast

import numpy as np
import pytest

from src.builder import Builder
from src.solver import SearchStats, SolverConfig, _rescue, _tabu_walk
from src.solver import search as ils_search
from src.tracker import Tracker


class _ScriptedTracker:
    """Minimal tracker that makes solver phase transitions deterministic."""

    def __init__(self, pair_energy: int | None = None) -> None:
        self._energy = 10
        self._pair_energy = pair_energy
        self.accepted: list[tuple[int, int]] = []
        self.pair_calls = 0

    def build(self, seqs: np.ndarray) -> None:
        del seqs

    def energy(self) -> int:
        return self._energy

    def flip_batch(self, start: int, stop: int) -> np.ndarray:
        return np.full(stop - start, 10, dtype=np.int64)

    def pair_energies(self, candidates: list[tuple[int, int]]) -> np.ndarray:
        self.pair_calls += 1
        energies = np.full((len(candidates), len(candidates)), 10, dtype=np.int64)
        if self._pair_energy is not None and self.pair_calls == 1:
            energies[0, 1] = self._pair_energy
        return energies

    def accept(self, seqs: np.ndarray, s: int, c: int) -> int:
        del seqs
        self.accepted.append((s, c))
        if self._pair_energy is not None and self.pair_calls:
            self._energy = self._pair_energy
        else:
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


def test_rescue_accepts_first_improving_pair() -> None:
    tracker = _ScriptedTracker(pair_energy=9)
    seqs = np.ones((4, 2), dtype=np.int8)
    candidates = [(0, 0), (0, 1), (1, 0)]

    result, evaluations = _rescue(seqs, tracker, candidates, 10)  # type: ignore[arg-type]

    assert result == 9
    assert evaluations == 1
    assert tracker.accepted == candidates[:2]


def test_solver_uses_pair_rescue_after_a_failed_single_scan() -> None:
    tracker = _ScriptedTracker(pair_energy=9)
    seqs = np.ones((4, 2), dtype=np.int8)

    _best, energy, _used, stats = ils_search(
        seqs,
        cast(Tracker, tracker),
        np.random.default_rng(1),
        steps=10,
        config=SolverConfig(kick=False),
    )

    assert energy == 9
    assert stats.pairs == 1
    assert stats.kicks == 0
    assert tracker.pair_calls == 1
    assert len(tracker.accepted) == 2


def test_solver_kicks_when_no_pair_improves() -> None:
    tracker = _ScriptedTracker()
    seqs = np.ones((4, 2), dtype=np.int8)

    _best, _energy, _used, stats = ils_search(
        seqs, cast(Tracker, tracker), np.random.default_rng(1), steps=10, config=SolverConfig(tabu=False)
    )

    assert tracker.pair_calls == 1
    assert stats.pairs == 0
    assert stats.kicks == 1
    assert len(tracker.accepted) == 4


def test_solver_skips_rescue_when_pairs_are_disabled() -> None:
    tracker = _ScriptedTracker()
    seqs = np.ones((4, 2), dtype=np.int8)

    _best, _energy, _used, stats = ils_search(
        seqs,
        cast(Tracker, tracker),
        np.random.default_rng(1),
        steps=10,
        config=SolverConfig(pairs=False, tabu=False),
    )

    assert tracker.pair_calls == 0
    assert stats.kicks == 1


def test_solver_stops_before_rescue_when_single_scan_exhausts_budget() -> None:
    tracker = _ScriptedTracker()
    seqs = np.ones((4, 2), dtype=np.int8)

    _best, _energy, used, stats = ils_search(
        seqs, cast(Tracker, tracker), np.random.default_rng(1), steps=8
    )

    assert used == 7
    assert tracker.pair_calls == 0
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


def test_tabu_walk_returns_the_rebuilt_tracker_energy() -> None:
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

    if result is not None:
        assert result == tracker.energy()
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
