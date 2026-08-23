"""Tests for KFlip solver."""

from __future__ import annotations

from itertools import product
from typing import cast

import numpy as np
import pytest

from src.builder import build_gs4
from src.models import CandidateBudget, SearchPhase, SearchStats, SolverResult
from src.solver import (
    SolverConfig,
    TabuWalkResult,
    _tabu_walk,
    _targeted_candidates,
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
    seqs = np.random.default_rng(42).choice((-1, 1), size=(4, 5)).astype(np.int8)
    tracker = Tracker()
    tracker.build(seqs)
    result = ils_search(seqs, tracker, np.random.default_rng(42), candidate_budget=500_000)

    assert result.solved
    matrix = build_gs4(result.sequences)
    G = matrix.astype(np.int64) @ matrix.astype(np.int64).T
    assert np.array_equal(G, 20 * np.eye(20, dtype=np.int64))


def test_solver_budget_not_exceeded() -> None:
    seqs = np.ones((4, 3), dtype=np.int8)
    tracker = Tracker()
    tracker.build(seqs)

    budget = 1000
    result = ils_search(seqs, tracker, np.random.default_rng(99), candidate_budget=budget)
    assert result.candidate_evals <= budget
    assert result.candidate_evals == result.stats.total_candidate_evals


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

    result = ils_search(
        seqs,
        cast(Tracker, tracker),
        np.random.default_rng(1),
        candidate_budget=10,
        config=SolverConfig(tabu=False),
    )

    assert result.stats.random_kicks == 1
    assert len(tracker.accepted) == 2


def _targeted_fixture() -> tuple[np.ndarray, np.ndarray]:
    n = 24
    u = np.zeros(11, dtype=np.int32)
    u[[2, 7]] = [1, -1]
    delta = np.zeros((4 * n, 11), dtype=np.int8)
    for s in range(4):
        delta[s * n + np.arange(5), 2] = -1
        delta[s * n + np.arange(10, 15), 7] = 1
    return delta, u


def test_targeted_candidates_are_deterministic_bounded_and_cover_support_lags() -> None:
    delta, u = _targeted_fixture()

    candidates = _targeted_candidates(delta, u, 24)

    assert candidates == _targeted_candidates(delta, u, 24)
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


def test_targeted_candidates_deduplicate_and_backfill() -> None:
    n = 8
    u = np.zeros(3, dtype=np.int32)
    u[[0, 1]] = 1
    delta = np.zeros((4 * n, 3), dtype=np.int8)
    for s in range(4):
        rows = s * n + np.arange(5)
        delta[rows, 0] = -1
        delta[rows, 1] = -1

    candidates = _targeted_candidates(delta, u, n)

    assert len(candidates) == 625
    assert len(candidates) == len(set(candidates))


def test_targeted_escape_uses_support_lags(monkeypatch: pytest.MonkeyPatch) -> None:
    from src import solver

    seqs = np.ones((4, 3), dtype=np.int8)
    tracker = Tracker()
    tracker.build(seqs)

    def candidates(*_args: object) -> list[tuple[int, int, int, int]]:
        return [(0, 0, 0, 0)]

    monkeypatch.setattr(solver, "_targeted_candidates", candidates)

    def fake_search(
        cand: np.ndarray, empty_tracker: Tracker, *_args: object, **_kwargs: object
    ) -> SolverResult:
        assert empty_tracker._u is None
        return SolverResult(cand, 1, 1, SearchStats())

    monkeypatch.setattr(solver, "_search", fake_search)
    budget = CandidateBudget(100)
    result = solver._targeted_kick(
        seqs,
        tracker,
        np.random.default_rng(1),
        budget,
        seqs.copy(),
        tracker.energy(),
        SolverConfig(),
        SearchStats(),
    )

    assert result.solved is False


@pytest.mark.parametrize(
    ("config", "outer_budget", "expected_budget"),
    [
        (SolverConfig(), 100_000_000, 10_000_000),
        (SolverConfig(escape_quench_budget=17), 100_000, 17),
        (SolverConfig(escape_quench_budget=17), 12, 11),
        (SolverConfig(escape_quench_budget=17), 3, 2),
    ],
)
def test_targeted_escape_passes_limited_quench_budget(
    monkeypatch: pytest.MonkeyPatch,
    config: SolverConfig,
    outer_budget: int,
    expected_budget: int,
) -> None:
    from src import solver

    seqs = np.ones((4, 3), dtype=np.int8)
    tracker = Tracker()
    tracker.build(seqs)
    seen_budgets: list[int] = []

    def candidates(*_args: object) -> list[tuple[int, int, int, int]]:
        return [(0, 0, 0, 0)]

    monkeypatch.setattr(solver, "_targeted_candidates", candidates)

    def fake_search(
        cand: np.ndarray, _tracker: Tracker, *_args: object, **kwargs: object
    ) -> SolverResult:
        seen_budgets.append(cast(CandidateBudget, kwargs["budget"]).limit)
        return SolverResult(cand, 1, 1, SearchStats())

    monkeypatch.setattr(solver, "_search", fake_search)
    solver._targeted_kick(
        seqs,
        tracker,
        np.random.default_rng(1),
        CandidateBudget(outer_budget),
        seqs.copy(),
        tracker.energy(),
        config,
        SearchStats(),
    )

    assert seen_budgets == [expected_budget]


def test_targeted_escape_records_nested_solve_and_candidate_work(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src import solver

    seqs = np.ones((4, 3), dtype=np.int8)
    tracker = Tracker()
    tracker.build(seqs)

    def candidates(*_args: object) -> list[tuple[int, int, int, int]]:
        return [(0, 0, 0, 0)]

    monkeypatch.setattr(solver, "_targeted_candidates", candidates)

    def fake_search(
        cand: np.ndarray, _tracker: Tracker, *_args: object, **_kwargs: object
    ) -> SolverResult:
        nested = SearchStats(
            greedy_moves=2,
            tabu_moves=3,
            random_kicks=1,
            greedy_candidate_evals=7,
        )
        nested.record_solve("greedy", 1)
        return SolverResult(cand, 0, 7, nested)

    monkeypatch.setattr(solver, "_search", fake_search)
    stats = SearchStats()
    budget = CandidateBudget(100)
    result = solver._targeted_kick(
        seqs,
        tracker,
        np.random.default_rng(1),
        budget,
        seqs.copy(),
        tracker.energy(),
        SolverConfig(),
        stats,
    )

    assert result.solved
    assert budget.used == 8
    assert stats.targeted_quenches == 1
    assert stats.random_kicks == 0
    assert stats.quench_moves == 6
    assert stats.total_accepted_moves == 7
    assert stats.quench_candidate_evals == 7
    assert stats.total_candidate_evals == 8
    assert stats.solve_phase == "targeted:greedy"
    assert stats.solve_q_before == 1


def test_targeted_escape_rejects_non_positive_quench_budget() -> None:
    with pytest.raises(ValueError, match="escape_quench_budget must be positive"):
        SolverConfig(escape_quench_budget=0)


def test_solver_stops_when_single_scan_exhausts_budget() -> None:
    tracker = _ScriptedTracker()
    seqs = np.ones((4, 2), dtype=np.int8)

    result = ils_search(seqs, cast(Tracker, tracker), np.random.default_rng(1), candidate_budget=8)

    assert result.candidate_evals == 8
    assert result.stats.random_kicks == 0
    assert tracker.accepted == []


def test_search_stats_display_flushes_streak() -> None:
    stats = SearchStats()
    stats.hit_greedy()
    stats.hit_greedy()
    stats.record_solve("greedy", 2)

    assert "moves=2(2/0/0/0/0)" in stats.display()
    assert "solve=greedy@Q2" in stats.display()
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

    result = _tabu_walk(
        seqs,
        tracker,
        before,
        np.random.default_rng(1),
        SolverConfig(tabu=True, tabu_steps=1, tabu_noise=0.0),
    )

    assert result.steps == 1
    assert result.energy == expected_energy
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

    result = _tabu_walk(
        seqs,
        tracker,
        tracker.energy(),
        np.random.default_rng(2),
        SolverConfig(tabu=True, tabu_steps=10),
    )

    assert result.energy is not None
    assert result.energy == tracker.energy()
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

    result = _tabu_walk(
        seqs,
        tracker,
        0,
        np.random.default_rng(1),
        SolverConfig(tabu=True, tabu_steps=3, tabu_noise=0.0),
    )

    assert result.energy is None
    assert result.steps == 3
    assert np.array_equal(seqs, original)
    assert tracker.energy() == 0


def test_tabu_walk_does_not_rebuild_a_temporary_tracker(monkeypatch: pytest.MonkeyPatch) -> None:
    seqs = np.random.default_rng(8).choice((-1, 1), size=(4, 5)).astype(np.int8)
    tracker = Tracker()
    tracker.build(seqs)

    def unexpected_build(*_args: object) -> None:
        raise AssertionError("Tabu walk must reuse the main tracker cache")

    monkeypatch.setattr(Tracker, "build", unexpected_build)
    result = _tabu_walk(
        seqs,
        tracker,
        tracker.energy(),
        np.random.default_rng(2),
        SolverConfig(tabu=True, tabu_steps=10),
    )

    assert result.energy == tracker.energy()
    assert result.steps == 1


def test_solver_kicks_after_an_unsuccessful_tabu_walk(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[int] = []

    def failed_tabu(*_args: object, **_kwargs: object) -> TabuWalkResult:
        calls.append(1)
        return TabuWalkResult(None, 3)

    monkeypatch.setattr("src.solver._tabu_walk", failed_tabu)
    tracker = _ScriptedTracker()
    seqs = np.ones((4, 2), dtype=np.int8)

    result = ils_search(
        seqs,
        cast(Tracker, tracker),
        np.random.default_rng(1),
        candidate_budget=33,
        config=SolverConfig(tabu=True),
    )

    assert calls == [1]
    assert result.stats.tabu_walks == 1
    assert result.stats.tabu_improvements == 0
    assert result.stats.tabu_moves == 3
    assert result.stats.random_kicks == 1


def test_solver_counts_tabu_outcomes_by_start_q(monkeypatch: pytest.MonkeyPatch) -> None:
    def successful_tabu(*_args: object, **_kwargs: object) -> TabuWalkResult:
        return TabuWalkResult(0, 3, solve_q_before=4)

    monkeypatch.setattr("src.solver._tabu_walk", successful_tabu)
    tracker = _ScriptedTracker(initial_energy=256)
    seqs = np.ones((4, 2), dtype=np.int8)

    result = ils_search(
        seqs,
        cast(Tracker, tracker),
        np.random.default_rng(1),
        candidate_budget=32,
        config=SolverConfig(kick=False, tabu=True),
    )

    stats = result.stats
    assert result.solved
    assert stats.tabu_walks_q2 == 1
    assert stats.tabu_improvements_q2 == 1
    assert stats.tabu_walks_q1 == 0
    assert stats.tabu_improvements_q1 == 0
    assert stats.tabu_solves == 1
    assert stats.tabu_solves_q2 == 1
    assert stats.solve_phase == "tabu"
    assert stats.solve_q_before == 4
    assert stats.tabu_candidate_evals == 24


def test_solver_config_validates_search_parameters() -> None:
    with pytest.raises(ValueError, match="tabu_steps cannot be negative"):
        SolverConfig(tabu_steps=-1)


def test_phase_trace_is_cost_complete() -> None:
    sequences = np.random.default_rng(4).choice((-1, 1), size=(4, 9)).astype(np.int8)
    result = ils_search(
        sequences,
        Tracker(),
        np.random.default_rng(5),
        candidate_budget=20_000,
        config=SolverConfig(targeted_escape=False, trace_phases=True),
    )

    events = result.stats.phase_events
    assert events[0].phase == SearchPhase.INITIALIZE
    assert sum(event.candidate_evals for event in events) == result.candidate_evals
    assert sum(event.accepted_moves for event in events) == result.stats.total_accepted_moves
    assert all(event.state_hash_after and event.orbit_hash_after for event in events)
    for event in events:
        if event.phase == SearchPhase.TABU:
            assert (
                event.downhill_moves + event.lateral_moves + event.uphill_moves
                == event.accepted_moves
            )
