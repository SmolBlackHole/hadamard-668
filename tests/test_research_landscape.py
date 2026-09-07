"""Independent small-state checks for the landscape diagnostic and scan prototype."""

from itertools import combinations

import numpy as np

from lab.recover import descend, geometry
from src.constructions import paley_ng_sequences
from src.models import CandidateBudget, SearchStats
from src.solver import SolverConfig
from src.solver.greedy import greedy_descent
from src.tracker import Tracker
from src.verify import verify_candidate


def test_pair_geometry_matches_independent_rebuilds() -> None:
    state = paley_ng_sequences(6)
    state[0, 0] *= -1
    actual = geometry(state)
    pair_values: list[int] = []
    for first, second in combinations(range(state.size), 2):
        candidate = state.copy()
        candidate.flat[first] *= -1
        candidate.flat[second] *= -1
        tracker = Tracker()
        tracker.build(candidate)
        pair_values.append(tracker.energy() // (64 * 6))
    assert actual["best_pair"] == min(pair_values)
    assert actual["best_single"] == 0


def test_full_scan_takes_direct_solution_and_charges_every_score() -> None:
    state = paley_ng_sequences(52)
    state[-1, -1] *= -1
    tracker = Tracker()
    tracker.build(state)
    budget = CandidateBudget(208)
    stats = SearchStats()
    positions = tuple((s, c) for s in range(4) for c in range(52))
    improved, energy, used = greedy_descent(
        state, tracker, positions, tracker.energy(), budget, 64 * 52, SolverConfig(), stats
    )
    assert improved and energy == 0
    assert used == budget.used == stats.greedy_candidate_evals == 208
    verify_candidate(state)


def test_descent_never_increases_q_or_mutates_input() -> None:
    state = np.random.default_rng(4).choice(np.array([-1, 1], dtype=np.int8), size=(4, 12))
    before = state.copy()
    tracker = Tracker()
    tracker.build(state)
    initial = tracker.energy()
    result = descend(state)
    tracker.build(result)
    assert tracker.energy() <= initial
    assert np.array_equal(state, before)


def test_full_scan_respects_partial_budget() -> None:
    state = paley_ng_sequences(6)
    state[0, 0] *= -1
    tracker = Tracker()
    tracker.build(state)
    budget = CandidateBudget(1)
    positions = tuple((s, c) for s in range(4) for c in range(6))
    _, _, used = greedy_descent(
        state, tracker, positions, tracker.energy(), budget, 64 * 6, SolverConfig(), SearchStats()
    )
    assert used == budget.used == 1
