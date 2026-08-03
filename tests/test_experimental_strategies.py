"""Contracts for the experimental search strategies."""
from __future__ import annotations

import numpy as np
import pytest

from constructions import (
    apply_sequence_flip,
    apply_symmetric_flip,
    autocorrelation_state,
    correlation_energy,
    periodic_autocorrelation_energy,
    symmetric_circulant,
)
from gpu import check_orthogonality, to_numpy, xp
from strategies.baumert import BaumertHallSearch
from strategies.ca import CASearch
from strategies.diffset import DiffsetSearch
from strategies.genetic import GeneticSearch
from strategies.gold import GoldSearch
from strategies.ising import IsingSearch
from strategies.montecarlo import MonteCarloSearch
from strategies.sat import SatSearch
from strategies.walsh import WalshSearch
from strategies.base import Pipeline, SearchStrategy
from verifier.known import sylvester


def reference_periodic_energy(sequences: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> int:
    total = 0
    for displacement in range(1, len(sequences[0])):
        correlation = sum(
            int(np.dot(sequence, np.roll(sequence, -displacement)))
            for sequence in sequences)
        total += correlation * correlation
    return total


def test_periodic_autocorrelation_matches_reference() -> None:
    sequences = tuple(np.array([1, -1, 1], dtype=np.int8) for _ in range(4))
    assert periodic_autocorrelation_energy(
        sequences) == reference_periodic_energy(sequences)


def test_walsh_builds_sylvester_order_four() -> None:
    _, metrics, _ = WalshSearch(4).search(steps=1, seed=0)
    assert metrics["energy"] == 0


def test_gold_search_is_deterministic() -> None:
    first, first_metrics, _ = GoldSearch(4).search(steps=1, seed=1)
    second, second_metrics, _ = GoldSearch(4).search(steps=1, seed=2)
    assert np.array_equal(first, second)
    assert first_metrics == second_metrics
    assert np.all(np.isin(first, (-1, 1)))


def test_sat_search_solves_order_four() -> None:
    _, metrics, _ = SatSearch(4).search(steps=16, seed=0)
    assert metrics["energy"] == 0


def test_ising_search_is_seeded() -> None:
    first, _, _ = IsingSearch(4).search(steps=3, seed=4)
    second, _, _ = IsingSearch(4).search(steps=3, seed=4)
    assert np.array_equal(first, second)


@pytest.mark.parametrize("strategy", [DiffsetSearch(4), GeneticSearch(4, population_size=4)])
def test_cyclic_strategies_solve_order_four(strategy) -> None:
    _, metrics, _ = strategy.search(steps=2, seed=0)
    assert metrics["energy"] == 0


@pytest.mark.skipif(xp.__name__ != "cupy", reason="requires active CuPy backend")
def test_montecarlo_search_uses_gpu() -> None:
    _, metrics, _ = MonteCarloSearch(4, batch_size=4).search(steps=2, seed=0)
    assert metrics["energy"] == 0


def test_ca_identity_rule_preserves_hadamard() -> None:
    matrix = sylvester(4)
    rule = np.zeros((3, 3))
    rule[1, 1] = 1
    assert np.array_equal(CASearch._apply_ca(matrix, rule), matrix)


def test_ca_rule_mutation_changes_one_weight() -> None:
    rule = np.zeros((3, 3))
    mutated = CASearch._mutate_rule(rule, np.random.default_rng(0))
    assert np.count_nonzero(mutated != rule) == 1
    assert np.all(np.isin(mutated, (-1.0, 0.0, 1.0)))


def test_ca_refinement_preserves_input_and_best_energy() -> None:
    original = sylvester(4)
    broken = original.copy()
    broken[1, 1] *= -1
    before = broken.copy()
    initial_energy = check_orthogonality(broken)["energy"]
    refined, metrics, _ = CASearch(order=4).refine(broken, steps=4, seed=0)
    assert np.array_equal(broken, before)
    assert metrics["energy"] <= initial_energy
    assert np.all(np.isin(refined, (-1, 1)))


def test_ca_runs_as_a_pipeline_refinement() -> None:
    class Source(SearchStrategy):
        ORDER = 4

        @property
        def name(self) -> str:
            return "source"

        def search(self, steps: int, seed: int) -> tuple[np.ndarray, dict[str, int], float]:
            matrix = sylvester(4)
            matrix[1, 1] *= -1
            return matrix, check_orthogonality(matrix), 0.0

    _, metrics, _ = Pipeline([(Source(), 0), (CASearch(order=4), 2)]).search(0, 0)
    assert metrics["energy"] <= 48


def test_baumert_hall_solves_order_four() -> None:
    matrix, metrics, _ = BaumertHallSearch(
        ORDER=4, T=1, HALF=1).search(steps=0, seed=0)
    assert matrix.shape == (4, 4)
    assert metrics["energy"] == 0


def test_baumert_hall_rejects_even_sequence_order() -> None:
    with pytest.raises(ValueError, match="odd T"):
        BaumertHallSearch(ORDER=8, T=2, HALF=1)


def test_baumert_hall_weighted_energy_matches_built_matrix() -> None:
    strategy = BaumertHallSearch(ORDER=12, T=3, HALF=2)
    a = symmetric_circulant(np.array((1, -1), dtype=np.int8))
    b = symmetric_circulant(np.array((1, 1), dtype=np.int8))
    c = symmetric_circulant(np.array((-1, 1), dtype=np.int8))
    proxy = strategy._bh_energy(a, b, c)
    assert check_orthogonality(strategy._build(a, b, c))["energy"] == 12 * proxy


def test_baumert_incremental_weighted_deltas_match_reference() -> None:
    strategy = BaumertHallSearch(ORDER=28, T=7, HALF=4)
    rng = np.random.default_rng(4)
    sequences = np.stack([
        symmetric_circulant(rng.choice([-1, 1], size=4).astype(np.int8))
        for _ in range(3)
    ])
    weights = np.array((1, 2, 1), dtype=np.int64)
    correlations = autocorrelation_state(sequences, weights)
    for sequence, half_index in rng.integers((0, 0), (3, 4), size=(20, 2)):
        energy = apply_symmetric_flip(
            sequences, correlations, int(sequence), int(half_index),
            weight=int(weights[sequence])) // 2
        assert energy == strategy._bh_energy(*sequences)


def test_diffset_incremental_deltas_match_reference() -> None:
    strategy = DiffsetSearch(28)
    rng = np.random.default_rng(5)
    sequences = np.stack(strategy._seed())
    correlations = autocorrelation_state(sequences)
    for sequence, index in rng.integers((0, 0), (4, 7), size=(20, 2)):
        energy = apply_sequence_flip(
            sequences, correlations, int(sequence), int(index))
        assert energy == periodic_autocorrelation_energy(tuple(sequences))
        assert energy == correlation_energy(correlations)


@pytest.mark.parametrize(("seed", "population_size", "length"), [
    (0, 2, 1), (1, 5, 3), (2, 17, 7), (3, 128, 7),
])
def test_genetic_batch_fitness_matches_individual_reference(
    seed: int, population_size: int, length: int,
) -> None:
    population = np.random.default_rng(seed).choice(
        [-1, 1], size=(population_size, 4, length)).astype(np.int8)
    strategy = GeneticSearch(order=4 * length, population_size=population_size)
    expected = np.array([
        reference_periodic_energy(tuple(candidate))
        for candidate in population
    ], dtype=np.int64)
    assert np.array_equal(strategy._energies(population), expected)


def test_genetic_search_does_not_use_individual_fitness(monkeypatch) -> None:
    strategy = GeneticSearch(order=12, population_size=4)

    def fail(_candidate):
        raise AssertionError("search must evaluate the population as one batch")

    monkeypatch.setattr(strategy, "_energy", fail)
    strategy.search(steps=2, seed=0)


@pytest.mark.skipif(xp.__name__ != "cupy", reason="requires active CuPy backend")
def test_montecarlo_incremental_batch_step_matches_full_fft() -> None:
    host = np.random.default_rng(6).choice(
        [-1, 1], size=(5, 4, 7)).astype(np.int8)
    batch = xp.asarray(host)
    correlations = MonteCarloSearch._correlations(batch)
    energies = MonteCarloSearch._energies(correlations)
    initial_energies = to_numpy(energies).copy()
    sequence = xp.asarray([0, 1, 2, 3, 1])
    column = xp.asarray([0, 2, 4, 6, 3])
    forward, backward = MonteCarloSearch._shift_indices(7)
    MonteCarloSearch._apply_flips(
        batch, correlations, energies, sequence, column, forward, backward)

    candidates = host.copy()
    rows = np.arange(len(host))
    candidates[rows, to_numpy(sequence), to_numpy(column)] *= -1
    candidate_correlations = MonteCarloSearch._correlations(xp.asarray(candidates))
    candidate_energies = np.array([
        reference_periodic_energy(tuple(candidate))
        for candidate in candidates
    ], dtype=np.int64)
    accepted = candidate_energies <= initial_energies
    expected_batch = host.copy()
    expected_batch[accepted] = candidates[accepted]
    expected_correlations = MonteCarloSearch._correlations(xp.asarray(host))
    expected_correlations[accepted] = candidate_correlations[accepted]
    expected_energies = np.where(
        accepted, candidate_energies, initial_energies)

    assert np.array_equal(to_numpy(batch), expected_batch)
    assert np.array_equal(to_numpy(correlations),
                          to_numpy(expected_correlations))
    assert np.array_equal(to_numpy(energies), expected_energies)
