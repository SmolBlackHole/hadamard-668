"""Contracts for the experimental search strategies."""
from __future__ import annotations

import numpy as np
import pytest

from constructions import (
    apply_sequence_flip,
    apply_symmetric_flip,
    autocorrelation_state,
    build_goethals_seidel,
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
from strategies.rowwise import RowwiseSearch
from strategies.sat import SatSearch
from strategies.spectral import SpectralSearch
from strategies.walsh import WalshSearch, _fwht
from strategies.base import Pipeline, SearchStrategy


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


@pytest.mark.parametrize("order", [4, 8, 12])
def test_sat_search_solves_small_known_orders(order: int) -> None:
    _, metrics, _ = SatSearch(order, timeout_seconds=5).search(
        steps=1, seed=0)
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


def test_spectral_fourier_projection_satisfies_power_condition() -> None:
    state = np.random.default_rng(0).normal(size=(4, 7)).astype(np.float32)
    projected = SpectralSearch._project_fourier(state)
    power = np.sum(np.abs(np.fft.fft(projected, axis=1)) ** 2, axis=0)
    assert np.allclose(power, 28.0, atol=1e-5)
    assert np.isrealobj(projected)


def test_spectral_zero_projection_is_deterministic() -> None:
    state = np.zeros((4, 7), dtype=np.float32)
    first = SpectralSearch._project_fourier(state)
    second = SpectralSearch._project_fourier(state)
    assert np.array_equal(first, second)
    assert np.all(np.isfinite(first))


def test_spectral_douglas_rachford_step_uses_averaged_reflections() -> None:
    state = np.random.default_rng(1).normal(size=(4, 5)).astype(np.float32)
    orthogonal = SpectralSearch._project_fourier(state)
    reflected_orthogonal = 2 * orthogonal - state
    signed = SpectralSearch._project_sign(reflected_orthogonal)
    expected = 0.5 * (state + 2 * signed - reflected_orthogonal)
    assert np.allclose(
        SpectralSearch._douglas_rachford_step(state), expected)


def test_ising_gradient_matches_finite_differences() -> None:
    state = np.random.default_rng(2).normal(size=(4, 5)).astype(np.float32)
    energy, gradient = IsingSearch._energy_gradient(state)
    epsilon = 1e-3
    numerical = np.empty_like(state)
    for sequence in range(4):
        for index in range(5):
            shifted = state.copy()
            shifted[sequence, index] += epsilon
            shifted_energy, _ = IsingSearch._energy_gradient(shifted)
            numerical[sequence, index] = (shifted_energy - energy) / epsilon
    assert np.allclose(gradient, numerical, rtol=2e-3, atol=2e-2)


def test_ca_identity_rule_preserves_sequences() -> None:
    sequences = np.random.default_rng(3).choice(
        (-1, 1), size=(4, 7)).astype(np.int8)
    rule = np.array((0, 1, 0), dtype=np.float32)
    assert np.array_equal(CASearch._apply_ca(sequences, rule), sequences)


def test_ca_identity_spectral_filter_preserves_sequences() -> None:
    sequences = np.random.default_rng(4).choice(
        (-1, 1), size=(4, 7)).astype(np.int8)
    weights = np.ones(7 // 2 + 1)
    assert np.array_equal(
        CASearch._apply_spectral(sequences, weights), sequences)


def test_ca_rule_mutation_changes_one_weight() -> None:
    rule = np.zeros(5)
    mutated = CASearch._mutate_rule(rule, np.random.default_rng(0))
    assert np.count_nonzero(mutated != rule) == 1
    assert np.all(np.isin(mutated, (-1.0, 0.0, 1.0)))


def test_ca_refinement_preserves_input_and_best_energy() -> None:
    sequences = np.random.default_rng(5).choice(
        (-1, 1), size=(4, 3)).astype(np.int8)
    original = build_goethals_seidel(*sequences)
    before = original.copy()
    refined, metrics, _ = CASearch(order=12).refine(
        original, steps=4, seed=0)
    assert np.array_equal(original, before)
    assert metrics == check_orthogonality(refined)
    assert np.all(np.isin(refined, (-1, 1)))


def test_ca_refinement_rejects_incompatible_matrix() -> None:
    with pytest.raises(ValueError, match="Goethals-Seidel"):
        CASearch(order=4).refine(
            np.ones((4, 4), dtype=np.int8), steps=1, seed=0)


def test_ca_runs_as_a_pipeline_refinement() -> None:
    class Source(SearchStrategy):
        ORDER = 12

        @property
        def name(self) -> str:
            return "source"

        def search(self, steps: int, seed: int) -> tuple[np.ndarray, dict[str, int], float]:
            sequences = np.random.default_rng(6).choice(
                (-1, 1), size=(4, 3)).astype(np.int8)
            matrix = build_goethals_seidel(*sequences)
            return matrix, check_orthogonality(matrix), 0.0

    matrix, metrics, _ = Pipeline(
        [(Source(), 0), (CASearch(order=12), 2)]).search(0, 0)
    assert metrics == check_orthogonality(matrix)


def test_walsh_transform_round_trip() -> None:
    values = np.random.default_rng(7).normal(size=(4, 8)).astype(np.float32)
    assert np.allclose(_fwht(_fwht(values)) / 8, values)


def test_walsh_sequence_mutation_stays_compact_and_binary() -> None:
    sequences = np.random.default_rng(8).choice(
        (-1, 1), size=(4, 7)).astype(np.int8)
    mutated = WalshSearch._mutate_sequences(
        sequences, 8, np.random.default_rng(9))
    assert mutated.shape == (4, 7)
    assert mutated.dtype == np.int8
    assert np.all(np.isin(mutated, (-1, 1)))


def test_sat_uses_four_compact_sequences_as_base_variables() -> None:
    strategy = SatSearch(28, timeout_seconds=1)
    variables = strategy._variables()
    assert len(variables) == 4
    assert sum(map(len, variables)) == 28


@pytest.mark.parametrize("strategy", [
    SpectralSearch(ORDER=4, inner_steps=1),
    IsingSearch(4),
    CASearch(4),
    WalshSearch(4),
    SatSearch(4, timeout_seconds=1),
    GoldSearch(4),
    RowwiseSearch(4),
])
def test_phase6_strategies_follow_result_contract(strategy) -> None:
    matrix, metrics, elapsed = strategy.search(steps=1, seed=1)
    assert isinstance(matrix, np.ndarray)
    assert matrix.shape == (4, 4)
    assert matrix.dtype == np.int8
    assert np.all(np.isin(matrix, (-1, 1)))
    assert all(isinstance(value, int) for value in metrics.values())
    assert isinstance(elapsed, float)


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
