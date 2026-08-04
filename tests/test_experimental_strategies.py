"""Contracts for experimental and Turyn search strategies."""
from __future__ import annotations

import numpy as np
import pytest

from correlations import periodic_autocorrelation_energy
from fourier import project_power_complementarity, project_weighted_nonperiodic_power
from gpu import check_orthogonality, to_numpy, xp
from strategies.greedy import TurynGreedySearch
from strategies.annealing import TurynAnnealingSearch
from strategies.genetic import GeneticSearch
from strategies.ising import IsingSearch
from strategies.montecarlo import MonteCarloSearch
from strategies.spectral_descent import TurynSpectralDescentSearch
from strategies.pocs import PocsSearch
from strategies.steepest import TurynSteepestSearch


def test_periodic_autocorrelation_of_singletons_is_zero() -> None:
    sequence = np.array((1,), dtype=np.int8)
    assert periodic_autocorrelation_energy((sequence,) * 4) == 0


def test_ising_search_is_seeded() -> None:
    first, _, _ = IsingSearch(92).search(steps=3, seed=4)
    second, _, _ = IsingSearch(92).search(steps=3, seed=4)
    assert np.array_equal(first, second)


def test_genetic_search_returns_a_turyn_candidate() -> None:
    matrix, metrics, _ = GeneticSearch(
        92, population_size=4).search(steps=2, seed=0)
    assert matrix.shape == (92, 92)
    assert metrics == check_orthogonality(matrix)


def test_shared_power_projection_satisfies_power_condition() -> None:
    state = np.random.default_rng(10).normal(size=(4, 7))
    projected = project_power_complementarity(state)
    power = np.sum(np.abs(np.fft.fft(projected, axis=1)) ** 2, axis=0)
    assert np.allclose(power, 28.0, atol=1e-5)


def test_weighted_nonperiodic_projection_supports_batches() -> None:
    state = np.random.default_rng(10).normal(size=(3, 4, 8))
    lengths = np.array((8, 8, 8, 7))
    weights = np.array((1, 1, 2, 2))
    projected = project_weighted_nonperiodic_power(
        state, lengths=lengths, weights=weights)
    expected = np.stack([
        project_weighted_nonperiodic_power(
            candidate, lengths=lengths, weights=weights)
        for candidate in state
    ])
    assert projected.shape == state.shape
    assert np.all(projected[:, 3, -1] == 0)
    assert np.allclose(projected, expected)


def test_turyn_pocs_gradient_is_finite() -> None:
    strategy = TurynSpectralDescentSearch(n=8)
    gradient = strategy._gradient(strategy._seed(np.random.default_rng(2)))
    assert gradient.shape == (4, 8)
    assert np.all(np.isfinite(gradient))


@pytest.mark.parametrize("strategy", [
    TurynGreedySearch(n=8), TurynAnnealingSearch(n=8),
    TurynSpectralDescentSearch(n=8), TurynSteepestSearch(n=8, candidates=8),
])
def test_turyn_strategies_follow_result_contract(strategy) -> None:
    matrix, metrics, elapsed = strategy.search(steps=1, seed=1)
    assert matrix.shape == (92, 92)
    assert matrix.dtype == np.int8
    assert np.all(np.isin(matrix, (-1, 1)))
    assert metrics == check_orthogonality(matrix)
    assert isinstance(elapsed, float)


@pytest.mark.skipif(xp.__name__ != "cupy", reason="requires active CuPy backend")
def test_montecarlo_search_uses_gpu() -> None:
    matrix, metrics, _ = MonteCarloSearch(
        92, batch_size=4).search(steps=2, seed=0)
    assert matrix.shape == (92, 92)
    assert metrics == check_orthogonality(matrix)
    assert isinstance(to_numpy(matrix), np.ndarray)


@pytest.mark.skipif(xp.__name__ != "cupy", reason="requires active CuPy backend")
def test_montecarlo_persistent_turyn_delta_matches_full_fft() -> None:
    strategy = MonteCarloSearch(92, batch_size=5)
    host = np.random.default_rng(7).choice(
        (-1, 1), size=(5, 4, 8)).astype(np.int8)
    host[:, 3, -1] = 0
    batch = xp.asarray(host)
    correlations = strategy._correlations(batch)
    energies = strategy._energies_from_correlations(correlations)
    sequence = xp.asarray((0, 1, 2, 3, 3), dtype=xp.int64)
    column = xp.asarray((0, 2, 4, 5, 1), dtype=xp.int64)
    strategy._apply_flips(
        batch, correlations, energies, sequence, column,
        strategy.LENGTHS, strategy.WEIGHTS)
    expected = strategy._correlations(batch)
    assert np.array_equal(to_numpy(correlations), to_numpy(expected))
    assert np.array_equal(
        to_numpy(energies), to_numpy(strategy._energies_from_correlations(expected)))
