"""Contracts for experimental and Turyn search strategies."""
from __future__ import annotations

import numpy as np
import pytest

from correlations import periodic_autocorrelation_energy
from fourier import project_power_complementarity
from gpu import check_orthogonality, to_numpy, xp
from strategies.circulant import TurynGreedySearch
from strategies.annealing import TurynAnnealingSearch
from strategies.genetic import GeneticSearch
from strategies.ising import IsingSearch
from strategies.montecarlo import MonteCarloSearch
from strategies.pocs import TurynPocsSearch
from strategies.spectral import SpectralSearch
from strategies.turyn_steepest import TurynSteepestSearch


def test_periodic_autocorrelation_of_singletons_is_zero() -> None:
    sequence = np.array((1,), dtype=np.int8)
    assert periodic_autocorrelation_energy((sequence,) * 4) == 0


def test_ising_search_is_seeded() -> None:
    first, _, _ = IsingSearch(92).search(steps=3, seed=4)
    second, _, _ = IsingSearch(92).search(steps=3, seed=4)
    assert np.array_equal(first, second)


def test_genetic_search_returns_a_turyn_candidate() -> None:
    matrix, metrics, _ = GeneticSearch(92, population_size=4).search(steps=2, seed=0)
    assert matrix.shape == (92, 92)
    assert metrics == check_orthogonality(matrix)


def test_shared_power_projection_satisfies_power_condition() -> None:
    state = np.random.default_rng(10).normal(size=(4, 7))
    projected = project_power_complementarity(state)
    power = np.sum(np.abs(np.fft.fft(projected, axis=1)) ** 2, axis=0)
    assert np.allclose(power, 28.0, atol=1e-5)


def test_turyn_pocs_gradient_is_finite() -> None:
    strategy = TurynPocsSearch(n=8)
    gradient = strategy._gradient(strategy._seed(np.random.default_rng(2)))
    assert gradient.shape == (4, 8)
    assert np.all(np.isfinite(gradient))


@pytest.mark.parametrize("strategy", [
    TurynGreedySearch(n=8), TurynAnnealingSearch(n=8),
    TurynPocsSearch(n=8), TurynSteepestSearch(n=8, candidates=8),
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
