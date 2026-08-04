"""Numerical validation of the continuous TT gradient used for POCS ranking."""
from __future__ import annotations

import numpy as np
import pytest

from correlations import (apply_nonperiodic_flip, nonperiodic_autocorrelation_state,
                          nonperiodic_correlation_energy)
from strategies.spectral_descent import TurynSpectralDescentSearch as SFDS


def _continuous_energy(values: np.ndarray, lengths: np.ndarray, weights: np.ndarray) -> float:
    correlations = np.zeros(len(values[0]), dtype=np.float64)
    for sequence, length, weight in zip(values, lengths, weights):
        for lag in range(1, int(length)):
            correlations[lag] += weight * np.dot(
                sequence[:length - lag], sequence[lag:length])
    return float(np.dot(correlations[1:], correlations[1:]))


@pytest.mark.parametrize("n", (4, 6, 8))
def test_pocs_gradient_matches_finite_differences_and_directional_derivative(n: int) -> None:
    strategy = SFDS(n=n, sieve=False)
    rng = np.random.default_rng(n)
    values = rng.normal(size=(4, n))
    values[3, -1] = 0.0
    gradient = strategy._gradient(values)
    finite_difference = np.zeros_like(values)
    epsilon = 1e-6
    for row, length in enumerate(strategy.LENGTHS):
        for column in range(int(length)):
            values[row, column] += epsilon
            upper = _continuous_energy(values, strategy.LENGTHS, strategy.WEIGHTS)
            values[row, column] -= 2 * epsilon
            lower = _continuous_energy(values, strategy.LENGTHS, strategy.WEIGHTS)
            values[row, column] += epsilon
            finite_difference[row, column] = (upper - lower) / (2 * epsilon)
    active = np.ones_like(values, dtype=bool)
    active[3, -1] = False
    relative_error = np.linalg.norm(
        (gradient - finite_difference)[active]) / np.linalg.norm(finite_difference[active])
    direction = rng.normal(size=values.shape)
    direction[~active] = 0.0
    numerical_direction = (
        _continuous_energy(values + epsilon * direction,
                           strategy.LENGTHS, strategy.WEIGHTS)
        - _continuous_energy(values - epsilon * direction,
                             strategy.LENGTHS, strategy.WEIGHTS)
    ) / (2 * epsilon)
    assert relative_error < 1e-6
    assert np.isclose(numerical_direction, np.sum(gradient * direction), rtol=1e-6)


def test_pocs_gradient_ranks_exact_single_flip_deltas() -> None:
    correlations_by_seed = []
    hits = 0
    for seed in range(12):
        strategy = SFDS(n=8, sieve=False, candidates=4)
        sequences = strategy._seed(np.random.default_rng(seed))
        state = nonperiodic_autocorrelation_state(
            sequences, lengths=strategy.LENGTHS, weights=strategy.WEIGHTS)
        energy = nonperiodic_correlation_energy(state)
        gradient = strategy._gradient(sequences)
        proxies, deltas = [], []
        for row, length in enumerate(strategy.LENGTHS):
            for column in range(int(length)):
                candidate, candidate_state = sequences.copy(), state.copy()
                updated = apply_nonperiodic_flip(
                    candidate, candidate_state, row, column,
                    lengths=strategy.LENGTHS, weight=int(strategy.WEIGHTS[row]))
                proxies.append(-2.0 * sequences[row, column] * gradient[row, column])
                deltas.append(updated - energy)
        proxy_ranks = np.argsort(np.argsort(proxies))
        delta_ranks = np.argsort(np.argsort(deltas))
        correlations_by_seed.append(
            np.corrcoef(proxy_ranks, delta_ranks)[0, 1])
        hits += bool(set(np.argsort(proxies)[:4])
                     & set(np.argsort(deltas)[:4]))
    assert float(np.median(correlations_by_seed)) > 0.7
    assert hits >= 9
