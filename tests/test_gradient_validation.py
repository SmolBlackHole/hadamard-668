"""Numerical validation of the continuous TT gradient used for L-BFGS repair."""

from __future__ import annotations

import numpy as np
import pytest

from correlations import (
    TURYN_WEIGHTS,
    apply_nonperiodic_flip,
    nonperiodic_autocorrelation_state,
    nonperiodic_correlation_energy,
    npa_f_gradient,
)


def _continuous_energy(values: np.ndarray, lengths: np.ndarray, weights: np.ndarray) -> float:
    correlations = np.zeros(len(values[0]), dtype=np.float64)
    for sequence, length, weight in zip(values, lengths, weights, strict=False):
        for lag in range(1, int(length)):
            correlations[lag] += weight * np.dot(sequence[: length - lag], sequence[lag:length])
    return float(np.dot(correlations[1:], correlations[1:]))


@pytest.mark.parametrize("n", (4, 6, 8))
def test_gradient_matches_finite_differences_and_directional_derivative(n: int) -> None:
    lengths = np.array((n, n, n, n - 1), dtype=np.int64)
    rng = np.random.default_rng(n)
    values = rng.normal(size=(4, n))
    values[3, -1] = 0.0
    gradient = npa_f_gradient(values, lengths=lengths, weights=TURYN_WEIGHTS)
    finite_difference = np.zeros_like(values)
    epsilon = 1e-6
    for row, length in enumerate(lengths):
        for column in range(int(length)):
            values[row, column] += epsilon
            upper = _continuous_energy(values, lengths, TURYN_WEIGHTS)
            values[row, column] -= 2 * epsilon
            lower = _continuous_energy(values, lengths, TURYN_WEIGHTS)
            values[row, column] += epsilon
            finite_difference[row, column] = (upper - lower) / (2 * epsilon)
    active = np.ones_like(values, dtype=bool)
    active[3, -1] = False
    relative_error = np.linalg.norm((gradient - finite_difference)[active]) / np.linalg.norm(
        finite_difference[active]
    )
    direction = rng.normal(size=values.shape)
    direction[~active] = 0.0
    numerical_direction = (
        _continuous_energy(values + epsilon * direction, lengths, TURYN_WEIGHTS)
        - _continuous_energy(values - epsilon * direction, lengths, TURYN_WEIGHTS)
    ) / (2 * epsilon)
    assert relative_error < 1e-6
    assert np.isclose(numerical_direction, np.sum(gradient * direction), rtol=1e-6)


def test_gradient_ranks_exact_single_flip_deltas() -> None:
    correlations_by_seed = []
    hits = 0
    for seed in range(12):
        n = 8
        lengths = np.array((n, n, n, n - 1), dtype=np.int64)
        rng = np.random.default_rng(seed)
        sequences = np.zeros((4, n), dtype=np.int8)
        for i, L in enumerate(lengths):
            sequences[i, : int(L)] = rng.choice((-1, 1), size=int(L)).astype(np.int8)
        sequences[3, -1] = 0

        state = nonperiodic_autocorrelation_state(sequences, lengths=lengths, weights=TURYN_WEIGHTS)
        energy = nonperiodic_correlation_energy(state)
        gradient = npa_f_gradient(
            sequences.astype(np.float64), lengths=lengths, weights=TURYN_WEIGHTS
        )
        proxies, deltas = [], []
        for row, length in enumerate(lengths):
            for column in range(int(length)):
                candidate, candidate_state = sequences.copy(), state.copy()
                updated = apply_nonperiodic_flip(
                    candidate,
                    candidate_state,
                    row,
                    column,
                    lengths=lengths,
                    weight=int(TURYN_WEIGHTS[row]),
                )
                proxies.append(-2.0 * sequences[row, column] * gradient[row, column])
                deltas.append(updated - energy)
        proxy_ranks = np.argsort(np.argsort(proxies))
        delta_ranks = np.argsort(np.argsort(deltas))
        correlations_by_seed.append(np.corrcoef(proxy_ranks, delta_ranks)[0, 1])
        hits += bool(set(np.argsort(proxies)[:4]) & set(np.argsort(deltas)[:4]))
    assert float(np.median(correlations_by_seed)) > 0.7
    assert hits >= 9
