"""Contracts for the experimental search strategies."""
from __future__ import annotations

import numpy as np
import pytest

from constructions import periodic_autocorrelation_energy, symmetric_circulant
from gpu import check_orthogonality, xp
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
