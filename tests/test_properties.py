"""Generative property tests for Hadamard constructions and result contracts."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

import constructions
from constructions import (
    apply_sequence_flip,
    apply_symmetric_flip,
    autocorrelation_energy,
    autocorrelation_state,
    build_goethals_seidel,
    build_propus,
    build_williamson,
    circulant,
    correlation_energy,
    periodic_autocorrelation_energy,
    symmetric_circulant,
)
from gpu import check_orthogonality
from output import save_run
from strategies.annealing import AnnealingSearch
from strategies.base import Pipeline, SearchStrategy
from strategies.direct import DirectSearch
from strategies.repair import RepairSearch
from verifier.known import paley, sylvester
from verifier.review import review_bundle
from verifier.validate import InvalidManifest, load_and_validate_manifest


@st.composite
def sign_sequences(draw: st.DrawFn) -> np.ndarray:
    length = draw(st.integers(min_value=1, max_value=8))
    values = draw(st.lists(st.sampled_from((-1, 1)),
                  min_size=length, max_size=length))
    return np.array(values, dtype=np.int8)


@st.composite
def four_sign_sequences(draw: st.DrawFn) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    length = draw(st.integers(min_value=1, max_value=7))
    values = st.lists(st.sampled_from((-1, 1)),
                      min_size=length, max_size=length)
    return (
        np.array(draw(values), dtype=np.int8),
        np.array(draw(values), dtype=np.int8),
        np.array(draw(values), dtype=np.int8),
        np.array(draw(values), dtype=np.int8),
    )


@st.composite
def four_symmetric_halves(draw: st.DrawFn) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    length = draw(st.integers(min_value=1, max_value=5))
    values = st.lists(st.sampled_from((-1, 1)),
                      min_size=length, max_size=length)
    return (
        np.array(draw(values), dtype=np.int8),
        np.array(draw(values), dtype=np.int8),
        np.array(draw(values), dtype=np.int8),
        np.array(draw(values), dtype=np.int8),
    )


@st.composite
def sign_matrices(draw: st.DrawFn) -> np.ndarray:
    order = draw(st.integers(min_value=1, max_value=12))
    values = draw(st.lists(st.sampled_from((-1, 1)),
                  min_size=order * order, max_size=order * order))
    return np.array(values, dtype=np.int8).reshape(order, order)


def reference_metrics(matrix: np.ndarray) -> dict[str, int]:
    gram = matrix.astype(np.int64) @ matrix.astype(np.int64).T
    upper = gram[np.triu_indices(matrix.shape[0], k=1)]
    return {
        "energy": int(np.dot(upper, upper)),
        "orthogonal_pairs": int(np.count_nonzero(upper == 0)),
        "max_abs_correlation": int(np.abs(upper).max(initial=0)),
    }


def reference_autocorrelation_energy(sequences: tuple[np.ndarray, ...]) -> int:
    size = len(sequences[0])
    total = np.zeros(size, dtype=np.int64)
    for displacement in range(size):
        total[displacement] = sum(
            int(np.dot(sequence, np.roll(sequence, -displacement)))
            for sequence in sequences)
    return int(np.sum(total[1:(size + 1) // 2] ** 2))


def reference_correlations(
    sequences: tuple[np.ndarray, ...] | np.ndarray,
    weights: tuple[int, ...] | None = None,
) -> np.ndarray:
    values = tuple(sequences)
    actual_weights = weights or (1,) * len(values)
    size = len(values[0])
    total = np.zeros(size, dtype=np.int64)
    for displacement in range(size):
        total[displacement] = sum(
            weight * int(np.dot(sequence, np.roll(sequence, -displacement)))
            for sequence, weight in zip(values, actual_weights)
        )
    return total


@settings(max_examples=50, deadline=None)
@given(sign_sequences())
def test_symmetric_circulant_properties(half: np.ndarray) -> None:
    sequence = symmetric_circulant(half)
    assert len(sequence) == 2 * len(half) - 1
    assert np.array_equal(sequence[:len(half)], half)
    matrix = circulant(sequence)
    assert matrix.shape == (len(sequence), len(sequence))
    assert np.array_equal(matrix[0], sequence)
    assert np.array_equal(matrix, matrix.T)


@settings(max_examples=30, deadline=None)
@given(four_sign_sequences())
def test_construction_properties(sequences: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> None:
    order = len(sequences[0])
    for build in (build_williamson, build_goethals_seidel):
        matrix = build(*sequences)
        assert matrix.shape == (4 * order, 4 * order)
        assert matrix.dtype == np.int8
        assert np.all(np.isin(matrix, (-1, 1)))
    propus = build_propus(sequences[0], sequences[1], sequences[3])
    assert propus.shape == (4 * order, 4 * order)
    assert propus.dtype == np.int8
    assert np.all(np.isin(propus, (-1, 1)))
    assert autocorrelation_energy(
        sequences) == reference_autocorrelation_energy(sequences)


@settings(max_examples=30, deadline=None)
@given(four_symmetric_halves())
def test_compact_energy_matches_valid_block_constructions(
    halves: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    a, b, c, d = (symmetric_circulant(half) for half in halves)
    order = 4 * len(a)
    four_energy = autocorrelation_energy((a, b, c, d))
    assert check_orthogonality(build_williamson(a, b, c, d))["energy"] == order * four_energy
    assert check_orthogonality(build_goethals_seidel(a, b, c, d))["energy"] == order * four_energy
    propus_energy = autocorrelation_energy((a, b, b, d))
    assert check_orthogonality(build_propus(a, b, d))["energy"] == order * propus_energy


@settings(max_examples=30, deadline=None)
@given(
    four_sign_sequences(),
    st.lists(st.tuples(st.integers(0, 3), st.integers(0, 100)),
             min_size=0, max_size=20),
)
def test_incremental_sequence_flips_match_full_recomputation(
    original: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    raw_flips: list[tuple[int, int]],
) -> None:
    sequences = np.stack(original)
    initial = sequences.copy()
    correlations = autocorrelation_state(sequences)
    flips = [(sequence_index, raw_index % sequences.shape[1])
             for sequence_index, raw_index in raw_flips]
    for sequence_index, value_index in flips:
        energy = apply_sequence_flip(
            sequences, correlations, sequence_index, value_index)
        expected = reference_correlations(sequences)
        assert np.array_equal(correlations, expected)
        assert energy == periodic_autocorrelation_energy(tuple(sequences))
    for sequence_index, value_index in reversed(flips):
        apply_sequence_flip(
            sequences, correlations, sequence_index, value_index)
    assert np.array_equal(sequences, initial)
    assert np.array_equal(correlations, reference_correlations(initial))


@settings(max_examples=30, deadline=None)
@given(
    four_symmetric_halves(),
    st.lists(st.tuples(st.integers(0, 3), st.integers(0, 100)),
             min_size=0, max_size=20),
)
def test_incremental_symmetric_flips_preserve_circulant_symmetry(
    halves: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    raw_flips: list[tuple[int, int]],
) -> None:
    sequences = np.stack([symmetric_circulant(half) for half in halves])
    initial = sequences.copy()
    correlations = autocorrelation_state(sequences)
    half_size = len(halves[0])
    flips = [(sequence_index, raw_index % half_size)
             for sequence_index, raw_index in raw_flips]
    for sequence_index, half_index in flips:
        energy = apply_symmetric_flip(
            sequences, correlations, sequence_index, half_index)
        assert np.array_equal(correlations, reference_correlations(sequences))
        assert energy == correlation_energy(correlations)
        assert all(np.array_equal(circulant(sequence), circulant(sequence).T)
                   for sequence in sequences)
    for sequence_index, half_index in reversed(flips):
        apply_symmetric_flip(
            sequences, correlations, sequence_index, half_index)
    assert np.array_equal(sequences, initial)
    assert np.array_equal(correlations, reference_correlations(initial))


@settings(max_examples=30, deadline=None)
@given(four_sign_sequences())
def test_weighted_autocorrelation_state_matches_reference(
    sequences: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    weights = np.array((1, 2, 1, 3), dtype=np.int64)
    assert np.array_equal(
        autocorrelation_state(sequences, weights),
        reference_correlations(sequences, tuple(int(value) for value in weights)),
    )


def test_weighted_incremental_flip_matches_reference() -> None:
    sequences = np.array([
        [1, -1, 1],
        [-1, 1, 1],
        [1, 1, -1],
    ], dtype=np.int8)
    weights = np.array((1, 2, 1), dtype=np.int64)
    correlations = autocorrelation_state(sequences, weights)
    apply_sequence_flip(sequences, correlations, 1, 2, weight=2)
    assert np.array_equal(
        correlations, reference_correlations(sequences, (1, 2, 1)))


@pytest.mark.parametrize("half_index", [0, 1, 3])
def test_symmetric_flip_handles_first_inner_and_last_half_index(
    half_index: int,
) -> None:
    half = np.array((1, -1, 1, -1), dtype=np.int8)
    sequences = np.stack([symmetric_circulant(half) for _ in range(4)])
    correlations = autocorrelation_state(sequences)
    apply_symmetric_flip(sequences, correlations, 0, half_index)
    assert np.array_equal(correlations, reference_correlations(sequences))
    assert np.array_equal(circulant(sequences[0]), circulant(sequences[0]).T)


def test_autocorrelation_energy_falls_back_to_numpy(monkeypatch) -> None:
    sequences = tuple(np.array([1, -1, 1], dtype=np.int8) for _ in range(4))
    monkeypatch.setattr(constructions, "NUMBA_AVAILABLE", False)
    assert constructions.autocorrelation_energy(
        sequences) == reference_autocorrelation_energy(sequences)
    matrix = np.stack(sequences)
    correlations = constructions.autocorrelation_state(matrix)
    constructions.apply_sequence_flip(matrix, correlations, 2, 1)
    assert np.array_equal(correlations, reference_correlations(matrix))


def test_goethals_seidel_slicing_matches_permutation_matrix_reference() -> None:
    sequences = tuple(np.array([1, -1, 1], dtype=np.int8)
                      for _ in range(4))
    A, B, C, D = (circulant(sequence) for sequence in sequences)
    reverse = np.fliplr(np.eye(3, dtype=np.int8))
    BR, CR, DR = B @ reverse, C @ reverse, D @ reverse
    expected = np.block([
        [A, BR, CR, DR],
        [-BR, A, -DR.T, CR.T],
        [-CR, DR.T, A, -BR.T],
        [-DR, -CR.T, BR.T, A],
    ]).astype(np.int8)
    assert np.array_equal(build_goethals_seidel(*sequences), expected)


@settings(max_examples=50, deadline=None)
@given(sign_matrices())
def test_metrics_match_independent_reference(matrix: np.ndarray) -> None:
    assert check_orthogonality(matrix) == reference_metrics(matrix)


def test_refinement_preserves_input_when_no_steps_are_requested() -> None:
    candidate = sylvester(4)
    for strategy in (DirectSearch(order=4), RepairSearch(order=4)):
        refined, metrics, _ = strategy.refine(candidate, steps=0, seed=0)
        assert np.array_equal(candidate, sylvester(4))
        assert np.array_equal(refined, candidate)
        assert metrics == check_orthogonality(candidate)


def test_annealing_refinement_requires_williamson_matrix() -> None:
    strategy = AnnealingSearch(ORDER=4, K=1, HALF=1)
    williamson = build_williamson(
        *(np.ones(1, dtype=np.int8) for _ in range(4)))
    refined, metrics, _ = strategy.refine(williamson, steps=0, seed=0)
    assert np.array_equal(refined, williamson)
    assert metrics["energy"] == 0
    with pytest.raises(ValueError, match="Williamson"):
        strategy.refine(np.ones((4, 4), dtype=np.int8), steps=0, seed=0)


def test_pipeline_passes_matrix_and_incremented_seed() -> None:
    class Source(SearchStrategy):
        ORDER = 2

        @property
        def name(self) -> str:
            return "source"

        def search(self, steps: int, seed: int) -> tuple[np.ndarray, dict[str, int], float]:
            assert (steps, seed) == (3, 20)
            return np.array([[1, 1], [1, -1]], dtype=np.int8), {"energy": 1, "orthogonal_pairs": 0, "max_abs_correlation": 2}, 0.1

    class Sink(SearchStrategy):
        ORDER = 2

        @property
        def name(self) -> str:
            return "sink"

        def search(self, steps: int, seed: int) -> tuple[np.ndarray, dict[str, int], float]:
            raise AssertionError("Pipeline must call refine on later stages")

        def refine(self, matrix: np.ndarray, steps: int, seed: int) -> tuple[np.ndarray, dict[str, int], float]:
            assert np.array_equal(matrix, np.array(
                [[1, 1], [1, -1]], dtype=np.int8))
            assert (steps, seed) == (5, 21)
            return matrix, {"energy": 1, "orthogonal_pairs": 0, "max_abs_correlation": 2}, 0.2

    matrix, metrics, elapsed = Pipeline(
        [(Source(), 3), (Sink(), 5)]).search(steps=0, seed=20)
    assert matrix.shape == (2, 2)
    assert metrics["energy"] == 1
    assert np.isclose(elapsed, 0.3)


@pytest.mark.parametrize("matrix", [sylvester(4), paley(12)])
def test_output_manifest_and_review(matrix: np.ndarray, tmp_path) -> None:
    save_run(
        matrix,
        check_orthogonality(matrix),
        tmp_path,
        method_family="williamson_propus",
        search_scope="property-test",
        seed=0,
        steps=1,
        wall_seconds=0.0,
        hardware_summary="test",
    )
    manifest_path = tmp_path / "run.json"
    manifest = load_and_validate_manifest(manifest_path)
    assert manifest["candidate_sha256"] == hashlib.sha256(
        (tmp_path / "candidate.csv").read_bytes()).hexdigest()
    assert review_bundle(tmp_path, len(matrix))["exact_solution"]


def test_manifest_rejects_unknown_method_family(tmp_path) -> None:
    matrix = sylvester(4)
    save_run(matrix, check_orthogonality(matrix), tmp_path, method_family="williamson_propus",
             search_scope="test", seed=0, steps=1, wall_seconds=0.0, hardware_summary="test")
    manifest_path = tmp_path / "run.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data["method_family"] = "not-a-method"
    manifest_path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(InvalidManifest):
        load_and_validate_manifest(manifest_path)
