"""Generative property tests for Hadamard constructions and result contracts."""
from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

import correlations
from builders import (
    base_to_t_sequences,
    build_goethals_seidel,
    build_propus,
    build_turyn,
    t_sequences_to_sign_sequences,
    turyn_to_base,
)
from correlations import (
    apply_nonperiodic_flip,
    apply_sequence_flip,
    apply_symmetric_flip,
    autocorrelation_state,
    correlation_energy,
    expand_symmetric_sequence,
    nonperiodic_autocorrelation_state,
    nonperiodic_correlation_energy,
    periodic_autocorrelation_energy,
)
from gpu import (
    apply_entry_flip,
    check_orthogonality,
    correlation_histogram,
    entry_flip_delta,
    entry_flip_deltas,
    gram_matrix,
    metrics_from_gram,
    to_numpy,
    xp,
)
from output import save_run
from strategies.base import Pipeline, SearchStrategy
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
def test_expand_symmetric_sequence_properties(half: np.ndarray) -> None:
    sequence = expand_symmetric_sequence(half)
    assert len(sequence) == 2 * len(half) - 1
    assert np.array_equal(sequence[:len(half)], half)
    assert np.array_equal(sequence[1:], sequence[:0:-1])


@settings(max_examples=30, deadline=None)
@given(four_sign_sequences())
def test_construction_properties(sequences: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> None:
    order = len(sequences[0])
    matrix = build_goethals_seidel(*sequences)
    assert matrix.shape == (4 * order, 4 * order)
    assert matrix.dtype == np.int8
    assert np.all(np.isin(matrix, (-1, 1)))
    propus = build_propus(sequences[0], sequences[1], sequences[3])
    assert propus.shape == (4 * order, 4 * order)
    assert propus.dtype == np.int8
    assert np.all(np.isin(propus, (-1, 1)))


@settings(max_examples=30, deadline=None)
@given(four_symmetric_halves())
def test_compact_energy_matches_valid_block_constructions(
    halves: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    a, b, c, d = (expand_symmetric_sequence(half) for half in halves)
    order = 4 * len(a)
    four_energy = correlation_energy(autocorrelation_state((a, b, c, d))) // 2
    assert check_orthogonality(build_goethals_seidel(a, b, c, d))[
        "energy"] == order * four_energy
    propus_energy = correlation_energy(
        autocorrelation_state((a, b, b, d))) // 2
    assert check_orthogonality(build_propus(a, b, d))[
        "energy"] == order * propus_energy


@settings(max_examples=30, deadline=None)
@given(four_sign_sequences())
def test_general_goethals_seidel_energy_matches_compact_energy(
    sequences: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    order = 4 * len(sequences[0])
    compact = periodic_autocorrelation_energy(sequences)
    matrix = build_goethals_seidel(*sequences)
    assert check_orthogonality(matrix)["energy"] == order * compact // 2


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
    sequences = np.stack([expand_symmetric_sequence(half) for half in halves])
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
        assert all(np.array_equal(sequence[1:], sequence[:0:-1])
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
        reference_correlations(sequences, tuple(int(value)
                               for value in weights)),
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
    sequences = np.stack([expand_symmetric_sequence(half) for _ in range(4)])
    correlations = autocorrelation_state(sequences)
    apply_symmetric_flip(sequences, correlations, 0, half_index)
    assert np.array_equal(correlations, reference_correlations(sequences))
    assert np.array_equal(sequences[0, 1:], sequences[0, :0:-1])


def test_correlation_state_falls_back_to_numpy(monkeypatch) -> None:
    sequences = tuple(np.array([1, -1, 1], dtype=np.int8) for _ in range(4))
    monkeypatch.setattr(correlations, "NUMBA_AVAILABLE", False)
    matrix = np.stack(sequences)
    state = correlations.autocorrelation_state(matrix)
    correlations.apply_sequence_flip(matrix, state, 2, 1)
    assert np.array_equal(state, reference_correlations(matrix))


def test_goethals_seidel_slicing_matches_permutation_matrix_reference() -> None:
    sequences = (
        np.array([1, -1, 1], dtype=np.int8),
        np.array([-1, -1, 1], dtype=np.int8),
        np.array([1, 1, -1], dtype=np.int8),
        np.array([-1, 1, 1], dtype=np.int8),
    )
    A, B, C, D = (
        np.array([np.roll(sequence, index)
                 for index in range(len(sequence))], dtype=np.int8)
        for sequence in sequences
    )
    reverse = np.fliplr(np.eye(3, dtype=np.int8))
    BR, CR, DR = B @ reverse, C @ reverse, D @ reverse
    BtR, CtR, DtR = B.T @ reverse, C.T @ reverse, D.T @ reverse
    expected = np.block([
        [A, BR, CR, DR],
        [-BR, A, -DtR, CtR],
        [-CR, DtR, A, -BtR],
        [-DR, -CtR, BtR, A],
    ]).astype(np.int8)
    assert np.array_equal(build_goethals_seidel(*sequences), expected)


def test_turyn_type_eight_builds_order_ninety_two_hadamard() -> None:
    x = np.array((1, 1, -1, 1, -1, 1, -1, 1), dtype=np.int8)
    y = np.array((1, -1, -1, -1, -1, -1, -1, 1), dtype=np.int8)
    z = np.array((1, -1, -1, 1, 1, 1, 1, -1), dtype=np.int8)
    w = np.array((1, 1, 1, -1, 1, 1, -1), dtype=np.int8)
    base = turyn_to_base(x, y, z, w)
    t_sequences = base_to_t_sequences(*base)
    signs = t_sequences_to_sign_sequences(*t_sequences)
    assert [len(sequence) for sequence in base] == [15, 15, 8, 8]
    assert np.all(np.sum(np.abs(np.stack(t_sequences)), axis=0) == 1)
    assert all(np.all(np.isin(sequence, (-1, 1))) for sequence in signs)
    matrix = build_turyn(x, y, z, w)
    assert matrix.shape == (92, 92)
    assert check_orthogonality(matrix)["energy"] == 0


def test_nonperiodic_flip_state_matches_recomputation() -> None:
    lengths = np.array((5, 5, 5, 4), dtype=np.int64)
    weights = np.array((1, 1, 2, 2), dtype=np.int64)
    sequences = np.array([
        [1, -1, 1, -1, 1], [-1, -1, 1, 1, -1],
        [1, 1, -1, -1, 1], [-1, 1, 1, -1, 0],
    ], dtype=np.int8)
    state = nonperiodic_autocorrelation_state(
        sequences, lengths=lengths, weights=weights)
    for sequence_index, value_index in ((0, 2), (3, 1), (2, 4)):
        energy = apply_nonperiodic_flip(
            sequences, state, sequence_index, value_index,
            lengths=lengths, weight=int(weights[sequence_index]))
        expected = nonperiodic_autocorrelation_state(
            sequences, lengths=lengths, weights=weights)
        assert np.array_equal(state, expected)
        assert energy == nonperiodic_correlation_energy(expected)


@settings(max_examples=50, deadline=None)
@given(sign_matrices())
def test_metrics_match_independent_reference(matrix: np.ndarray) -> None:
    assert check_orthogonality(matrix) == reference_metrics(matrix)


@settings(max_examples=30, deadline=None)
@given(sign_matrices())
def test_gram_primitives_match_independent_reference(matrix: np.ndarray) -> None:
    gram = gram_matrix(matrix)
    expected = matrix.astype(np.int64) @ matrix.astype(np.int64).T
    np.fill_diagonal(expected, 0)
    assert np.array_equal(to_numpy(gram), expected)
    assert metrics_from_gram(gram) == reference_metrics(matrix)


def test_correlation_histogram_counts_unordered_pairs() -> None:
    gram = np.array([
        [0, -2, 4],
        [-2, 0, 2],
        [4, 2, 0],
    ], dtype=np.int32)
    assert correlation_histogram(gram) == {"2": 2, "4": 1}


@settings(max_examples=20, deadline=None)
@given(
    sign_matrices(),
    st.lists(st.tuples(st.integers(0, 100), st.integers(0, 100)),
             min_size=0, max_size=10),
)
def test_incremental_entry_flips_match_full_recomputation(
    original: np.ndarray,
    raw_flips: list[tuple[int, int]],
) -> None:
    matrix = xp.asarray(original, dtype=xp.int8).copy()
    initial_gram = gram_matrix(matrix)
    gram = initial_gram.copy()
    energy = metrics_from_gram(gram)["energy"]
    flips = [(row % len(original), column % len(original))
             for row, column in raw_flips]

    for row, column in flips:
        predicted = entry_flip_delta(matrix, gram, row, column)
        assert apply_entry_flip(matrix, gram, row, column) == predicted
        energy += predicted
        expected_gram = gram_matrix(matrix)
        assert np.array_equal(to_numpy(gram), to_numpy(expected_gram))
        assert energy == metrics_from_gram(expected_gram)["energy"]

    for row, column in reversed(flips):
        apply_entry_flip(matrix, gram, row, column)
    assert np.array_equal(to_numpy(matrix), original)
    assert np.array_equal(to_numpy(gram), to_numpy(initial_gram))


@settings(max_examples=20, deadline=None)
@given(sign_matrices(), st.integers(0, 100))
def test_batched_entry_flip_deltas_match_full_recomputation(
    original: np.ndarray,
    raw_row: int,
) -> None:
    row = raw_row % len(original)
    matrix = xp.asarray(original, dtype=xp.int8)
    gram = gram_matrix(matrix)
    energy = reference_metrics(original)["energy"]
    deltas = entry_flip_deltas(matrix, gram, row, np.arange(len(original)))
    for column, delta in enumerate(deltas):
        flipped = original.copy()
        flipped[row, column] *= -1
        assert int(delta) == reference_metrics(flipped)["energy"] - energy


def test_repair_selects_columns_for_correlation_sign() -> None:
    matrix = np.asarray([
        [1, 1, 1, -1],
        [1, 1, -1, 1],
    ], dtype=np.int8)
    rng = np.random.default_rng(0)
    positive = RepairSearch._candidate_columns(matrix, 0, 1, 2, rng)
    negative = RepairSearch._candidate_columns(matrix, 0, 1, -2, rng)
    assert np.all(matrix[0, positive] == matrix[1, positive])
    assert np.all(matrix[0, negative] != matrix[1, negative])


def test_repair_evaluates_flips_in_both_violated_rows() -> None:
    matrix = np.asarray([
        [1, 1, 1, 1],
        [1, 1, -1, -1],
        [1, -1, 1, -1],
        [-1, 1, 1, -1],
    ], dtype=np.int8)
    gram = gram_matrix(matrix, backend=np)
    columns = np.array([0, 1], dtype=np.int64)
    moves = RepairSearch._candidate_moves(matrix, gram, (0, 1), columns)
    assert {row for _, row, _ in moves} == {0, 1}
    assert len(moves) == 4


def test_repair_violation_cache_tracks_exact_global_maximum() -> None:
    matrix = np.random.default_rng(9).choice(
        [-1, 1], size=(12, 12)).astype(np.int8)
    gram = gram_matrix(matrix, backend=np)
    row_argmax = np.argmax(np.abs(gram), axis=1)
    row_max = np.abs(gram[np.arange(len(matrix)), row_argmax])
    for row, column in np.random.default_rng(10).integers(0, 12, size=(20, 2)):
        apply_entry_flip(matrix, gram, int(row), int(column))
        RepairSearch._refresh_violations(
            gram, int(row), row_argmax, row_max)
        cached = RepairSearch._most_violated_pair(
            gram, row_argmax, row_max)
        assert abs(cached[2]) == int(np.abs(gram).max())


def test_refinement_preserves_input_when_no_steps_are_requested() -> None:
    candidate = sylvester(4)
    refined, metrics, _ = RepairSearch(
        order=4).refine(candidate, steps=0, seed=0)
    assert np.array_equal(candidate, sylvester(4))
    assert np.array_equal(refined, candidate)
    assert metrics == check_orthogonality(candidate)


def test_incremental_repair_search_returns_exact_metrics() -> None:
    matrix, metrics, _ = RepairSearch(12).search(steps=25, seed=4)
    assert metrics == reference_metrics(matrix)


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
        method_family="gs_sds",
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
    save_run(matrix, check_orthogonality(matrix), tmp_path, method_family="gs_sds",
             search_scope="test", seed=0, steps=1, wall_seconds=0.0, hardware_summary="test")
    manifest_path = tmp_path / "run.json"
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    data["method_family"] = "not-a-method"
    manifest_path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(InvalidManifest):
        load_and_validate_manifest(manifest_path)
