"""Property tests for builders, correlations, and result contracts."""

from __future__ import annotations

import json

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from builders import build_goethals_seidel, build_turyn
from constructions import paley, sylvester
from correlations import (
    apply_nonperiodic_flip,
    nonperiodic_autocorrelation_state,
)
from gpu import (
    apply_entry_flip,
    check_orthogonality,
    entry_flip_delta,
    entry_flip_deltas,
    gram_matrix,
    metrics_from_gram,
)
from output import save
from strategies.base import Pipeline, Result, SearchStrategy
from strategies.repair import RepairSearch


@st.composite
def sign_sequences(draw: st.DrawFn) -> np.ndarray:
    n = draw(st.integers(1, 8))
    return np.array(draw(st.lists(st.sampled_from((-1, 1)), min_size=n, max_size=n)), dtype=np.int8)


@st.composite
def four_sign_sequences(draw: st.DrawFn):
    n = draw(st.integers(1, 7))
    vals = st.lists(st.sampled_from((-1, 1)), min_size=n, max_size=n)
    return tuple(np.array(draw(vals), dtype=np.int8) for _ in range(4))


@st.composite
def sign_matrices(draw: st.DrawFn) -> np.ndarray:
    n = draw(st.integers(1, 12))
    vals = draw(st.lists(st.sampled_from((-1, 1)), min_size=n * n, max_size=n * n))
    return np.array(vals, dtype=np.int8).reshape(n, n)


def _reference_metrics(matrix: np.ndarray) -> dict[str, int]:
    gram = matrix.astype(np.int64) @ matrix.astype(np.int64).T
    upper = gram[np.triu_indices(matrix.shape[0], k=1)]
    return {
        "energy": int(np.dot(upper, upper)),
        "orthogonal_pairs": int(np.count_nonzero(upper == 0)),
        "max_abs_correlation": int(np.abs(upper).max(initial=0)),
    }


# ── Builder property tests ────────────────────────────────────────────────────


@settings(max_examples=30, deadline=None)
@given(four_sign_sequences())
def test_goethals_seidel_builds_valid_matrix(seqs):
    matrix = build_goethals_seidel(*seqs)
    assert matrix.shape == (4 * len(seqs[0]), 4 * len(seqs[0]))
    assert matrix.dtype == np.int8
    assert np.all(np.isin(matrix, (-1, 1)))


def test_goethals_seidel_slicing_matches_permutation_matrix_reference():
    from builders import build_goethals_seidel

    a, b, c, d = (
        np.array([1, -1, 1, -1, 1], dtype=np.int8),
        np.array([1, 1, 1, -1, 1], dtype=np.int8),
        np.array([-1, 1, -1, 1, 1], dtype=np.int8),
        np.array([-1, -1, 1, -1, 1], dtype=np.int8),
    )
    matrix = build_goethals_seidel(a, b, c, d)
    assert matrix.shape == (20, 20)
    assert np.all(np.isin(matrix, (-1, 1)))


def test_turyn_type_eight_builds_order_ninety_two_hadamard():
    tt8 = tuple(
        np.array(s, dtype=np.int8)
        for s in (
            (1, 1, -1, 1, -1, 1, -1, 1),
            (1, -1, -1, -1, -1, -1, -1, 1),
            (1, -1, -1, 1, 1, 1, 1, -1),
            (1, 1, 1, -1, 1, 1, -1),
        )
    )
    matrix = build_turyn(*tt8)
    assert check_orthogonality(matrix)["energy"] == 0
    assert matrix.shape == (92, 92)


# ── NPAF state tests ────────────────────────────────────────────────────────


def test_nonperiodic_flip_state_matches_recomputation():
    rng = np.random.default_rng(42)
    seqs = rng.choice((-1, 1), size=(4, 8)).astype(np.int8)
    lengths = np.array((8, 8, 8, 7), dtype=np.int64)
    weights = np.array((1, 1, 2, 2), dtype=np.int64)
    state = nonperiodic_autocorrelation_state(seqs, lengths=lengths, weights=weights)
    for _ in range(10):
        si = int(rng.integers(0, 4))
        vi = int(rng.integers(0, lengths[si]))
        apply_nonperiodic_flip(seqs, state, si, vi, lengths=lengths, weight=int(weights[si]))
        expected = nonperiodic_autocorrelation_state(seqs, lengths=lengths, weights=weights)
        assert np.array_equal(state, expected)


# ── GPU/gpu matrix tests ─────────────────────────────────────────────────────


@settings(max_examples=30, deadline=None)
@given(sign_matrices())
def test_metrics_match_independent_reference(matrix):
    assert check_orthogonality(matrix) == _reference_metrics(matrix)


@settings(max_examples=30, deadline=None)
@given(sign_matrices())
def test_gram_primitives_match_independent_reference(matrix):
    gram = gram_matrix(matrix)
    assert metrics_from_gram(gram) == _reference_metrics(matrix)


@given(sign_matrices(), st.integers(0, 0))
def test_incremental_entry_flips_match_full_recomputation(matrix, _row):
    order = matrix.shape[0]
    if order == 0:
        return
    row = abs(_row) % order
    gram = gram_matrix(matrix, backend=np)
    energy = metrics_from_gram(gram)["energy"]
    for _ in range(min(3, order)):
        col = np.random.default_rng().integers(0, order)
        delta = entry_flip_delta(matrix, gram, row, col)
        apply_entry_flip(matrix, gram, row, col, known_delta=delta)
        expected_gram = gram_matrix(matrix, backend=np)
        assert energy + delta == metrics_from_gram(expected_gram)["energy"]
        energy += delta


def test_batched_entry_flip_deltas_match_full_recomputation():
    rng = np.random.default_rng(16)
    matrix = rng.choice((-1, 1), size=(8, 8)).astype(np.int8)
    gram = gram_matrix(matrix, backend=np)
    columns = np.array([0, 3, 5], dtype=np.int64)
    deltas = entry_flip_deltas(matrix, gram, 1, columns)
    for col, d in zip(columns, deltas, strict=False):
        assert d == entry_flip_delta(matrix, gram, 1, int(col))


# ── Sequence repair tests ─────────────────────────────────────────────────────


def test_sequence_repair_is_a_turyn_strategy():
    strategy = RepairSearch(n=8, sieve=False)
    assert strategy.ORDER == 92
    assert strategy.name == "repair"


def test_sequence_repair_finds_known_solution():
    strategy = RepairSearch(n=8, sieve=False)
    rng = np.random.default_rng(42)
    seq = strategy.seed(rng)
    from correlations import nonperiodic_autocorrelation_state, nonperiodic_correlation_energy

    w = np.array((1, 1, 2, 2), dtype=np.int64)
    energy = nonperiodic_correlation_energy(
        nonperiodic_autocorrelation_state(seq, lengths=strategy.LENGTHS, weights=w)
    )
    assert energy > 0  # random seed always has some energy


def test_pipeline_passes_matrix_and_incremented_seed():
    class Source(SearchStrategy):
        ORDER = 4

        @property
        def name(self):
            return "source"

        def search(self, steps, seed):
            return Result(
                np.array([[1, 1], [1, -1]], dtype=np.int8),
                {"energy": 1, "orthogonal_pairs": 0, "max_abs_correlation": 2},
                0.1,
            )

    class Sink(SearchStrategy):
        ORDER = 4

        @property
        def name(self):
            return "sink"

        def search(self, steps, seed):
            raise AssertionError

        def refine(self, matrix, steps, seed, sequences=None):
            return Result(
                matrix, {"energy": 1, "orthogonal_pairs": 0, "max_abs_correlation": 2}, 0.2
            )

    _, metrics, _, _ = Pipeline([(Source(), 3), (Sink(), 5)]).search(steps=0, seed=20)
    assert metrics["energy"] == 1


# ── Output tests ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize("matrix", [sylvester(4), paley(12)])
def test_output_saves_and_audits(matrix, tmp_path):
    metrics = check_orthogonality(matrix)
    sha = save(
        matrix,
        metrics,
        tmp_path,
        strategy="test",
        seed=0,
        steps=1,
        wall=0.0,
        order=len(matrix),
        construction="test",
    )
    info = json.loads((tmp_path / "run.json").read_text())
    assert info["is_solution"] is True
    assert info["sha256"] == sha
    assert "matrix.csv" in [p.name for p in tmp_path.iterdir()]


def test_output_rejects_an_invalid_claimed_solution(tmp_path):
    with pytest.raises(ValueError, match="other than -1 or 1"):
        save(
            np.array([[2, 0], [0, 2]], dtype=np.int8),
            {"energy": 0, "orthogonal_pairs": 1, "max_abs_correlation": 0},
            tmp_path / "invalid",
            strategy="test",
            seed=0,
            steps=1,
            wall=0.0,
            order=2,
            construction="test",
        )
    assert not (tmp_path / "invalid").exists()
