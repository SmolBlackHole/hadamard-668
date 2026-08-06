"""Property tests for metrics and gram primitives."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from src.builder import Builder
from src.metrics import Metrics, check_orthogonality, gram_matrix, metrics_from_gram
from src.tracker import Tracker


def _reference_metrics(matrix: np.ndarray) -> Metrics:
    gram = matrix.astype(np.int64) @ matrix.astype(np.int64).T
    upper = gram[np.triu_indices(matrix.shape[0], k=1)]
    return Metrics(
        energy=int(np.dot(upper, upper)),
        orthogonal_pairs=int(np.count_nonzero(upper == 0)),
        max_abs_correlation=int(np.abs(upper).max(initial=0)),
    )


def _hadamard2():
    return np.array([[1, 1], [1, -1]], dtype=np.int8)


def _hadamard_sylvester4():
    h2 = _hadamard2()
    return np.block([[h2, h2], [h2, -h2]])


@st.composite
def _tracker_case(
    draw: st.DrawFn,
) -> tuple[npt.NDArray[np.int8], int, int]:
    n = draw(st.integers(min_value=3, max_value=11))
    seqs = draw(
        hnp.arrays(
            dtype=np.int8,
            shape=(4, n),
            elements=st.sampled_from((-1, 1)),
        )
    )
    return seqs, draw(st.integers(0, 3)), draw(st.integers(0, n - 1))


@st.composite
def _tracker_combo(
    draw: st.DrawFn,
) -> tuple[npt.NDArray[np.int8], list[tuple[int, int]]]:
    n = draw(st.integers(min_value=3, max_value=11))
    seqs = draw(
        hnp.arrays(
            dtype=np.int8,
            shape=(4, n),
            elements=st.sampled_from((-1, 1)),
        )
    )
    flat = draw(st.lists(st.integers(0, 4 * n - 1), max_size=5, unique=True))
    return seqs, [(index // n, index % n) for index in flat]


def _full_tracker_energy(seqs: npt.NDArray[np.int8]) -> int:
    return _reference_metrics(Builder(kind="gs4", n=seqs.shape[1]).build(seqs)).energy


@pytest.mark.parametrize(
    "matrix",
    [
        _hadamard2(),
        _hadamard_sylvester4(),
        np.ones((3, 3), dtype=np.int8),
    ],
)
def test_metrics_and_gram_primitives_match_independent_reference(matrix):
    expected = _reference_metrics(matrix)
    gram = gram_matrix(matrix)
    assert check_orthogonality(matrix) == expected
    assert metrics_from_gram(gram) == expected


@given(case=_tracker_case())
@settings(max_examples=50, deadline=None)
def test_tracker_single_flip_and_accept_match_full_gram(
    case: tuple[npt.NDArray[np.int8], int, int],
) -> None:
    seqs, s, c = case
    tracker = Tracker()
    tracker.build(seqs)

    flipped = seqs.copy()
    flipped[s, c] *= -1
    expected = _full_tracker_energy(flipped)

    assert tracker.flip(s, c) == expected
    assert tracker.accept(seqs, s, c) == expected
    assert tracker.energy() == expected
    assert np.array_equal(seqs, flipped)

    fresh = Tracker()
    fresh.build(flipped)
    assert np.array_equal(tracker.flip_energies(), fresh.flip_energies())


@given(case=_tracker_combo())
@settings(max_examples=50, deadline=None)
def test_tracker_combo_matches_full_gram(
    case: tuple[npt.NDArray[np.int8], list[tuple[int, int]]],
) -> None:
    seqs, combo = case
    tracker = Tracker()
    tracker.build(seqs)

    flipped = seqs.copy()
    if combo:
        positions = np.asarray(combo, dtype=np.intp)
        flipped[positions[:, 0], positions[:, 1]] *= -1

    assert tracker.combo_energy(combo) == _full_tracker_energy(flipped)
