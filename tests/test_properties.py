"""Property tests for exact tracker energy updates."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as hnp

from src.builder import build_gs4
from src.tracker import Tracker


def _matrix_energy(matrix: np.ndarray) -> int:
    gram = matrix.astype(np.int64) @ matrix.astype(np.int64).T
    upper = gram[np.triu_indices(matrix.shape[0], k=1)]
    return int(np.dot(upper, upper))


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
    return _matrix_energy(build_gs4(seqs))


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
