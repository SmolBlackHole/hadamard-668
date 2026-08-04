"""Contracts for generic Turyn PSD and row-sum sieving."""
from __future__ import annotations

import numpy as np
import pytest

from sieve import seed_turyn_batch, turyn_psd_mask, turyn_sum_patterns


@pytest.mark.parametrize("n", (2, 4, 5, 6, 8, 9, 36, 56))
def test_sieved_turyn_seeds_have_valid_padding_sums_and_psd(n: int) -> None:
    batch = seed_turyn_batch(n, 3, np.random.default_rng(n), chunk_size=4_096)
    lengths = np.array((n, n, n, n - 1))
    sums = np.array([candidate[index, :length].sum()
                     for candidate in batch for index, length in enumerate(lengths)]).reshape(len(batch), 4)
    patterns = {tuple(pattern) for pattern in turyn_sum_patterns(n)}
    assert batch.shape == (3, 4, n)
    assert np.all(batch[:, 3, -1] == 0)
    assert all(tuple(row) in patterns for row in sums)
    assert np.all(turyn_psd_mask(batch, n=n))


@pytest.mark.parametrize("n", (3, 7))
def test_impossible_sum_patterns_stop_the_sieve(n: int) -> None:
    with pytest.raises(ValueError, match="no compatible row-sum pattern"):
        turyn_sum_patterns(n)


def test_psd_sieve_keeps_a_known_tt_two_solution() -> None:
    candidate = np.array(
        [[[-1, -1], [-1, -1], [-1, 1], [-1, 0]]], dtype=np.int8)
    assert np.array_equal(turyn_psd_mask(candidate, n=2), np.array([True]))
