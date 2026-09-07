"""Exact controls for laboratory lifts and projections."""

from __future__ import annotations

import numpy as np
import pytest

from lab.interleaving import alt_double, construct, deinterleave, main, verify_theorem
from src.constructions import double_gs4, paley_ng_sequences
from src.tracker import Tracker
from src.verify import verify_candidate


@pytest.mark.parametrize("n", [1, 2, 3, 4, 13, 30])
def test_lifts_roundtrip_and_alt_energy_scaling(n: int) -> None:
    rng = np.random.default_rng(42)
    for _ in range(10):
        original = rng.choice((-1, 1), size=(4, n)).astype(np.int8)
        lifted = alt_double(original)
        assert np.array_equal(deinterleave(lifted, "alt"), original)
        assert np.array_equal(deinterleave(double_gs4(original), "turyn"), original)
        base, doubled = Tracker(), Tracker()
        base.build(original)
        doubled.build(lifted)
        assert doubled.energy() == 8 * base.energy()
        assert not np.shares_memory(lifted, original)


@pytest.mark.parametrize("m", [1, 3, 5, 15])
def test_paley_fold_is_independently_verified(m: int) -> None:
    verify_candidate(deinterleave(paley_ng_sequences(2 * m)))


def test_construct_and_identity_controls() -> None:
    for order in (8, 16, 32, 64):
        sequences = construct(order)
        assert sequences is not None and sequences.shape == (4, order // 4)
        verify_candidate(sequences)
    assert construct(12) is None
    assert verify_theorem(12)
    with pytest.raises(ValueError, match="positive even"):
        verify_theorem(13)
    with pytest.raises(ValueError, match="shape"):
        deinterleave(np.ones((4, 3), dtype=np.int8))


def test_interleaving_cli_reports_errors_and_verified_controls() -> None:
    assert main(["roundtrip", "3", "4"]) == 0
    assert main(["check", "4"]) == 0
    assert main(["paley", "3"]) == 0
    assert main(["construct", "32"]) == 0
    assert main(["sweep", "--max-length", "3"]) == 0
    with pytest.raises(SystemExit) as error:
        main(["paley", "13"])
    assert error.value.code == 2
