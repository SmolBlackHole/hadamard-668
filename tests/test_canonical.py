"""Tests for stable GS4 state identities."""

from __future__ import annotations

import base64

import numpy as np
import pytest

from src.canonical import CANONICALIZER, canonical_sequences, identify_state, quick_orbit_size


def _negashift(sequence: np.ndarray) -> np.ndarray:
    return np.concatenate((-sequence[-1:], sequence[:-1])).astype(np.int8)


def test_identity_is_invariant_under_quick_orbit_operations() -> None:
    sequences = np.random.default_rng(42).choice((-1, 1), size=(4, 11)).astype(np.int8)
    transformed = sequences[[2, 0, 3, 1]].copy()
    transformed[0] = _negashift(_negashift(transformed[0]))
    transformed[1] = transformed[1, ::-1]
    transformed[2] *= -1

    original = identify_state(sequences)
    other = identify_state(transformed)

    assert original.validation_hash != other.validation_hash
    assert original.orbit_hash == other.orbit_hash
    assert original.canonical_b64 == other.canonical_b64
    assert original.canonicalizer == CANONICALIZER


def test_canonical_payload_roundtrips() -> None:
    sequences = np.random.default_rng(7).choice((-1, 1), size=(4, 9)).astype(np.int8)
    identity = identify_state(sequences)
    decoded = np.frombuffer(base64.b64decode(identity.canonical_b64), dtype=np.int8).reshape(4, 9)

    assert np.array_equal(decoded, canonical_sequences(sequences))


def test_identity_rejects_invalid_states() -> None:
    with pytest.raises(ValueError, match="shape"):
        identify_state(np.ones((3, 4), dtype=np.int8))
    with pytest.raises(ValueError, match="only"):
        identify_state(np.zeros((4, 4), dtype=np.int8))


def test_quick_orbit_size_accounts_for_identical_rows() -> None:
    row = np.array([1, 1, 1, 1, -1], dtype=np.int8)
    repeated = np.broadcast_to(row, (4, 5)).copy()
    distinct = repeated.copy()
    distinct[1] = np.array([-1, 1, -1, -1, -1], dtype=np.int8)

    assert quick_orbit_size(repeated) < quick_orbit_size(distinct)
