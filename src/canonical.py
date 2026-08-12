"""Stable identities for raw and quick-orbit GS4 states."""

from __future__ import annotations

import base64
import hashlib
from collections import Counter
from dataclasses import dataclass
from math import factorial

import numpy as np

from .models import Int8Array

CANONICALIZER = "gs4-quick-orbit-v1"


@dataclass(frozen=True)
class StateIdentity:
    validation_hash: str
    orbit_hash: str
    canonical_b64: str
    canonicalizer: str = CANONICALIZER


def _validate(sequences: Int8Array) -> None:
    if sequences.ndim != 2 or sequences.shape[0] != 4 or sequences.shape[1] < 1:
        raise ValueError("GS4 state must have shape (4, n) with positive n")
    if not np.all(np.abs(sequences) == 1):
        raise ValueError("GS4 state must contain only -1 and 1")


def validation_hash(sequences: Int8Array) -> str:
    _validate(sequences)
    return hashlib.sha256(sequences.astype(np.int8, copy=False).tobytes()).hexdigest()


def _to_int(sequence: Int8Array) -> int:
    value = 0
    for item in reversed(sequence):
        value = (value << 1) | (1 if item == 1 else 0)
    return value


def _canonical_int(value: int, n: int) -> int:
    mask = (1 << n) - 1
    best = value
    current = value
    for _ in range(2 * n):
        current = ((current << 1) & mask) | (1 ^ ((current >> (n - 1)) & 1))
        best = min(best, current)

    reversed_value = 0
    current = value
    for _ in range(n):
        reversed_value = (reversed_value << 1) | (current & 1)
        current >>= 1
    current = reversed_value
    for _ in range(2 * n):
        current = ((current << 1) & mask) | (1 ^ ((current >> (n - 1)) & 1))
        best = min(best, current)
    return best


def _sequence_orbit(value: int, n: int) -> set[int]:
    mask = (1 << n) - 1
    variants: set[int] = set()
    for initial in (value, int(f"{value:0{n}b}"[::-1], 2)):
        current = initial
        for _ in range(2 * n):
            variants.add(current)
            current = ((current << 1) & mask) | (1 ^ ((current >> (n - 1)) & 1))
    return variants


def canonical_sequences(sequences: Int8Array) -> Int8Array:
    _validate(sequences)
    n = sequences.shape[1]
    values = sorted(_canonical_int(_to_int(row), n) for row in sequences)
    result = np.empty_like(sequences)
    for row, value in enumerate(values):
        result[row] = np.fromiter(
            (1 if value & (1 << column) else -1 for column in range(n)),
            dtype=np.int8,
            count=n,
        )
    return result


def orbit_hash(sequences: Int8Array) -> str:
    _validate(sequences)
    n = sequences.shape[1]
    width = (n + 7) // 8
    values = sorted(_canonical_int(_to_int(row), n) for row in sequences)
    payload = b"".join(value.to_bytes(width, "big") for value in values)
    return hashlib.sha256(payload).hexdigest()


def quick_orbit_size(sequences: Int8Array) -> int:
    _validate(sequences)
    n = sequences.shape[1]
    values = [_to_int(row) for row in sequences]
    canonical_values = [_canonical_int(value, n) for value in values]
    row_permutations = factorial(4)
    for count in Counter(canonical_values).values():
        row_permutations //= factorial(count)
    return row_permutations * int(np.prod([len(_sequence_orbit(value, n)) for value in values]))


def identify_state(sequences: Int8Array) -> StateIdentity:
    canonical = canonical_sequences(sequences)
    raw = sequences.astype(np.int8, copy=False).tobytes()
    canonical_raw = canonical.tobytes()
    return StateIdentity(
        validation_hash=hashlib.sha256(raw).hexdigest(),
        orbit_hash=orbit_hash(sequences),
        canonical_b64=base64.b64encode(canonical_raw).decode("ascii"),
    )
