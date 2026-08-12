"""Sequence generation and exact construction helpers."""

from __future__ import annotations

import numpy as np

from .constructions import paley_ng_sequences, supports_paley_ng
from .models import Int8Array

STRATEGIES = ("gs4", "paley-ng", "construct")
START_KINDS = ("random", "cyclic")


def n_from_order(strategy: str, order: int) -> int:
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}, got {strategy!r}")
    if order <= 0 or order % 4:
        raise ValueError(f"{strategy} requires a positive order divisible by 4")
    return order // 4


def initial_sequences(n: int, rng: np.random.Generator, start_kind: str) -> Int8Array:
    if n <= 0:
        raise ValueError("n must be positive")
    if start_kind not in START_KINDS:
        raise ValueError(f"start_kind must be one of {START_KINDS}, got {start_kind!r}")
    if start_kind == "random":
        return rng.choice(np.array([-1, 1], dtype=np.int8), size=(4, n))

    base = rng.choice(np.array([-1, 1], dtype=np.int8), size=n)
    sequences = np.empty((4, n), dtype=np.int8)
    sequences[0] = base
    for index, shift in enumerate(rng.integers(1, n, size=3), 1):
        sequences[index] = np.roll(base, int(shift))
    return sequences


def exact_sequences(strategy: str, n: int) -> Int8Array | None:
    if strategy == "gs4":
        return None
    if strategy == "paley-ng":
        if n % 2:
            raise ValueError("paley-ng requires even n")
        if not supports_paley_ng(n):
            raise ValueError(f"paley-ng requires p=2n-1 to be prime, got {2 * n - 1}")
        return paley_ng_sequences(n)
    if strategy == "construct":
        return paley_ng_sequences(n) if supports_paley_ng(n) else None
    raise ValueError(f"strategy must be one of {STRATEGIES}, got {strategy!r}")
