"""Sequence generation and exact construction helpers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .constructions import paley_ng_sequences, supports_paley_ng
from .models import Int8Array

STRATEGIES = ("gs4", "paley-ng", "construct")


class StartConstruction(Protocol):
    @property
    def name(self) -> str: ...

    def build(self, n: int, rng: np.random.Generator) -> Int8Array: ...


@dataclass(frozen=True)
class _StartConstruction:
    name: str
    builder: Callable[[int, np.random.Generator], Int8Array]

    def build(self, n: int, rng: np.random.Generator) -> Int8Array:
        if n <= 0:
            raise ValueError("n must be positive")
        return self.builder(n, rng)


def n_from_order(strategy: str, order: int) -> int:
    if strategy not in STRATEGIES:
        raise ValueError(f"strategy must be one of {STRATEGIES}, got {strategy!r}")
    if order <= 0 or order % 4:
        raise ValueError(f"{strategy} requires a positive order divisible by 4")
    return order // 4


def _random_start(n: int, rng: np.random.Generator) -> Int8Array:
    return rng.choice(np.array([-1, 1], dtype=np.int8), size=(4, n))


def _cyclic_start(n: int, rng: np.random.Generator) -> Int8Array:
    base = rng.choice(np.array([-1, 1], dtype=np.int8), size=n)
    sequences = np.empty((4, n), dtype=np.int8)
    sequences[0] = base
    for index, shift in enumerate(rng.integers(1, n, size=3), 1):
        sequences[index] = np.roll(base, int(shift))
    return sequences


RANDOM_START: StartConstruction = _StartConstruction("random", _random_start)
CYCLIC_START: StartConstruction = _StartConstruction("cyclic", _cyclic_start)
START_CONSTRUCTIONS: dict[str, StartConstruction] = {
    construction.name: construction for construction in (RANDOM_START, CYCLIC_START)
}
START_KINDS = tuple(START_CONSTRUCTIONS)


def start_construction(name: str) -> StartConstruction:
    try:
        return START_CONSTRUCTIONS[name]
    except KeyError:
        raise ValueError(f"start kind must be one of {START_KINDS}, got {name!r}") from None


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
