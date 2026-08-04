"""Strategy lookup — map CLI name to SearchStrategy for a given order."""

from __future__ import annotations

from .base import Pipeline, SearchStrategy
from .greedy import TurynGreedySearch
from .pocs import PocsSearch
from .repair import RepairSearch
from .spectral_descent import TurynSpectralDescentSearch

_CLASSES: dict[str, type[SearchStrategy]] = {
    "greedy": TurynGreedySearch,
    "spectral_descent": TurynSpectralDescentSearch,
    "pocs": PocsSearch,
    "repair": RepairSearch,
}

GPU = frozenset({"pocs", "spectral_descent"})
DEFAULT = "spectral_descent"

_TURYN_CLASSES: set[type[SearchStrategy]] = {
    TurynGreedySearch,
    TurynSpectralDescentSearch,
    PocsSearch,
}


def _turyn_n(order: int) -> int:
    n = (order // 4 + 1) // 3
    if order != 4 * (3 * n - 1) or n < 2:
        raise ValueError(f"order {order} has no TT(n) construction")
    return n


def build(name: str, order: int) -> SearchStrategy:
    cls = _CLASSES[name]
    if cls in _TURYN_CLASSES:
        return cls(n=_turyn_n(order))  # type: ignore[call-arg]
    return cls(order=order)  # type: ignore[call-arg]


def parse(spec: str, order: int) -> SearchStrategy:
    """'name' or 'name:steps,name2:steps' → Strategy or Pipeline."""
    if ":" not in spec:
        return build(spec, order)
    stages = []
    for part in spec.split(","):
        n, st = part.rsplit(":", 1)
        stages.append((build(n, order), int(st)))
    return Pipeline(stages)
