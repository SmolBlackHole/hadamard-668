"""Strategy lookup — map CLI name to SearchStrategy for a given order."""
from __future__ import annotations

from .base import Pipeline, SearchStrategy, TurynStrategy
from .greedy import TurynGreedySearch
from .spectral_descent import TurynSpectralDescentSearch
from .pocs import PocsSearch
from .repair import RepairSearch

_CLASSES: dict[str, type[SearchStrategy]] = {
    "greedy": TurynGreedySearch,
    "spectral_descent": TurynSpectralDescentSearch,
    "pocs": PocsSearch,
    "repair": RepairSearch,
}

GPU = frozenset({"pocs", "spectral_descent"})
DEFAULT = "spectral_descent"


def _turyn_n(order: int) -> int:
    n = (order // 4 + 1) // 3
    if order != 4 * (3 * n - 1) or n < 2:
        raise ValueError(f"order {order} has no TT(n) construction")
    return n


def build(name: str, order: int) -> SearchStrategy:
    cls = _CLASSES[name]
    if issubclass(cls, TurynStrategy):
        return cls(n=_turyn_n(order))
    return cls(order=order)


def parse(spec: str, order: int) -> SearchStrategy:
    """'name' or 'name:steps,name2:steps' → Strategy or Pipeline."""
    if ":" not in spec:
        return build(spec, order)
    stages = []
    for part in spec.split(","):
        n, st = part.rsplit(":", 1)
        stages.append((build(n, order), int(st)))
    return Pipeline(stages)

