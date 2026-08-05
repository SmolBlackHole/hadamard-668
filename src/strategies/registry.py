"""Strategy lookup — map CLI name to SearchStrategy for a given order."""

from __future__ import annotations

from .base import Pipeline, SearchStrategy
from .custom import CustomSolver
from .kflip import KFlipRepair
from .pocs import PocsSearch
from .spectral_descent import TurynSpectralDescentSearch

try:
    from experiments.genetic import BatchedGeneticSearch as GeneticSearch
except ImportError:
    GeneticSearch = None  # type: ignore[assignment]

_CLASSES: dict[str, type[SearchStrategy] | None] = {
    "custom": CustomSolver,
    "kflip": KFlipRepair,
    "pocs": PocsSearch,
    "spectral_descent": TurynSpectralDescentSearch,
    "genetic": GeneticSearch,
}

GPU = frozenset({"pocs", "spectral_descent"})
DEFAULT = "custom"

_TURYN_CLASSES: set[type[SearchStrategy] | None] = {
    TurynSpectralDescentSearch,
    PocsSearch,
    KFlipRepair,
    GeneticSearch,
}


def _turyn_n(order: int) -> int:
    n = (order // 4 + 1) // 3
    if order != 4 * (3 * n - 1) or n < 2:
        raise ValueError(f"order {order} has no TT(n) construction")
    return n


def build(name: str, order: int) -> SearchStrategy:
    cls = _CLASSES[name]
    if cls is None:
        raise ImportError(f"strategy '{name}' requires packages not installed")
    if name == "custom":
        return CustomSolver.group(n=order // 4)  # type: ignore[return-value]
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
