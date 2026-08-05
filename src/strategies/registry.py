"""Strategy lookup — map CLI name to SearchStrategy for a given order."""

from __future__ import annotations

from strategies.base import Pipeline, SearchStrategy

from .custom import CustomSolver
from .pocs import PocsSearch
from .spectral_descent import TurynSpectralDescentSearch

try:
    from experiments.genetic import BatchedGeneticSearch as GeneticSearch
except ImportError:
    GeneticSearch = None  # type: ignore[assignment]

_CLASSES: dict[str, type[SearchStrategy] | None] = {
    "custom": CustomSolver,
    "pocs": PocsSearch,
    "spectral_descent": TurynSpectralDescentSearch,
    "genetic": GeneticSearch,
}

DEFAULT = "custom"


def build(name: str, order: int) -> SearchStrategy:
    cls = _CLASSES[name]
    if cls is None:
        raise ImportError(f"strategy '{name}' requires packages not installed")
    return cls.from_order(order)  # type: ignore[attr-defined]


def parse(spec: str, order: int) -> SearchStrategy:
    """'name' or 'name:steps,name2:steps' → Strategy or Pipeline."""
    if ":" not in spec:
        return build(spec, order)
    stages = []
    for part in spec.split(","):
        n, st = part.rsplit(":", 1)
        stages.append((build(n, order), int(st)))
    return Pipeline(stages)
