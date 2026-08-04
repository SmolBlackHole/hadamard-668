"""Strategy registry and factory for CLI dispatch."""
from __future__ import annotations

from .base import Pipeline, SearchStrategy
from .greedy import TurynGreedySearch
from .steepest import TurynSteepestSearch
from .spectral_descent import TurynSpectralDescentSearch
from .pocs import PocsSearch
from .montecarlo import MonteCarloSearch
from .repair import RepairSearch
from .annealing import TurynAnnealingSearch
from .ising import IsingSearch
from .genetic import GeneticSearch


_STRATEGY_CLASSES: dict[str, type[SearchStrategy]] = {
    "greedy": TurynGreedySearch,
    "steepest": TurynSteepestSearch,
    "spectral_descent": TurynSpectralDescentSearch,
    "pocs": PocsSearch,
    "montecarlo": MonteCarloSearch,
    "repair": RepairSearch,
    "annealing": TurynAnnealingSearch,
    "ising": IsingSearch,
    "genetic": GeneticSearch,
}

_TURYN_N_CLASSES = frozenset({
    "greedy", "steepest", "spectral_descent", "annealing",
})
_ORDER_KWARG_KEY: dict[str, str] = {
    "pocs": "ORDER",
    "montecarlo": "order",
    "repair": "order",
    "ising": "order",
    "genetic": "order",
}

GPU_NAMES = frozenset({
    "montecarlo", "spectral_descent", "pocs", "ising", "genetic",
})

DEFAULT_STRATEGY = "spectral_descent"


def list_names() -> list[str]:
    return sorted(_STRATEGY_CLASSES)


def _turyn_n(order: int) -> int:
    n = (order // 4 + 1) // 3
    if order != 4 * (3 * n - 1) or n < 2:
        raise ValueError(f"order {order} has no TT(n) construction")
    return n


def make_strategy(name: str, order: int) -> SearchStrategy:
    cls = _STRATEGY_CLASSES[name]
    if name in _TURYN_N_CLASSES:
        return cls(n=_turyn_n(order))
    kwarg = _ORDER_KWARG_KEY.get(name, "order")
    return cls(**{kwarg: order})


def parse_strategy(specification: str, order: int) -> SearchStrategy:
    if ":" not in specification:
        if specification not in _STRATEGY_CLASSES:
            raise ValueError(f"unknown strategy: {specification}")
        return make_strategy(specification, order)

    stages = []
    for part in specification.split(","):
        name, steps_text = part.rsplit(":", 1)
        if name not in _STRATEGY_CLASSES:
            raise ValueError(f"unknown strategy in pipeline: {name}")
        stages.append((make_strategy(name, order), int(steps_text)))
    return Pipeline(stages)
