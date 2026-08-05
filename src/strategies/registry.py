"""Strategy lookup — map CLI name to SearchStrategy for a given order."""

from __future__ import annotations

from strategies.base import Pipeline, SearchStrategy

from .custom import CustomSolver

_CLASSES: dict[str, type[SearchStrategy]] = {
    "custom": CustomSolver,
}

DEFAULT = "custom"


def build(name: str, order: int) -> SearchStrategy:
    return _CLASSES[name].from_order(order)


def parse(spec: str, order: int) -> SearchStrategy:
    if ":" not in spec:
        return build(spec, order)
    stages = []
    for part in spec.split(","):
        n, st = part.rsplit(":", 1)
        stages.append((build(n, order), int(st)))
    return Pipeline(stages)
