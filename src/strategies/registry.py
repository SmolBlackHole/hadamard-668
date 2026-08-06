"""Strategy lookup — map CLI name to SearchStrategy for a given order."""

from __future__ import annotations

from strategies.base import SearchStrategy

from .custom import CustomSolver

_STRATEGIES: dict[str, tuple[str, int]] = {
    "gs4": ("gs4", 4),
    "golay_2n": ("golay_2n", 2),
    "gs4_group": ("gs4_group", 4),
}

DEFAULT = "gs4"


def _best_factorization(n: int) -> list[int]:
    pairs = [(p, n // p) for p in range(3, int(n**0.5) + 1) if n % p == 0 and n // p >= 3]
    if not pairs:
        return [n]
    p, q = min(pairs, key=lambda pq: abs(pq[0] - pq[1]))
    return sorted([p, q], reverse=True)


def build(name: str, order: int) -> SearchStrategy:
    if name == "tensor":
        n = order // 16
        dims = _best_factorization(n)
        return CustomSolver.tensor(n1=dims[0], n2=dims[1])
    if name not in _STRATEGIES:
        raise ValueError(f"unknown strategy {name!r}, choose from {sorted(_STRATEGIES.keys())}")
    kind, divisor = _STRATEGIES[name]
    return CustomSolver(kind=kind, n=order // divisor)
