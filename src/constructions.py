"""Classical Hadamard constructions for solver calibration."""
from __future__ import annotations
import numpy as np


def sylvester(order: int) -> np.ndarray:
    if order < 1 or (order & (order - 1)) != 0:
        raise ValueError(f"order {order} is not a power of two")
    h = np.array([[1]], dtype=np.int8)
    for _ in range(order.bit_length() - 1):
        h = np.block([[h, h], [h, -h]])
    return h


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n % 2 == 0:
        return n == 2
    d = 3
    while d * d <= n:
        if n % d == 0:
            return False
        d += 2
    return True


def _legendre(x: int, p: int) -> int:
    if x % p == 0:
        return 0
    return 1 if pow(x, (p - 1) // 2, p) == 1 else -1


def paley(order: int) -> np.ndarray:
    q = order - 1
    if q < 3 or q % 4 != 3 or not _is_prime(q):
        raise ValueError(f"order {order} has no Paley construction")
    Q = np.zeros((q, q), dtype=np.int8)
    for i in range(q):
        for j in range(q):
            Q[i, j] = _legendre((j - i) % q, q)
    row = np.ones(order, dtype=np.int8)
    col = np.ones(order, dtype=np.int8)
    return np.block([[np.eye(1, dtype=np.int8), row[None, 1:]],
                     [col[1:, None], Q - np.eye(q, dtype=np.int8)]]).astype(np.int8)


def get_known(order: int) -> np.ndarray:
    if order <= 0:
        raise ValueError(f"invalid order {order}")
    if order == 1:
        return np.array([[1]], dtype=np.int8)
    if order == 2:
        return np.array([[1, 1], [1, -1]], dtype=np.int8)
    if (order & (order - 1)) == 0:
        return sylvester(order)
    q = order - 1
    if q >= 3 and q % 4 == 3 and _is_prime(q):
        return paley(order)
    raise ValueError(f"no known Hadamard of order {order}")
