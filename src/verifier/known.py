"""Generators for known Hadamard matrices (Sylvester, Paley).

Used as ground truth to verify the search engine and orthogonality checks.
"""
from __future__ import annotations

import numpy as np


def sylvester(order: int) -> np.ndarray:
    """Sylvester construction: H_{2^k} = H_2 ⊗ H_2 ⊗ ... ⊗ H_2.

    Produces a Hadamard matrix of order 2^k for any k >= 0.
    """
    if order == 1:
        return np.array([[1]], dtype=np.int8)
    if order & (order - 1) != 0:
        raise ValueError(f"Sylvester requires a power of 2, got {order}")

    H = np.array([[1, 1], [1, -1]], dtype=np.int8)
    n = 2
    while n < order:
        H = np.kron(H, np.array([[1, 1], [1, -1]], dtype=np.int8))
        n *= 2
    return H


def paley(order: int) -> np.ndarray:
    """Paley type I construction for order q+1 where q ≡ 3 (mod 4) is prime.

    Returns H_{q+1} = [
      [ 1   1 ... 1       1   ],
      [ 1                     ],
      [ .    χ(i-j) matrix    ],
      [ 1                     ],
    ] where χ is the Legendre symbol: 1 if QR, -1 if QNR, 0 on diagonal.
    """
    q = order - 1
    if q % 4 != 3 or not _is_prime(q):
        raise ValueError(
            f"Paley type I requires q = order-1 to be prime ≡ 3 mod 4, got {q}")

    H = np.ones((order, order), dtype=np.int8)
    for i in range(1, order):
        for j in range(1, order):
            diff = (i - j) % q
            if i == j:
                H[i, j] = -1  # chi(0) - 1 = 0 - 1 = -1
            else:
                chi = _legendre(diff, q)
                H[i, j] = chi  # for i != j, S[i][j] = chi(i-j), S-I = chi
    return H


def _legendre(a: int, p: int) -> int:
    """Legendre symbol (a|p): 1 if QR, -1 if QNR, 0 if divisible by p."""
    a_mod = a % p
    if a_mod == 0:
        return 0
    s = pow(a_mod, (p - 1) // 2, p)
    return 1 if s == 1 else -1


def _is_prime(n: int) -> bool:
    if n < 2:
        return False
    if n < 4:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True


def get_known(order: int) -> np.ndarray:
    """Return a known Hadamard matrix of the given order, or raise ValueError."""
    if order == 1:
        return sylvester(1)
    if order == 2:
        return sylvester(2)
    if (order & (order - 1)) == 0:
        return sylvester(order)
    if _is_prime(order - 1) and (order - 1) % 4 == 3:
        return paley(order)
    raise ValueError(f"No generator implemented for order {order}")
