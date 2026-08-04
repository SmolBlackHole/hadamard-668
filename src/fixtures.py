"""Known Hadamard solutions for testing and experiments.

Contains reference sequences and matrix generators for:
- Turyn TT(8) → H(92)
- Turyn TT(36) → H(428)
- Sylvester construction H(2^k)
- Paley construction H(q+1) for prime q ≡ 3 (mod 4)
"""
from __future__ import annotations

import numpy as np
from typing import Sequence

# ── Turyn TT(8) → Hadamard 92 ─────────────────────────────────────────────────

TT8_SEQUENCES: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]] = (
    (1, 1, -1, 1, -1, 1, -1, 1),
    (1, -1, -1, -1, -1, -1, -1, 1),
    (1, -1, -1, 1, 1, 1, 1, -1),
    (1, 1, 1, -1, 1, 1, -1),
)


def tt8_sequences() -> np.ndarray:
    """Return TT(8) sequences as a padded (4,8) int8 array."""
    arr = np.zeros((4, 8), dtype=np.int8)
    for i, s in enumerate(TT8_SEQUENCES):
        arr[i, : len(s)] = np.array(s, dtype=np.int8)
    return arr


TT8_LENGTHS: np.ndarray = np.array((8, 8, 8, 7), dtype=np.int64)


# ── Turyn TT(36) → Hadamard 428 ───────────────────────────────────────────────

TT36_HEX = "060989975b685d8fc80750b21c0212eceb26"


def _parse_hex_tt(hex_str: str) -> np.ndarray:
    """Parse a hex-encoded TT solution into a padded (4, n) int8 array."""
    seqs = [[], [], [], []]
    for digit in hex_str[:-1]:
        for idx, bit in enumerate(f"{int(digit, 16):04b}"):
            seqs[idx].append(1 if bit == "0" else -1)
    for idx, bit in enumerate(f"{int(hex_str[-1], 16):03b}"):
        seqs[idx].append(1 if bit == "0" else -1)
    m = max(len(s) for s in seqs)
    arr = np.zeros((4, m), dtype=np.int8)
    for i, s in enumerate(seqs):
        arr[i, : len(s)] = s
    return arr


TT36_SEQUENCES = _parse_hex_tt(TT36_HEX)


def tt36_sequences() -> np.ndarray:
    """Return TT(36) sequences as a padded (4, 36) int8 array."""
    return TT36_SEQUENCES.copy()


TT36_LENGTHS: np.ndarray = np.array((36, 36, 36, 35), dtype=np.int64)


# ── Turyn length helper ───────────────────────────────────────────────────────

def turyn_lengths(n: int) -> np.ndarray:
    """Return the TT(n) sequence lengths (n, n, n, n-1)."""
    return np.array((n, n, n, n - 1), dtype=np.int64)


def turyn_order(n: int) -> int:
    """Return the Hadamard order for TT(n): 4*(3*n-1)."""
    return 4 * (3 * n - 1)


# ── Sylvester construction ───────────────────────────────────────────────────

def sylvester(order: int) -> np.ndarray:
    """Sylvester Hadamard of given order (must be a power of two)."""
    if order < 1 or (order & (order - 1)) != 0:
        raise ValueError(f"order {order} is not a power of two")
    h = np.array([[1]], dtype=np.int8)
    for _ in range(order.bit_length() - 1):
        h = np.block([[h, h], [h, -h]])
    return h


# ── Paley construction ────────────────────────────────────────────────────────

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
    """Legendre symbol (x|p) for prime p."""
    if x % p == 0:
        return 0
    val = pow(x, (p - 1) // 2, p)
    return 1 if val == 1 else -1


def paley(order: int) -> np.ndarray:
    """Paley Type I Hadamard of given order for q = order-1 prime ≡ 3 (mod 4)."""
    q = order - 1
    if q < 3 or q % 4 != 3 or not _is_prime(q):
        raise ValueError(f"order {order} has no Paley construction (q={q} invalid)")
    # Build Jacobsthal matrix Q: Q[i,j] = χ(j-i mod q)
    Q = np.zeros((q, q), dtype=np.int8)
    for i in range(q):
        for j in range(q):
            Q[i, j] = _legendre((j - i) % q, q)
    # Q - I: replace diagonal zeros with -1s, gives all ±1 entries
    QminusI = Q - np.eye(q, dtype=np.int8)
    row = np.ones(order, dtype=np.int8)
    col = np.ones(order, dtype=np.int8)
    h = np.block([[np.eye(1, dtype=np.int8), row[None, 1:]],
                   [col[1:, None], QminusI]])
    return h.astype(np.int8)


# ── Registry ──────────────────────────────────────────────────────────────────

def get_known(order: int) -> np.ndarray:
    """Return a known Hadamard of the given order, or raise ValueError."""
    if order == 1:
        return np.array([[1]], dtype=np.int8)
    if order == 2:
        return np.array([[1, 1], [1, -1]], dtype=np.int8)
    # powers of 2: Sylvester
    if order > 0 and (order & (order - 1)) == 0:
        return sylvester(order)
    # Paley
    q = order - 1
    if q >= 3 and q % 4 == 3 and _is_prime(q):
        return paley(order)
    raise ValueError(f"no known Hadamard of order {order}")


def known_solution(n: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Return (sequences, lengths) for a known TT(n) solution, or None."""
    if n == 8:
        return tt8_sequences(), TT8_LENGTHS
    if n == 36:
        return tt36_sequences(), TT36_LENGTHS
    return None
