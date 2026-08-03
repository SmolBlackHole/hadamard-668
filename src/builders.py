"""Expand compact cyclic sequences into Hadamard matrix candidates."""
from __future__ import annotations

import numpy as np


def _circulant(values: np.ndarray) -> np.ndarray:
    return np.array([np.roll(values, index) for index in range(len(values))], dtype=np.int8)


def build_propus(a: np.ndarray, b: np.ndarray, d: np.ndarray) -> np.ndarray:
    """Build a Propus candidate from three cyclic sequences."""
    A, B, D = (_circulant(sequence) for sequence in (a, b, d))
    return np.block([
        [A, B, B, D], [B, D, -A, -B], [B, -A, -D, B], [D, -B, B, -A],
    ]).astype(np.int8)


def build_goethals_seidel(
    a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray,
) -> np.ndarray:
    """Build a Goethals-Seidel candidate from four cyclic sequences."""
    A, B, C, D = (_circulant(sequence) for sequence in (a, b, c, d))
    BR, CR, DR = B[:, ::-1], C[:, ::-1], D[:, ::-1]
    BtR, CtR, DtR = B.T[:, ::-1], C.T[:, ::-1], D.T[:, ::-1]
    return np.block([
        [A, BR, CR, DR], [-BR, A, -DtR, CtR],
        [-CR, DtR, A, -BtR], [-DR, -CtR, BtR, A],
    ]).astype(np.int8)


def _sign_sequence(values: np.ndarray, name: str) -> np.ndarray:
    sequence = np.asarray(values, dtype=np.int8)
    if sequence.ndim != 1 or not np.all(np.isin(sequence, (-1, 1))):
        raise ValueError(f"{name} must be a one-dimensional sign sequence")
    return sequence


def turyn_to_base(
    x: np.ndarray, y: np.ndarray, z: np.ndarray, w: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert TT(n) sign sequences to base sequences of lengths 2n-1,2n-1,n,n."""
    x, y, z, w = (_sign_sequence(values, name) for values, name in
                  zip((x, y, z, w), "XYZW"))
    size = len(x)
    if len(y) != size or len(z) != size or len(w) != size - 1:
        raise ValueError("Turyn sequences must have lengths (n, n, n, n-1)")
    return (
        np.concatenate((z, w)), np.concatenate((z, -w)), x.copy(), y.copy(),
    )


def base_to_t_sequences(
    a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Convert compatible base sequences to four disjoint T-sequences."""
    a, b, c, d = (_sign_sequence(values, name) for values, name in
                  zip((a, b, c, d), "ABCD"))
    if len(a) != len(b) or len(c) != len(d) or len(a) != 2 * len(c) - 1:
        raise ValueError("base sequences must have lengths (2n-1, 2n-1, n, n)")
    zero_short = np.zeros(len(c), dtype=np.int8)
    zero_long = np.zeros(len(a), dtype=np.int8)
    return (
        np.concatenate(((a + b) // 2, zero_short)),
        np.concatenate(((a - b) // 2, zero_short)),
        np.concatenate((zero_long, (c + d) // 2)),
        np.concatenate((zero_long, (c - d) // 2)),
    )


def t_sequences_to_sign_sequences(
    t1: np.ndarray, t2: np.ndarray, t3: np.ndarray, t4: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Turn four disjoint T-sequences into four equal-length sign sequences."""
    sequences = tuple(np.asarray(values, dtype=np.int8)
                      for values in (t1, t2, t3, t4))
    if any(sequence.ndim != 1 for sequence in sequences) or len({len(sequence) for sequence in sequences}) != 1:
        raise ValueError(
            "T-sequences must be one-dimensional and equally long")
    occupancy = np.sum(np.abs(np.stack(sequences)), axis=0)
    if not np.all(occupancy == 1):
        raise ValueError(
            "T-sequences must have exactly one non-zero value per position")
    t1, t2, t3, t4 = sequences
    result = (
        t1 + t2 + t3 + t4,
        -t1 + t2 + t3 - t4,
        -t1 - t2 + t3 + t4,
        -t1 + t2 - t3 + t4,
    )
    if not all(np.all(np.isin(sequence, (-1, 1))) for sequence in result):
        raise ValueError("T-sequences did not produce sign sequences")
    return result


def build_turyn(x: np.ndarray, y: np.ndarray, z: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Build a Hadamard candidate from TT(n) sequences via the G-S array."""
    return build_goethals_seidel(*t_sequences_to_sign_sequences(
        *base_to_t_sequences(*turyn_to_base(x, y, z, w))))
