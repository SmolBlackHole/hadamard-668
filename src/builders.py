"""Goethals-Seidel and Turyn matrix assembly."""

from __future__ import annotations

import numpy as np


def _circulant(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.int8)
    n = values.size
    return values[(np.arange(n)[None, :] - np.arange(n)[:, None]) % n]


def build_goethals_seidel(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
) -> np.ndarray:
    A, B, C, D = (_circulant(s) for s in (a, b, c, d))
    BR, CR, DR = B[:, ::-1], C[:, ::-1], D[:, ::-1]
    BtR, CtR, DtR = B.T[:, ::-1], C.T[:, ::-1], D.T[:, ::-1]
    return np.block(
        [
            [A, BR, CR, DR],
            [-BR, A, -DtR, CtR],
            [-CR, DtR, A, -BtR],
            [-DR, -CtR, BtR, A],
        ]
    ).astype(np.int8)


def _sign_sequence(values: np.ndarray, name: str) -> np.ndarray:
    sequence = np.asarray(values, dtype=np.int8)
    if sequence.ndim != 1 or not np.all(np.isin(sequence, (-1, 1))):
        raise ValueError(f"{name} must be a one-dimensional sign sequence")
    return sequence


def turyn_to_base(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    w: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x, y, z, w = (_sign_sequence(v, n) for v, n in zip((x, y, z, w), "XYZW", strict=False))
    size = len(x)
    if len(y) != size or len(z) != size or len(w) != size - 1:
        raise ValueError("Turyn sequences must have lengths (n, n, n, n-1)")
    return (np.concatenate((z, w)), np.concatenate((z, -w)), x.copy(), y.copy())


def base_to_t_sequences(
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    d: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    a, b, c, d = (_sign_sequence(v, n) for v, n in zip((a, b, c, d), "ABCD", strict=False))
    if len(a) != len(b) or len(c) != len(d) or len(a) != 2 * len(c) - 1:
        raise ValueError("base sequences must have lengths (2n-1, 2n-1, n, n)")
    zs = np.zeros(len(c), dtype=np.int8)
    zl = np.zeros(len(a), dtype=np.int8)
    return (
        np.concatenate(((a + b) // 2, zs)),
        np.concatenate(((a - b) // 2, zs)),
        np.concatenate((zl, (c + d) // 2)),
        np.concatenate((zl, (c - d) // 2)),
    )


def t_sequences_to_sign_sequences(
    t1: np.ndarray,
    t2: np.ndarray,
    t3: np.ndarray,
    t4: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    sequences = tuple(np.asarray(v, dtype=np.int8) for v in (t1, t2, t3, t4))
    if any(s.ndim != 1 for s in sequences) or len({len(s) for s in sequences}) != 1:
        raise ValueError("T-sequences must be one-dimensional and equally long")
    occupancy = np.sum(np.abs(np.stack(sequences)), axis=0)
    if not np.all(occupancy == 1):
        raise ValueError("T-sequences must have exactly one non-zero value per position")
    t1, t2, t3, t4 = sequences
    result = (
        t1 + t2 + t3 + t4,
        -t1 + t2 + t3 - t4,
        -t1 - t2 + t3 + t4,
        -t1 + t2 - t3 + t4,
    )
    if not all(np.all(np.isin(s, (-1, 1))) for s in result):
        raise ValueError("T-sequences did not produce sign sequences")
    return result


def build_turyn(
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    w: np.ndarray,
) -> np.ndarray:
    return build_goethals_seidel(
        *t_sequences_to_sign_sequences(*base_to_t_sequences(*turyn_to_base(x, y, z, w)))
    )
