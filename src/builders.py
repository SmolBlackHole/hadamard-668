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
