"""Hadamard matrix builder: GS4 Goethals-Seidel from four +/-1 sequences."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import ClassVar

import numpy as np
import numpy.typing as npt

Int8Array = npt.NDArray[np.int8]
Int64Array = npt.NDArray[np.int64]


@dataclass(frozen=True)
class MatrixMetrics:
    energy: int
    orthogonal_pairs: int
    max_abs_correlation: int


def gram_matrix(matrix: npt.ArrayLike) -> Int64Array:
    values = np.asarray(matrix, dtype=np.int64)
    gram = values @ values.T
    np.fill_diagonal(gram, 0)
    return gram


def check_orthogonality(matrix: npt.ArrayLike) -> MatrixMetrics:
    gram = gram_matrix(matrix)
    size = gram.shape[0]
    pair_count = size * (size - 1) // 2
    if pair_count == 0:
        return MatrixMetrics(0, 0, 0)
    upper = gram[np.triu_indices(size, k=1)]
    return MatrixMetrics(
        energy=int(upper @ upper),
        orthogonal_pairs=int(np.count_nonzero(upper == 0)),
        max_abs_correlation=int(np.abs(upper).max(initial=0)),
    )


# --- low-level helpers --------------------------------------------------------


@lru_cache(maxsize=8)
def _circ_idx(n: int) -> Int64Array:
    return (np.arange(n)[None, :] - np.arange(n)[:, None]) % n


@lru_cache(maxsize=8)
def _negacirc_sign(n: int) -> Int8Array:
    return np.where(np.arange(n)[None, :] >= np.arange(n)[:, None], np.int8(1), np.int8(-1))


def _circulant(values: Int8Array) -> Int8Array:
    return values[_circ_idx(values.size)]  # type: ignore[return-value]


def _negacirculant(values: Int8Array) -> Int8Array:
    sign = _negacirc_sign(values.size)
    return sign * _circulant(values)  # int8 * int8 -> int8


def _gs4_block(
    A: Int8Array,
    B: Int8Array,
    C: Int8Array,
    D: Int8Array,
) -> Int8Array:
    BR, CR, DR = B[:, ::-1], C[:, ::-1], D[:, ::-1]
    BtR, CtR, DtR = B.T[:, ::-1], C.T[:, ::-1], D.T[:, ::-1]
    return np.block(
        [
            [A, BR, CR, DR],
            [-BR, A, -DtR, CtR],
            [-CR, DtR, A, -BtR],
            [-DR, -CtR, BtR, A],
        ]
    )


class Builder:
    """Hadamard matrix from four +/-1 sequences via Goethals-Seidel.

    ``Builder(kind="gs4", n=167)`` — 4 negacyclic blocks, Goethals-Seidel.
    """

    _CONFIG: ClassVar[dict[str, tuple[int, int, int]]] = {
        "gs4": (4, 4, 4),
    }

    @classmethod
    def k_for(cls, kind: str) -> int:
        if kind not in cls._CONFIG:
            raise ValueError(f"unknown kind {kind!r}")
        return cls._CONFIG[kind][2]

    @staticmethod
    def factorize(n: int) -> list[int]:
        pairs = [(p, n // p) for p in range(3, int(n**0.5) + 1) if n % p == 0 and n // p >= 3]
        if not pairs:
            return [n]
        p, q = min(pairs, key=lambda pq: abs(pq[0] - pq[1]))
        return sorted([p, q], reverse=True)

    def __init__(self, *, kind: str, n: int) -> None:
        if kind not in self._CONFIG:
            raise ValueError(f"kind must be one of {tuple(self._CONFIG)}, got {kind!r}")
        self.kind = kind
        self.n = n
        cfg = self._CONFIG[kind]
        self.k: int = cfg[0]
        self.order: int = cfg[1] * n

    def build(self, seqs: Int8Array) -> Int8Array:
        return _gs4_block(
            _negacirculant(seqs[0]),
            _negacirculant(seqs[1]),
            _negacirculant(seqs[2]),
            _negacirculant(seqs[3]),
        )
