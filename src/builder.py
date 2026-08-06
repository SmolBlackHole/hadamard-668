"""Hadamard matrix builder: GS4 Goethals-Seidel from four +/-1 sequences."""

from __future__ import annotations

from functools import lru_cache
from typing import ClassVar

import numpy as np
import numpy.typing as npt

# --- low-level helpers --------------------------------------------------------


@lru_cache(maxsize=8)
def _circ_idx(n: int) -> npt.NDArray[np.int64]:
    return (np.arange(n)[None, :] - np.arange(n)[:, None]) % n


@lru_cache(maxsize=8)
def _negacirc_sign(n: int) -> npt.NDArray[np.int8]:
    return np.where(np.arange(n)[None, :] >= np.arange(n)[:, None], np.int8(1), np.int8(-1))


def _circulant(values: npt.NDArray[np.int8]) -> npt.NDArray[np.int8]:
    return values[_circ_idx(values.size)]  # type: ignore[return-value]


def _negacirculant(values: npt.NDArray[np.int8]) -> npt.NDArray[np.int8]:
    sign = _negacirc_sign(values.size)
    return sign * _circulant(values)  # int8 * int8 -> int8


def _gs4_block(
    A: npt.NDArray[np.int8],
    B: npt.NDArray[np.int8],
    C: npt.NDArray[np.int8],
    D: npt.NDArray[np.int8],
) -> npt.NDArray[np.int8]:
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

    @classmethod
    def class_hash(cls, seqs: npt.NDArray[np.int8]) -> str:
        """Invariant hash for deduplication.  Dispatches on ``self.kind``."""
        from fast_hash import gs4_class_hash

        return gs4_class_hash(seqs)

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

    def build(self, seqs: npt.NDArray[np.int8]) -> npt.NDArray[np.int8]:
        return _gs4_block(
            _negacirculant(seqs[0]),
            _negacirculant(seqs[1]),
            _negacirculant(seqs[2]),
            _negacirculant(seqs[3]),
        )
