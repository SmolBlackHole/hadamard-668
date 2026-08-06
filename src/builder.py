"""Hadamard matrix builders from ±1 sequences.

``Builder(kind="gs4", n)`` — 4 negacyclic blocks, Goethals-Seidel.
``Builder(kind="golay_2n", n)`` — 2 negacyclic blocks, Golay pair.
``Builder(kind="gs4_group", n)`` — 4 group-circulant blocks, GS4.

Each builder exposes ``.k`` (sequence count), ``.order`` (matrix size),
``.kind``, ``.build(seqs)``.
"""

from __future__ import annotations

import numpy as np

# ── low-level helpers ────────────────────────────────────────────────────────

_IDX_CACHE: dict[int, np.ndarray] = {}
_SIGN_CACHE: dict[int, np.ndarray] = {}


def _circulant(values):
    values = np.asarray(values, dtype=np.int8)
    n = values.size
    if n not in _IDX_CACHE:
        _IDX_CACHE[n] = (np.arange(n)[None, :] - np.arange(n)[:, None]) % n
        _SIGN_CACHE[n] = np.where(np.arange(n)[None, :] >= np.arange(n)[:, None], 1, -1)
    return values[_IDX_CACHE[n]]


def _negacirculant(values):
    c = _circulant(values)  # populates cache first
    return (_SIGN_CACHE[values.size] * c).astype(np.int8)


def _gs4_block(A, B, C, D):
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


_DIFF_CACHE: dict[tuple[int, ...], np.ndarray] = {}


def _diff_table(dims):
    if dims not in _DIFF_CACHE:
        n = int(np.prod(dims))
        coords = np.stack(np.unravel_index(np.arange(n), dims), axis=0)
        dims_arr = np.array(dims, dtype=np.int64)
        diff = (coords[:, :, None] - coords[:, None, :]) % dims_arr[:, None, None]
        table = np.ravel_multi_index(diff, dims)
        _DIFF_CACHE[dims] = np.asarray(table, dtype=np.int32)
    return _DIFF_CACHE[dims]


# ── Builder ──────────────────────────────────────────────────────────────────


class Builder:
    """Hadamard matrix from ±1 sequences.

    ``Builder(kind="gs4", n=167)`` — 4 negacyclic blocks, Goethals-Seidel.
    ``Builder(kind="golay_2n", n=334)`` — 2 negacyclic blocks, Golay pair.
    ``Builder(kind="gs4_group", n=167)`` — 4 group-circulant blocks, GS4.
    """

    _KINDS = ("gs4", "golay_2n", "gs4_group")

    def __init__(self, *, kind: str, n: int):
        if kind not in self._KINDS:
            raise ValueError(f"kind must be one of {self._KINDS}, got {kind!r}")
        self.kind = kind
        self.n = n
        self._dims = _best_factorization(n) if kind == "gs4_group" else None

    @property
    def k(self) -> int:
        """Number of sequences needed."""
        return 2 if self.kind == "golay_2n" else 4

    @property
    def order(self) -> int:
        """Resulting Hadamard matrix order."""
        return 2 * self.n if self.kind == "golay_2n" else 4 * self.n

    @property
    def label(self) -> str:
        if self.kind == "golay_2n":
            return " [2N]"
        if self._dims:
            return f" [{self._dims}]"
        return ""

    def build(self, seqs):
        """Build matrix from sequences. seqs shape is (k, n)."""
        if self.kind == "golay_2n":
            A = _negacirculant(seqs[0])
            B = _negacirculant(seqs[1])
            return np.block([[A, B], [-B.T, A.T]]).astype(np.int8)
        if self.kind == "gs4_group":
            dt = _diff_table(tuple(self._dims))

            def circ(flat):
                return flat[dt].astype(np.int8)

            return _gs4_block(
                circ(seqs[0, : self.n]),
                circ(seqs[1, : self.n]),
                circ(seqs[2, : self.n]),
                circ(seqs[3, : self.n]),
            )
        return _gs4_block(
            _negacirculant(seqs[0]),
            _negacirculant(seqs[1]),
            _negacirculant(seqs[2]),
            _negacirculant(seqs[3]),
        )


def _best_factorization(n: int) -> list[int]:
    pairs = [(p, n // p) for p in range(3, int(n**0.5) + 1) if n % p == 0 and n // p >= 3]
    if not pairs:
        return [n]
    p, q = min(pairs, key=lambda pq: abs(pq[0] - pq[1]))
    return sorted([p, q], reverse=True)
