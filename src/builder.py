"""Hadamard matrix builders from ±1 sequences.

``Builder(kind="gs4", n)`` — 4 negacyclic blocks, Goethals-Seidel.
``Builder(kind="golay_2n", n)`` — 2 negacyclic blocks, Golay pair.
``Builder(kind="gs4_group", n)`` — 4 group-circulant blocks, GS4.

Each builder exposes ``.k`` (sequence count), ``.order`` (matrix size),
``.kind``, ``.build(seqs)``.
"""

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
    return sign * _circulant(values)  # int8 * int8 → int8


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


@lru_cache(maxsize=8)
def _diff_table(dims: tuple[int, ...]) -> npt.NDArray[np.int32]:
    n = int(np.prod(dims))
    coords = np.stack(np.unravel_index(np.arange(n), dims), axis=0)
    dims_arr = np.array(dims, dtype=np.int64)
    diff = (coords[:, :, None] - coords[:, None, :]) % dims_arr[:, None, None]
    return np.asarray(np.ravel_multi_index(diff, dims), dtype=np.int32)


# --- Builder ------------------------------------------------------------------


class Builder:
    """Hadamard matrix from ±1 sequences.

    ``Builder(kind="gs4", n=167)`` — 4 negacyclic blocks, Goethals-Seidel.
    ``Builder(kind="golay_2n", n=334)`` — 2 negacyclic blocks, Golay pair.
    ``Builder(kind="gs4_group", n=167)`` — 4 group-circulant blocks, GS4.
    """

    _KINDS: ClassVar[tuple[str, ...]] = ("gs4", "golay_2n", "gs4_group")

    # kind → (k, order_factor, divisor)
    _CONFIG: ClassVar[dict[str, tuple[int, int, int]]] = {
        "gs4": (4, 4, 4),
        "golay_2n": (2, 2, 2),
        "gs4_group": (4, 4, 4),
    }

    @classmethod
    def k_for(cls, kind: str) -> int:
        """Return the divisor for a strategy kind."""
        if kind not in cls._CONFIG:
            raise ValueError(f"unknown kind {kind!r}")
        return cls._CONFIG[kind][2]

    @staticmethod
    def factorize(n: int) -> list[int]:
        """Best 2-factor split for group-circulant construction."""
        pairs = [(p, n // p) for p in range(3, int(n**0.5) + 1) if n % p == 0 and n // p >= 3]
        if not pairs:
            return [n]
        p, q = min(pairs, key=lambda pq: abs(pq[0] - pq[1]))
        return sorted([p, q], reverse=True)

    def __init__(self, *, kind: str, n: int) -> None:
        if kind not in self._KINDS:
            raise ValueError(f"kind must be one of {self._KINDS}, got {kind!r}")
        self.kind = kind
        self.n = n
        cfg = self._CONFIG[kind]
        self.k: int = cfg[0]
        self.order: int = cfg[1] * n
        self._dims: list[int] = self.factorize(n) if kind == "gs4_group" else []

        self._build = {
            "gs4": self._build_gs4,
            "golay_2n": self._build_2n,
            "gs4_group": self._build_group,
        }[kind]

        self.label: str = " [2N]" if kind == "golay_2n" else ""

        # lazy band-index tables (int16, ~4n entries per position)
        self._band: (
            tuple[list[list[npt.NDArray[np.int16]]], list[list[npt.NDArray[np.int16]]]] | None
        ) = None

    @property
    def band_rows(self) -> list[list[npt.NDArray[np.int16]]]:
        if self._band is None:
            self._band = self._make_band()
        return self._band[0]

    @property
    def band_cols(self) -> list[list[npt.NDArray[np.int16]]]:
        if self._band is None:
            self._band = self._make_band()
        return self._band[1]

    def _make_band(
        self,
    ) -> tuple[list[list[npt.NDArray[np.int16]]], list[list[npt.NDArray[np.int16]]]]:
        """Build band-index tables for this construction (lazy, cached)."""
        if self.kind == "golay_2n":
            return _build_band_2n(self.n)
        if self.kind == "gs4_group":
            return _build_band_group(self.n)
        return _build_band_gs4(self.n)

    def build(self, seqs: npt.NDArray[np.int8]) -> npt.NDArray[np.int8]:
        """Build matrix from sequences. seqs shape is (self.k, self.n)."""
        return self._build(seqs)

    def _build_gs4(self, seqs: npt.NDArray[np.int8]) -> npt.NDArray[np.int8]:
        return _gs4_block(
            _negacirculant(seqs[0]),
            _negacirculant(seqs[1]),
            _negacirculant(seqs[2]),
            _negacirculant(seqs[3]),
        )

    def _build_2n(self, seqs: npt.NDArray[np.int8]) -> npt.NDArray[np.int8]:
        A = _negacirculant(seqs[0])
        B = _negacirculant(seqs[1])
        return np.block([[A, B], [-B.T, A.T]])

    def _build_group(self, seqs: npt.NDArray[np.int8]) -> npt.NDArray[np.int8]:
        dt = _diff_table(tuple(self._dims))
        rn = self.n

        def circ(flat: npt.NDArray[np.int8]) -> npt.NDArray[np.int8]:
            return flat[dt]

        return _gs4_block(
            circ(seqs[0, :rn]),
            circ(seqs[1, :rn]),
            circ(seqs[2, :rn]),
            circ(seqs[3, :rn]),
        )


# --- Band-index builders -------------------------------------------------------


def _build_band_gs4(
    n: int,
) -> tuple[list[list[npt.NDArray[np.int16]]], list[list[npt.NDArray[np.int16]]]]:
    """Rows/cols band for 4N negacyclic Goethals-Seidel."""
    GS4_MAP: list[list[tuple[int, int, str, int]]] = [
        [(0, 0, "direct", 1), (1, 1, "direct", 1), (2, 2, "direct", 1), (3, 3, "direct", 1)],
        [(0, 1, "rev", 1), (1, 0, "rev", -1), (2, 3, "tr", -1), (3, 2, "tr", 1)],
        [(0, 2, "rev", 1), (2, 0, "rev", -1), (1, 3, "tr", 1), (3, 1, "tr", -1)],
        [(0, 3, "rev", 1), (3, 0, "rev", -1), (2, 1, "tr", 1), (1, 2, "tr", -1)],
    ]
    return _compute_band(n, 4, GS4_MAP)


def _build_band_2n(
    n: int,
) -> tuple[list[list[npt.NDArray[np.int16]]], list[list[npt.NDArray[np.int16]]]]:
    """Rows/cols band for 2N negaperiodic Golay pair."""
    _2N_MAP: list[list[tuple[int, int, str, int]]] = [
        [(0, 0, "direct", 1), (1, 1, "tr_direct", 1)],
        [(0, 1, "direct", 1), (1, 0, "tr_direct", -1)],
    ]
    return _compute_band(n, 2, _2N_MAP)


def _build_band_group(
    _n: int,
) -> tuple[list[list[npt.NDArray[np.int16]]], list[list[npt.NDArray[np.int16]]]]:
    """No band for group-circulant — uses diff_table which isn't a simple circulant."""
    return [], []


def _compute_band(
    n: int,
    k: int,
    block_map: list[list[tuple[int, int, str, int]]],
) -> tuple[list[list[npt.NDArray[np.int16]]], list[list[npt.NDArray[np.int16]]]]:
    rows_band: list[list[npt.NDArray[np.int16]]] = []
    cols_band: list[list[npt.NDArray[np.int16]]] = []
    for s in range(k):
        seq_rows: list[npt.NDArray[np.int16]] = []
        seq_cols: list[npt.NDArray[np.int16]] = []
        for c in range(n):
            row_list: list[int] = []
            col_list: list[int] = []
            for br, bc, transform, _mult in block_map[s]:
                for r in range(n):
                    if transform == "direct":
                        row, col = r, (r + c) % n
                    elif transform == "rev":
                        col_orig = (r + c) % n
                        row, col = r, n - 1 - col_orig
                    elif transform == "tr":
                        row, col = (r + c) % n, n - 1 - r
                    elif transform == "tr_direct":
                        row, col = (r + c) % n, r
                    else:
                        continue
                    row_list.append(br * n + row)
                    col_list.append(bc * n + col)
            seq_rows.append(np.array(row_list, dtype=np.int16))
            seq_cols.append(np.array(col_list, dtype=np.int16))
        rows_band.append(seq_rows)
        cols_band.append(seq_cols)
    return rows_band, cols_band
