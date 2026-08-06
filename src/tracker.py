"""Energy tracker with band-index fast path.

Uses float32 Gram matrix (all intermediate values are exact integers ≤ 2²⁴).
Single-flip energy: closed-form incremental formula, no dG buffer needed.
Multi-bit rescue: per-band scatter (avoids NumPy duplicate-index bug).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt


class GramTracker:
    """Tracks M (N x N, int8) and G (N x N, float32) for a Hadamard construction.

    ``band_rows`` / ``band_cols`` are precomputed int16 index tables from
    Builder; multi-bit rescue uses them directly via ``_combo_delta``.
    """

    def __init__(
        self,
        build_fn: Callable[[npt.NDArray[np.int8]], npt.NDArray[np.int8]],
    ) -> None:
        self._fn = build_fn
        self._k: int = 0
        self._n: int = 0
        self._N: int = 0
        self._rows_band: list[list[npt.NDArray[np.int16]]] | None = None
        self._cols_band: list[list[npt.NDArray[np.int16]]] | None = None
        self.M: npt.NDArray[np.int8] | None = None
        self.G: npt.NDArray[np.float32] | None = None
        self._MfT: npt.NDArray[np.float32] | None = None  # M^T in float32
        self._dG: npt.NDArray[np.float32] | None = None  # recycled combo buffer
        self._e: int = 0

    def _gram_energy(self, M: npt.NDArray[np.int8]) -> tuple[npt.NDArray[np.float32], int]:
        Mf = M.astype(np.float32)
        G = Mf @ Mf.T
        np.fill_diagonal(G, 0)
        e = int(float(np.einsum("ij,ij->", G, G, dtype=np.float64)))
        return G, e

    def build(
        self,
        seqs: npt.NDArray[np.int8],
        band_rows: list[list[npt.NDArray[np.int16]]] | None = None,
        band_cols: list[list[npt.NDArray[np.int16]]] | None = None,
    ) -> None:
        self._k, self._n = seqs.shape
        self.M = self._fn(seqs)
        self._N = self.M.shape[0]
        self._MfT = self.M.T.astype(np.float32)
        self._dG = np.zeros((self._N, self._N), dtype=np.float32) if self._dG is None else self._dG
        self.G, self._e = self._gram_energy(self.M)
        self._rows_band = band_rows
        self._cols_band = band_cols

    def energy(self) -> int:
        return self._e

    # --- fast single-flip delta (closed-form, no dG buffer) -------------------

    def _single(self, rows: npt.NDArray[np.int16], cols: npt.NDArray[np.int16]) -> int:
        """Energy delta for toggling M[rows, cols].  Exact, float32-safe.

        E_new = E + 4*T1 + 2*S_O + 2*S_x + 16*trace(Orr) + 16*b
        where S_O = 4*b*N (every O entry is ±2 since band cols are a permutation).
        """
        assert self.M is not None and self.G is not None and self._MfT is not None

        b = rows.size
        vn = -2.0 * self.M[rows, cols].astype(np.float32)
        O_T = self._MfT[cols, :] * vn[:, None]  # b x N, contiguous row gather
        Orr = O_T[:, rows]  # b x b

        T1 = float(np.einsum("ij,ij->", O_T, self.G[rows, :], dtype=np.float64))
        S_O = 4.0 * b * self._N
        S_x = float(np.einsum("ij,ji->", Orr, Orr, dtype=np.float64))

        self._O_T_scratch = O_T  # stash for accept
        return int(4 * T1 + 2 * S_O + 2 * S_x + 16.0 * float(np.trace(Orr)) + 16.0 * b)

    # --- backward-compatible wrapper (used by tests) --------------------------

    def _compute_delta(self, rows: npt.NDArray[np.int16], cols: npt.NDArray[np.int16]) -> int:
        """Absolute energy after toggling M[rows, cols].  Does NOT modify M or G."""
        return self._e + self._single(rows, cols)

    # --- flip / accept --------------------------------------------------------

    def flip(self, _seqs: npt.NDArray[np.int8], s: int, c: int) -> int:
        if not self._rows_band or not self._cols_band:
            return self._flip_full(_seqs, s, c)
        return self._e + self._single(self._rows_band[s][c], self._cols_band[s][c])

    def _flip_full(self, seqs: npt.NDArray[np.int8], s: int, c: int) -> int:
        seqs[s, c] *= -1
        try:
            assert self.M is not None and self._MfT is not None
            M_new = self._fn(seqs)
            dM = np.subtract(M_new, self.M, dtype=np.int16)
            rn, cn = np.nonzero(dM)
            if rn.size == 0:
                return self._e
            return self._e + self._single(rn.astype(np.int16), cn.astype(np.int16))
        finally:
            seqs[s, c] *= -1

    def accept(self, seqs: npt.NDArray[np.int8], s: int, c: int) -> int:
        if not self._rows_band or not self._cols_band:
            seqs[s, c] *= -1
            self.build(seqs)
            return self._e

        assert self.M is not None and self.G is not None and self._MfT is not None

        seqs[s, c] *= -1
        rows = self._rows_band[s][c]
        cols = self._cols_band[s][c]

        e_new = self._e + self._single(rows, cols)
        O = self._O_T_scratch.T  # N x b, from _single

        self.M[rows, cols] *= -1
        self._MfT[cols, rows] *= -1
        self.G[:, rows] += O
        self.G[rows, :] += O.T
        np.fill_diagonal(self.G, 0)
        self._e = e_new
        return self._e

    # --- correct multi-bit delta (per-band scatter withoutno duplicate-index) ----

    def _combo_delta(self, bands: list[tuple[npt.NDArray[np.int16], npt.NDArray[np.int16]]]) -> int:
        """Energy after toggling multiple flips.  Exact per-band accumulation.

        Each band's rows/cols are unique -> += is exact. Cross-band terms
        are scattered sparsely along matched column pairs.
        """
        assert self.M is not None and self.G is not None and self._MfT is not None
        assert self._dG is not None

        self._dG.fill(0.0)
        vns: list[tuple[npt.NDArray[np.float32], npt.NDArray[np.int16], npt.NDArray[np.int16]]] = []

        for rows, cols in bands:
            vn = -2.0 * self.M[rows, cols].astype(np.float32)
            O = self._MfT[cols, :].T * vn
            self._dG[:, rows] += O
            self._dG[rows, :] += O.T
            vns.append((vn, rows, cols))

        # cross-band sparse scatter
        for f in range(len(bands)):
            for g in range(f + 1, len(bands)):
                vnA, rowsA, colsA = vns[f]
                vnB, rowsB, colsB = vns[g]
                j2: npt.NDArray[np.int64] = np.argsort(colsB)[colsA]
                vals = vnA * vnB[j2]
                self._dG[rowsA, rowsB[j2]] += vals
                self._dG[rowsB[j2], rowsA] += vals

        G_new = self.G + self._dG
        np.fill_diagonal(G_new, 0)
        return int(float(np.einsum("ij,ij->", G_new, G_new, dtype=np.float64)))
