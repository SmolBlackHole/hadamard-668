"""Energy tracker with band-index fast path and recycled dG buffer."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt


class GramTracker:
    """Tracks M (N x N, int8) and G (N x N, float64) for a Hadamard construction.

    ``band_rows`` / ``band_cols`` are precomputed int16 index tables from
    Builder; multi-bit rescue uses them directly via ``_compute_delta``.
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
        self.G: npt.NDArray[np.float64] | None = None
        self._Mf64: npt.NDArray[np.float64] | None = None
        self._dG: npt.NDArray[np.float64] | None = None
        self._e: int = 0

    def _gram_energy(self, M: npt.NDArray[np.int8]) -> tuple[npt.NDArray[np.float64], int]:
        G32 = M.astype(np.float32) @ M.astype(np.float32).T
        G64 = G32.astype(np.float64)
        np.fill_diagonal(G64, 0)
        return G64, int(float(np.vdot(G64, G64)))

    def build(
        self,
        seqs: npt.NDArray[np.int8],
        band_rows: list[list[npt.NDArray[np.int16]]] | None = None,
        band_cols: list[list[npt.NDArray[np.int16]]] | None = None,
    ) -> None:
        self._k, self._n = seqs.shape
        self.M = self._fn(seqs)
        self._N = self.M.shape[0]
        self._Mf64 = self.M.astype(np.float64)
        self._dG = np.zeros((self._N, self._N), dtype=np.float64) if self._dG is None else self._dG
        self.G, self._e = self._gram_energy(self.M)
        self._rows_band = band_rows
        self._cols_band = band_cols

    def energy(self) -> int:
        return self._e

    # --- public API (used by solver) --------------------------------------------

    def _compute_delta(self, rows: npt.NDArray[np.int16], cols: npt.NDArray[np.int16]) -> int:
        """Energy after toggling every M[row, col].  Does NOT modify M or G.

        Used directly by solver._rescue for multi-bit band-index concatenation.
        """
        assert self.M is not None and self.G is not None and self._Mf64 is not None
        assert self._dG is not None

        old_vals = self.M[rows, cols].astype(np.float64)
        vn = -2.0 * old_vals
        O = self._Mf64[:, cols] * vn

        self._dG.fill(0.0)
        self._dG[:, rows] += O
        self._dG[rows, :] += O.T
        self._dG[np.ix_(rows, rows)] += np.multiply.outer(vn, vn) * (cols[:, None] == cols[None, :])

        G_new: npt.NDArray[np.float64] = self.G + self._dG
        np.fill_diagonal(G_new, 0)
        return int(float(np.vdot(G_new, G_new)))

    def flip(self, seqs: npt.NDArray[np.int8], s: int, c: int) -> int:
        """Return energy if bit (s,c) were flipped.  Does NOT modify state."""
        if not self._rows_band or not self._cols_band:
            return self._flip_full(seqs, s, c)
        return self._compute_delta(self._rows_band[s][c], self._cols_band[s][c])

    def _flip_full(self, seqs: npt.NDArray[np.int8], s: int, c: int) -> int:
        """Fallback: full rebuild for constructions without band tables."""
        seqs[s, c] *= -1
        try:
            assert self.M is not None
            M_new = self._fn(seqs)
            dM = np.subtract(M_new, self.M, dtype=np.int16)
            rn, cn = np.nonzero(dM)
            if rn.size == 0:
                return self._e
            return self._compute_delta(rn.astype(np.int16), cn.astype(np.int16))
        finally:
            seqs[s, c] *= -1

    def accept(self, seqs: npt.NDArray[np.int8], s: int, c: int) -> int:
        """Commit flip (s,c). Returns new energy."""
        if not self._rows_band or not self._cols_band:
            seqs[s, c] *= -1
            self.build(seqs)
            return self._e

        assert self.M is not None and self.G is not None and self._Mf64 is not None
        assert self._dG is not None

        seqs[s, c] *= -1
        rows = self._rows_band[s][c]
        cols = self._cols_band[s][c]

        e_new = self._compute_delta(rows, cols)

        self.M[rows, cols] *= -1
        self._Mf64[rows, cols] *= -1
        self.G += self._dG
        np.fill_diagonal(self.G, 0)
        self._e = e_new
        return self._e
