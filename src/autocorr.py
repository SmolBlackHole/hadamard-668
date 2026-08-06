"""GS4 energy tracker via negaperiodic autocorrelation residuals.

For GS4 with four negacyclic sequences a,b,c,d of length n:

    r_t = NAF_a(t) + NAF_b(t) + NAF_c(t) + NAF_d(t),  t = 1..n-1
    E   = 2n * sum_{t=1}^{n-1} r_t²

A single flip at position (s,c) changes each r_t by δ_t ∈ {-4,0,+4}.

This replaces the O(N^2) GramTracker (N=4n) with an O(n) tracker —
a factor ~5000x less state at n=167.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


class AutocorrTracker:
    """Tracks GS4 orthogonality energy via NAF residuals.

    Public API is source-compatible with GramTracker (build, energy, flip,
    accept).  ``_combo_delta_native(flips)`` is used by the solver's rescue
    instead of the band-based ``_combo_delta``.
    """

    def __init__(self) -> None:
        self._n: int = 0
        self._seqs: npt.NDArray[np.int8] | None = None
        self._r: npt.NDArray[np.int32] | None = None
        self._e: int = 0
        self._rows_band: None = None  # API compat
        self._cols_band: None = None  # API compat

    # --- public API (GramTracker-compatible) ----------------------------------

    def build(
        self,
        seqs: npt.NDArray[np.int8],
        band_rows: object = None,
        band_cols: object = None,
    ) -> None:
        del band_rows, band_cols
        self._n = seqs.shape[1]
        self._seqs = seqs.copy()
        self._r = self._compute_residual(self._seqs)
        self._e = self._energy(self._r)

    def energy(self) -> int:
        return self._e

    def flip(self, _seqs: npt.NDArray[np.int8], s: int, c: int) -> int:
        assert self._seqs is not None and self._r is not None
        d = _delta(self._seqs[s], c, self._n)
        return self._energy(self._r + d)

    def accept(self, seqs: npt.NDArray[np.int8], s: int, c: int) -> int:
        assert self._seqs is not None and self._r is not None
        d = _delta(self._seqs[s], c, self._n)
        self._seqs[s, c] *= -1
        seqs[s, c] *= -1
        self._r += d
        self._e = self._energy(self._r)
        return self._e

    # --- rescue (native, no band tables) --------------------------------------

    def _combo_delta_native(self, flips: list[tuple[int, int]]) -> int:
        """Energy after applying all flips in ``flips`` (list of (s,c)).

        Brute-force via temp copy + full residual — O(n²) for n≤167 is
        comparable to GramTracker._combo_delta O(bN) but simpler.
        Switch to incremental delta-addition + pairwise correction when
        k > 5 or n > 500.
        """
        assert self._seqs is not None
        tmp = self._seqs.copy()
        for s, c in flips:
            tmp[s, c] *= -1
        return self._energy(self._compute_residual(tmp))

    # --- GramTracker _combo_delta stub (not used; _combo_delta_native replaces it)

    def _combo_delta(self, bands: object) -> int:
        raise NotImplementedError("use _combo_delta_native instead")

    # --- internal -------------------------------------------------------------

    @staticmethod
    def _naf(a: npt.NDArray[np.int8]) -> npt.NDArray[np.int32]:
        n = len(a)
        naf = np.zeros(n, dtype=np.int32)
        for t in range(n):
            forward = np.sum(a[: n - t] * a[t:])
            wraparound = np.sum(a[n - t :] * a[:t])
            naf[t] = forward - wraparound
        return naf

    @classmethod
    def _compute_residual(cls, seqs: npt.NDArray[np.int8]) -> npt.NDArray[np.int32]:
        return cls._naf(seqs[0]) + cls._naf(seqs[1]) + cls._naf(seqs[2]) + cls._naf(seqs[3])

    @staticmethod
    def _energy(r: npt.NDArray[np.int32]) -> int:
        n = len(r)
        return int(2 * n * np.sum(r[1:] ** 2))


def _delta(a: npt.NDArray[np.int8], c: int, n: int) -> npt.NDArray[np.int32]:
    """NAF delta for flipping a[c].  O(n)."""
    v = int(a[c])
    d = np.zeros(n, dtype=np.int32)
    for t in range(1, n):
        # c as first factor in NAF product
        if c + t < n:
            d[t] += -2 * v * int(a[c + t])
        else:
            d[t] += 2 * v * int(a[c + t - n])
        # c as second factor
        if t <= c:
            d[t] += -2 * v * int(a[c - t])
        else:
            d[t] += 2 * v * int(a[c - t])
    return d
