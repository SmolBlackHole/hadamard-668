"""GS4 energy tracker via negaperiodic autocorrelation residuals.

For GS4 with four negacyclic sequences a,b,c,d of length n:

    r_t = NAF_a(t) + NAF_b(t) + NAF_c(t) + NAF_d(t),  t = 1..n-1
    E   = 2n * sum_{t=1}^{n-1} r_t²

Delta cache: all 4n delta vectors precomputed at build() and incrementally
updated on accept().  flip() uses the closed-form delta-E formula:

    dE = 4n·dot(r1, d1) + 2n·dot(d1, d1)

avoiding temporary arrays on the hot path.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt


class Tracker:
    """Tracks GS4 orthogonality energy via NAF residuals with delta cache."""

    def __init__(self) -> None:
        self._n: int = 0
        self._seqs: npt.NDArray[np.int8] | None = None
        self._r: npt.NDArray[np.int32] | None = None
        self._r1: npt.NDArray[np.int32] | None = None  # r[1:] view for dot products
        self._e: int = 0
        self._delta: npt.NDArray[np.int32] | None = None  # (4*n, n) cache
        self._rows_band: None = None
        self._cols_band: None = None

    # --- public API -----------------------------------------------------------

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
        self._r1 = self._r[1:]
        self._e = self._energy_from_r1(self._r1)
        self._delta = _build_delta_cache(self._seqs, self._n)

    def energy(self) -> int:
        return self._e

    def flip(self, _seqs: npt.NDArray[np.int8], s: int, c: int) -> int:
        assert self._r1 is not None and self._delta is not None
        d1 = self._delta[s * self._n + c, 1:]
        return self._e + self._delta_e(self._r1, d1)

    def accept(self, seqs: npt.NDArray[np.int8], s: int, c: int) -> int:
        assert self._seqs is not None and self._r is not None and self._delta is not None
        r = self._r
        d = self._delta[s * self._n + c]
        self._seqs[s, c] *= -1
        seqs[s, c] *= -1
        r += d
        r1 = r[1:]
        self._r1 = r1
        _rebuild_delta_slice(self._delta, self._seqs[s], s, self._n)
        self._e = self._energy_from_r1(r1)
        return self._e

    # --- rescue ---------------------------------------------------------------

    def _combo_delta_native(self, flips: list[tuple[int, int]]) -> int:
        """Energy after toggling multiple flips.

        Cross-sequence:    add cached deltas, closed-form energy delta.
        Same-sequence 2:   delta1 + delta2 + O(1) correction.
        Same-sequence 3+:  brute-force rebuild (rare).
        """
        assert self._r is not None and self._delta is not None and self._seqs is not None
        n = self._n

        by_seq: dict[int, list[int]] = {}
        for s, c in flips:
            by_seq.setdefault(s, []).append(c)

        # all different sequences — sum deltas, one dot per flip
        if all(len(cs) == 1 for cs in by_seq.values()):
            r1 = self._r[1:].astype(np.int64)
            e = self._e
            for s, cs in by_seq.items():
                d1 = self._delta[s * n + cs[0], 1:]
                e += self._delta_e(r1, d1)
                r1 += d1
            return e

        # same-sequence pair with O(1) correction
        if len(flips) == 2 and len(by_seq) == 1:
            s, cs = next(iter(by_seq.items()))
            c1, c2 = cs[0], cs[1]
            r1 = self._r[1:].astype(np.int64)
            d1_1 = self._delta[s * n + c1, 1:]
            d1_2 = self._delta[s * n + c2, 1:]
            e = self._e + self._delta_e(r1, d1_1 + d1_2)
            t = abs(c1 - c2)
            if t != 0 and t != n // 2:
                v1 = int(self._seqs[s, c1])
                v2 = int(self._seqs[s, c2])
                # correction adds ±4·v1·v2 at distances t and n-t
                r_t = int(r1[t - 1]) + int(d1_1[t - 1]) + int(d1_2[t - 1])
                corr = 4 * v1 * v2
                e += int(4 * n * r_t * corr + 2 * n * corr * corr)
                r_nt_pos = n - t - 1
                r_nt = int(r1[r_nt_pos]) + int(d1_1[r_nt_pos]) + int(d1_2[r_nt_pos])
                corr2 = -corr
                e += int(4 * n * r_nt * corr2 + 2 * n * corr2 * corr2)
            return e

        # 3+ flips in same sequence — rare
        tmp = self._seqs.copy()
        for s, c in flips:
            tmp[s, c] *= -1
        return self._energy_from_r1(self._compute_residual(tmp)[1:])

    def _combo_delta(self, bands: object) -> int:
        raise NotImplementedError("use _combo_delta_native instead")

    # --- internal -------------------------------------------------------------

    @staticmethod
    def _energy_from_r1(r1: npt.NDArray[np.int32]) -> int:
        n_plus_1 = len(r1) + 1
        return int(2 * n_plus_1 * np.dot(r1, r1))

    @staticmethod
    def _delta_e(r1: npt.NDArray[np.int64 | np.int32], d1: npt.NDArray) -> int:
        n_plus_1 = len(r1) + 1
        return int(4 * n_plus_1 * np.dot(r1, d1) + 2 * n_plus_1 * np.dot(d1, d1))

    @classmethod
    def _compute_residual(cls, seqs: npt.NDArray[np.int8]) -> npt.NDArray[np.int32]:
        return cls._naf(seqs[0]) + cls._naf(seqs[1]) + cls._naf(seqs[2]) + cls._naf(seqs[3])

    @staticmethod
    def _naf(a: npt.NDArray[np.int8]) -> npt.NDArray[np.int32]:
        n = len(a)
        ai = a.astype(np.int32)
        naf = np.empty(n, dtype=np.int32)
        for t in range(n):
            forward = np.dot(ai[: n - t], ai[t:])
            wraparound = np.dot(ai[n - t :], ai[:t])
            naf[t] = forward - wraparound
        return naf


# --- delta cache --------------------------------------------------------------


def _compute_delta(a: npt.NDArray[np.int8], c: int, n: int) -> npt.NDArray[np.int32]:
    v = int(a[c])
    d = np.zeros(n, dtype=np.int32)
    d[1 : n - c] = -2 * v * a[c + 1 : n].astype(np.int32)
    d[n - c : n] = 2 * v * a[:c].astype(np.int32)
    d[1 : c + 1] += -2 * v * a[c - 1 :: -1].astype(np.int32)[:c]
    if c < n - 1:
        d[c + 1 : n] += 2 * v * a[: c - n : -1].astype(np.int32)[: n - c - 1]
    return d


def _build_delta_cache(seqs: npt.NDArray[np.int8], n: int) -> npt.NDArray[np.int32]:
    cache = np.empty((4 * n, n), dtype=np.int32)
    for s in range(4):
        a = seqs[s]
        for c in range(n):
            cache[s * n + c] = _compute_delta(a, c, n)
    return cache


def _rebuild_delta_slice(
    cache: npt.NDArray[np.int32], a: npt.NDArray[np.int8], s: int, n: int
) -> None:
    off = s * n
    for c in range(n):
        cache[off + c, :] = _compute_delta(a, c, n)
