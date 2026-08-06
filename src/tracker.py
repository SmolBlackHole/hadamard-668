"""Energy tracker: O(N^2) sparse delta-update per flip.

Generic — works for any build_fn(seqs) -> matrix.  All array dtypes are
explicit: seqs=int8, M=int8, G=float64, dM=int16.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt


class GramTracker:
    """Tracks M (N x N, int8) and G (N x N, float64) for a Hadamard construction."""

    def __init__(self, build_fn: Callable[[npt.NDArray[np.int8]], npt.NDArray[np.int8]]) -> None:
        self._fn = build_fn
        self._k: int = 0
        self._n: int = 0
        self._N: int = 0
        self.M: npt.NDArray[np.int8] | None = None
        self.G: npt.NDArray[np.float64] | None = None
        self._e: int = 0

    def _gram_energy(self, M: npt.NDArray[np.int8]) -> tuple[npt.NDArray[np.float64], int]:
        G32 = M.astype(np.float32) @ M.astype(np.float32).T
        G64 = G32.astype(np.float64)
        np.fill_diagonal(G64, 0)
        return G64, int(float(np.sum(G64 * G64)))

    def build(self, seqs: npt.NDArray[np.int8]) -> None:
        self._k, self._n = seqs.shape
        self.M = self._fn(seqs)
        self._N = self.M.shape[0]
        self.G, self._e = self._gram_energy(self.M)

    def energy(self) -> int:
        return self._e

    # ── O(N^2) flip via sparse delta ──

    def _delta_energy(self, M_new: npt.NDArray[np.int8]) -> int:
        assert self.M is not None and self.G is not None
        dM = M_new.astype(np.int16) - self.M.astype(np.int16)
        nz = np.where(dM != 0)
        if len(nz[0]) == 0:
            return self._e

        rn = nz[0]
        cn = nz[1]
        vn = dM[rn, cn].astype(np.float64)
        Mf = self.M.astype(np.float64)
        N = self._N
        dG = np.zeros((N, N), dtype=np.float64)

        for i in range(len(rn)):
            r, c, v = int(rn[i]), int(cn[i]), float(vn[i])
            if v == 0.0:
                continue
            dG[:, r] += Mf[:, c] * v
            dG[r, :] += Mf[:, c] * v

        for a in range(len(rn)):
            ca = cn[a]
            for b in range(a, len(rn)):
                if cn[b] == ca:
                    p = vn[a] * vn[b]
                    ra, rb = int(rn[a]), int(rn[b])
                    dG[ra, rb] += p
                    if ra != rb:
                        dG[rb, ra] += p

        Gn = self.G + dG
        np.fill_diagonal(Gn, 0)
        return int(float(np.sum(Gn.astype(np.float64) ** 2)))

    # ── flip testing (restores state) ──

    def flip(self, seqs: npt.NDArray[np.int8], s: int, c: int) -> int:
        return self.flip_batch(seqs, [(s, c)])

    def flip_batch(self, seqs: npt.NDArray[np.int8], flips: list[tuple[int, int]]) -> int:
        for s, c in flips:
            seqs[s, c] *= -1
        try:
            return self._delta_energy(self._fn(seqs))
        finally:
            for s, c in reversed(flips):
                seqs[s, c] *= -1

    # ── accept (commits state) ──

    def accept(self, seqs: npt.NDArray[np.int8], s: int, c: int) -> int:
        return self.accept_many(seqs, [(s, c)])

    def accept_many(self, seqs: npt.NDArray[np.int8], flips: list[tuple[int, int]]) -> int:
        for s, c in flips:
            seqs[s, c] *= -1
        self.build(seqs)
        return self._e
