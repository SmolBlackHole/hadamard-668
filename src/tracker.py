"""Energy tracker: O(N^2) sparse delta-update per flip.

Generic — works for any build_fn(seqs) -> matrix.  seqs is (k, n) int8.
"""

from __future__ import annotations

import numpy as np


class GramTracker:
    """Tracks M (N x N, int8) and G (float64) for a Hadamard construction."""

    def __init__(self, build_fn):
        self._fn = build_fn
        self._k = 0
        self._n = 0
        self._N = 0
        self.M = None
        self.G = None
        self._e = 0

    def _gram_energy(self, M):
        G32 = M.astype(np.float32) @ M.astype(np.float32).T
        G64 = G32.astype(np.float64)
        np.fill_diagonal(G64, 0)
        return G64, int(float(np.sum(G64 * G64)))

    def build(self, seqs):
        self._k, self._n = seqs.shape
        self.M = self._fn(seqs)
        self._N = self.M.shape[0]
        self.G, self._e = self._gram_energy(self.M)

    rebuild = build

    def energy(self):
        return self._e

    # ── O(N^2) flip via sparse delta ──

    def _delta_energy(self, M_new):
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

    def flip(self, seqs, s, c):
        return self.flip_batch(seqs, [(s, c)])

    def flip_batch(self, seqs, flips):
        for s, c in flips:
            seqs[s, c] *= -1
        try:
            return self._delta_energy(self._fn(seqs))
        finally:
            for s, c in reversed(flips):
                seqs[s, c] *= -1

    # ── accept (commits state) ──

    def accept(self, seqs, s, c):
        return self.accept_many(seqs, [(s, c)])

    def accept_many(self, seqs, flips):
        for s, c in flips:
            seqs[s, c] *= -1
        self.rebuild(seqs)
        return self._e
