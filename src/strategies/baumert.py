"""Baumert-Hall-Suche mit drei symmetrischen zirkulanten Sequenzen."""
from __future__ import annotations

import time

import numpy as np
from tqdm import tqdm

from gpu import check_orthogonality
from .base import SearchStrategy


class BaumertHallSearch(SearchStrategy):
    """Sucht Baumert-Hall-Arrays der Ordnung ``4t``.

    Zweck: Reduziert die Suche auf drei symmetrische zirkulante Sequenzen ``A``, ``B`` und ``C``; ``D`` wird als ``B^T`` festgelegt.
    Mechanik: Flippt Halbsequenzeintraege und minimiert die gewichtete periodische Autokorrelationsenergie von ``A, B, C``.
    Grundlage: Die Nebenbedingung ``A A^T + 2 B B^T + C C^T = 4t I`` liefert mit der Baumert-Hall-Blockanordnung einen Hadamard-Kandidaten.
    Pipeline: Kann nur eine Pipeline eroeffnen, weil keine ``refine``-Methode existiert.
    Grenzen: Die Suche betrachtet ausschliesslich diese symmetrische Baumert-Hall-Teilfamilie; Energie null der Sequenzen wird abschliessend an der vollstaendigen Matrix geprueft.
    """

    ORDER = 668
    T = 167
    HALF = 84

    def __init__(self, *, ORDER: int = ORDER, T: int = T, HALF: int = HALF) -> None:
        if ORDER != 4 * T or HALF != (T + 1) // 2:
            raise ValueError("ORDER, T, HALF must satisfy ORDER = 4*T, HALF = (T+1)//2")
        self.ORDER = ORDER
        self.T = T
        self.HALF = HALF

    @property
    def name(self) -> str:
        return "baumert_hall"

    @staticmethod
    def _sym(half: np.ndarray) -> np.ndarray:
        n = 2 * len(half) - 1
        r = np.zeros(n, dtype=np.int8)
        r[:len(half)] = half
        r[len(half):] = half[n - len(half) - 1::-1]
        return r

    @staticmethod
    def _circulant(v: np.ndarray) -> np.ndarray:
        return np.array([np.roll(v, i) for i in range(len(v))], dtype=np.int8)

    def _bh_energy(self, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> int:
        """A A^T + 2 B B^T + C C^T off-diagonal squared sum."""
        t = self.T
        total = np.zeros(t, dtype=np.int64)
        for s, weight in [(a, 1), (b, 2), (c, 1)]:
            for d in range(t):
                total[d] += weight * int(np.dot(s, np.roll(s, -d)))
        return int(np.sum(total[1:(t + 1) // 2] ** 2))

    def _build(self, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
        t = self.T
        A = self._circulant(a)
        B = self._circulant(b)
        C = self._circulant(c)
        D = B.T
        R = np.fliplr(np.eye(t, dtype=np.int8))
        BR, CR, DR = B @ R, C @ R, D @ R
        return np.block([
            [A,    BR,   CR,   DR],
            [-BR,   A, -DR.T,  CR.T],
            [-CR, DR.T,    A, -BR.T],
            [-DR, -CR.T, BR.T,    A],
        ]).astype(np.int8)

    def search(self, steps: int, seed: int) -> tuple[np.ndarray, dict[str, int], float]:
        t0 = time.perf_counter()
        rng = np.random.default_rng(seed)
        half = self.HALF

        cur = [rng.choice([-1, 1], size=half).astype(np.int8) for _ in range(3)]
        best_half = [s.copy() for s in cur]
        e = best_e = self._bh_energy(
            *[self._sym(s) for s in cur])
        accepted = best_at = 0

        def _best():
            seqs = [self._sym(s) for s in best_half]
            M = self._build(seqs[0], seqs[1], seqs[2])
            return M, check_orthogonality(M)

        best_M, best_met = _best()
        if best_met["energy"] == 0:
            return best_M, best_met, 0.0

        print(f"  t={self.T}  vars={3*half}  energy_start={e}")
        pbar = tqdm(total=steps, desc="baumert_hall", unit="steps", ncols=100)
        for step in range(steps):
            mi = rng.integers(0, 3)
            pi = rng.integers(0, half)
            cur[mi][pi] *= -1
            ne = self._bh_energy(*[self._sym(s) for s in cur])
            if ne <= e:
                e, accepted = ne, accepted + 1
                if ne < best_e:
                    best_e = ne
                    best_half = [s.copy() for s in cur]
                    best_at = step
                    if ne == 0:
                        break
                    if step % 100 == 0:
                        best_M, best_met = _best()
                        if best_met["energy"] == 0:
                            break
            else:
                cur[mi][pi] *= -1
            if step % 50 == 0:
                pbar.set_postfix(e=e, best=best_e, acc=accepted)
                pbar.update(50)
        pbar.set_postfix(e=e, best=best_e, acc=accepted)
        pbar.close()

        best_M, best_met = _best()
        elapsed = time.perf_counter() - t0
        print(f"  seed={seed}  best_energy={best_e}  found@step={best_at}  accepted={accepted}  {elapsed:.1f}s")
        return best_M, best_met, elapsed
