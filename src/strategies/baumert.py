"""Baumert-Hall-Suche mit drei symmetrischen zirkulanten Sequenzen."""
from __future__ import annotations

import time

import numpy as np
from tqdm import tqdm

from constructions import autocorrelation_state, build_goethals_seidel, build_propus, symmetric_circulant
from gpu import check_orthogonality
from .base import SearchStrategy


class BaumertHallSearch(SearchStrategy):
    """Sucht Baumert-Hall-Arrays der Ordnung ``4t``.

    Zweck: Reduziert Baumert-Hall- und Propus-Kandidaten auf drei symmetrische zirkulante Sequenzen ``A``, ``B`` und ``C``.
    Mechanik: Flippt Halbsequenzeintraege und minimiert die gewichtete periodische Autokorrelationsenergie von ``A, B, C``.
    Grundlage: Die Nebenbedingung ``A A^T + 2 B B^T + C C^T = 4t I`` liefert mit der Baumert-Hall-Blockanordnung einen Hadamard-Kandidaten.
    Pipeline: Kann nur eine Pipeline eroeffnen, weil keine ``refine``-Methode existiert.
    Grenzen: Die Suche betrachtet ausschliesslich diese symmetrische Baumert-Hall-Teilfamilie; Energie null der Sequenzen wird abschliessend an der vollstaendigen Matrix geprueft.
    """

    ORDER = 668
    T = 167
    HALF = 84

    def __init__(self, *, ORDER: int = ORDER, T: int = T, HALF: int = HALF) -> None:
        if T % 2 == 0 or ORDER != 4 * T or HALF != (T + 1) // 2:
            raise ValueError(
                "Baumert-Hall requires odd T, ORDER = 4*T, and HALF = (T+1)//2")
        self.ORDER = ORDER
        self.T = T
        self.HALF = HALF

    @property
    def name(self) -> str:
        return "baumert_hall"

    def _bh_energy(self, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> int:
        """A A^T + 2 B B^T + C C^T off-diagonal squared sum."""
        total = autocorrelation_state(
            np.stack((a, b, c)), np.array((1, 2, 1), dtype=np.int64))
        independent = total[1:(self.T + 1) // 2]
        return int(np.dot(independent, independent))

    def _build(self, a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
        return build_goethals_seidel(a, b, c, b)

    def search(self, steps: int, seed: int) -> tuple[np.ndarray, dict[str, int], float]:
        t0 = time.perf_counter()
        rng = np.random.default_rng(seed)
        half = self.HALF

        cur = [rng.choice([-1, 1], size=half).astype(np.int8) for _ in range(3)]
        best_half = [s.copy() for s in cur]
        e = best_e = self._bh_energy(
            *[symmetric_circulant(s) for s in cur])
        accepted = best_at = 0

        def _best():
            seqs = [symmetric_circulant(s) for s in best_half]
            matrices = (
                self._build(seqs[0], seqs[1], seqs[2]),
                build_propus(seqs[0], seqs[1], seqs[2]),
            )
            candidates = ((matrix, check_orthogonality(matrix))
                          for matrix in matrices)
            return min(candidates, key=lambda candidate: candidate[1]["energy"])

        best_M, best_met = _best()
        if best_met["energy"] == 0:
            return best_M, best_met, 0.0

        print(f"  t={self.T}  vars={3*half}  energy_start={e}")
        pbar = tqdm(total=steps, desc="baumert_hall", unit="steps", ncols=100)
        for step in range(steps):
            mi = rng.integers(0, 3)
            pi = rng.integers(0, half)
            cur[mi][pi] *= -1
            ne = self._bh_energy(
                *[symmetric_circulant(s) for s in cur])
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
