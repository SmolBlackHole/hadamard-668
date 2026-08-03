"""Spektrale Projektion zwischen orthogonalen und Vorzeichenmatrizen."""
from __future__ import annotations
import time

import numpy as np
from tqdm import tqdm

from gpu import check_orthogonality
from .base import SearchStrategy


class SpectralSearch(SearchStrategy):
    """Projiziert alternierend zwischen orthogonalen und Vorzeichenmatrizen.

    Zweck: Erprobt eine Douglas-Rachford-artige Spektralheuristik im vollen Matrixraum.
    Mechanik: Erzeugt per SVD eine skalierte orthogonale Projektion, reflektiert und rundet auf Vorzeichen.
    Grundlage: Gesucht ist der Schnitt von ``{Q | Q Qᵀ = nI}`` und ``{H | Hᵢⱼ ∈ {-1, 1}}``.
    Pipeline: Kann nur eine Pipeline eröffnen, weil keine ``refine``-Methode existiert.
    Grenzen: Die diskrete Sign-Projektion macht den Ablauf heuristisch und garantiert keine Konvergenz.
    """

    def __init__(self, inner_steps: int = 50, *, ORDER: int = 668) -> None:
        self.inner_steps = inner_steps
        self.ORDER = ORDER

    @property
    def name(self) -> str:
        return "spectral"

    def _project_orthogonal(self, M: np.ndarray) -> np.ndarray:
        """Projiziere M auf die naechste orthogonale Matrix (via SVD).

        Q = U V^T wobei U S V^T = M. Q^T Q = I.
        """
        U, _, Vt = np.linalg.svd(M.astype(np.float64), full_matrices=False)
        return (U @ Vt) * np.sqrt(self.ORDER)

    def _project_sign(self, M: np.ndarray) -> np.ndarray:
        """Runde auf naechste ±1-Matrix."""
        return np.sign(M).astype(np.int8)

    def search(self, steps: int, seed: int) -> tuple[np.ndarray, dict[str, int], float]:
        t0 = time.perf_counter()
        rng = np.random.default_rng(seed)
        n = self.ORDER

        # Start: zufaellige ±1 Matrix
        H = rng.choice([-1, 1], size=(n, n)).astype(np.int8).astype(np.float64)
        H[0] = 1.0  # normalisiert
        best_H = H.astype(np.int8)
        best_met = {"energy": 2**63}
        best_at = 0

        total = n * (n - 1) // 2

        pbar = tqdm(total=steps, desc="spectral", unit="steps")
        for step in range(steps):
            # Douglas-Rachford: z = H, dann inner loop
            z = H.copy()
            for _ in range(self.inner_steps):
                # Projektion auf orthogonalen Raum
                q = self._project_orthogonal(z)
                # Reflexion
                z = 2 * q - z
                # Projektion auf ±1
                p = self._project_sign(z)
                # Reflexion
                z = 2 * p - z

            # Naechste Iteration startet von der Sign-Projektion
            H_next = self._project_sign(z)
            met = check_orthogonality(H_next)

            if met["energy"] < best_met["energy"]:
                best_met = met
                best_H = H_next.copy()
                best_at = step
                if met["energy"] == 0:
                    break

            H = H_next

            if step % 5 == 0:
                pbar.set_postfix(e=met["energy"], best=best_met["energy"],
                                 orth=f"{best_met.get('orthogonal_pairs', 0)/total*100:.0f}%")
                pbar.update(5)

        pbar.close()
        elapsed = time.perf_counter() - t0
        print(
            f"  seed={seed}  best_energy={best_met['energy']}  found@step={best_at}  {elapsed:.1f}s")
        return best_H.astype(np.int8), best_met, elapsed
