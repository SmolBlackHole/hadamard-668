"""FFT-Gradient guided search on circulant sequences."""
from __future__ import annotations

import time

import numpy as np
from tqdm import tqdm

from builders import build_goethals_seidel
from correlations import (
    apply_symmetric_flip,
    autocorrelation_state,
    correlation_energy,
    expand_symmetric_sequence,
)
from fourier import half_sequence_gradient, project_power_complementarity
from gpu import check_orthogonality
from .base import SearchStrategy


class PocsSearch(SearchStrategy):
    """FFT-Gradient-guided bit selection.

    Zweck: Priorisiert lokale Flips in vier symmetrischen Folgen mit einem kontinuierlichen FFT-Gradienten.
    Mechanik: Startet mit einer kurzen Fourier-Projektion, bewertet Gradientenpositionen und fällt bei ausbleibender Verbesserung auf einen Zufallsflip zurück.
    Grundlage: Der Gradient der periodischen Autokorrelationsenergie schätzt den Einfluss jedes diskreten Vorzeichenwechsels.
    Pipeline: Kann nur eine Pipeline eröffnen, weil keine ``refine``-Methode existiert.
    Grenzen: Die Gradientenrangfolge ist ein Proxy; nur die exakte kompakte Energie und die finale Gram-Metrik entscheiden.
    """

    ORDER = 668
    K = 167
    HALF = 84

    def __init__(self, projection_steps: int = 1, *, ORDER: int = ORDER, K: int = K, HALF: int = HALF) -> None:
        if projection_steps < 0:
            raise ValueError("projection_steps must be non-negative")
        self.projection_steps = projection_steps
        self.ORDER = ORDER
        self.K = K
        self.HALF = HALF

    @property
    def name(self) -> str:
        return "pocs"

    @staticmethod
    def _project_seed(half_sequences: list[np.ndarray], steps: int) -> list[np.ndarray]:
        full = np.stack(
            [expand_symmetric_sequence(half) for half in half_sequences],
        ).astype(np.float64)
        for _ in range(steps):
            full = np.where(full >= 0, 1.0, -1.0)
            full = project_power_complementarity(full)
        return [np.where(sequence >= 0, 1, -1).astype(np.int8)
                for sequence in full]

    def search(self, steps: int, seed: int) -> tuple[np.ndarray, dict[str, int], float]:
        t0 = time.perf_counter()
        rng = np.random.default_rng(seed)
        half = self.HALF

        current = [rng.choice([-1, 1], size=half).astype(np.int8) for _ in range(4)]
        if self.projection_steps:
            projected = self._project_seed(current, self.projection_steps)
            current = [sequence[:half].copy() for sequence in projected]
        sequences = np.stack([expand_symmetric_sequence(value) for value in current])
        correlations = autocorrelation_state(sequences)
        best_half = [c.copy() for c in current]
        e = best_e = correlation_energy(correlations) // 2
        accepted = best_at = 0

        def matrices():
            full = [expand_symmetric_sequence(h) for h in best_half]
            matrix = build_goethals_seidel(*full)
            metrics = check_orthogonality(matrix)
            return matrix, metrics, metrics["energy"] == 0

        best_M, best_met, found = matrices()
        if found:
            return best_M, best_met, 0.0

        print(f"  sub_order={self.K}  vars={4*half}  energy_start={e}")
        pbar = tqdm(total=steps, desc="pocs", unit="steps", ncols=100)
        for step in range(steps):
            pbar.update(1)
            # Every 7 steps: FFT gradient, try top-5 (tuned: 5 seeds x 20k steps)
            if step % 7 == 0:
                grad = half_sequence_gradient(current)
                sign = np.array([c.astype(np.float64) for c in current])
                flip_score = -2.0 * sign * grad
                flat = flip_score.ravel()
                count = min(5, flat.size)
                idx = np.argpartition(flat, count - 1)[:count]
                idx = idx[np.argsort(flat[idx])]

                improved = False
                for k in idx:
                    si, pi = divmod(k, half)
                    current[si][pi] *= -1
                    ne = apply_symmetric_flip(
                        sequences, correlations, si, pi) // 2
                    if ne < e:
                        e = ne
                        accepted += 1
                        improved = True
                        if ne < best_e:
                            best_e = ne
                            best_half = [c.copy() for c in current]
                            best_at = step
                            if ne == 0:
                                break
                            if step % 50 == 0:
                                best_M, best_met, found = matrices()
                                if found:
                                    break
                        break
                    current[si][pi] *= -1
                    apply_symmetric_flip(sequences, correlations, si, pi)

                if improved:
                    if ne == 0:
                        break
                    continue

            # Random fallback
            mi = rng.integers(0, 4)
            pi = rng.integers(0, half)
            current[mi][pi] *= -1
            ne = apply_symmetric_flip(sequences, correlations, mi, pi) // 2
            if ne <= e:
                e, accepted = ne, accepted + 1
                if ne < best_e:
                    best_e = ne
                    best_half = [c.copy() for c in current]
                    best_at = step
                    if ne == 0:
                        break
                    if step % 50 == 0:
                        best_M, best_met, found = matrices()
                        if found:
                            break
            else:
                current[mi][pi] *= -1
                apply_symmetric_flip(sequences, correlations, mi, pi)

            if step % 50 == 0:
                pbar.set_postfix(e=e, best=best_e, acc=accepted)
        pbar.set_postfix(e=e, best=best_e, acc=accepted)
        pbar.close()

        best_M, best_met, _ = matrices()
        elapsed = time.perf_counter() - t0
        print(f"  seed={seed}  best_energy={best_e}  found@step={best_at}  accepted={accepted}  {elapsed:.1f}s")
        return best_M, best_met, elapsed
