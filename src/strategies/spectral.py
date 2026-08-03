"""Douglas-Rachford-Projektion im Fourier-Raum von vier zyklischen Folgen."""
from __future__ import annotations

import time

import numpy as np
from tqdm import tqdm

from constructions import build_goethals_seidel, periodic_autocorrelation_energy
from gpu import check_orthogonality
from .base import SearchStrategy


class SpectralSearch(SearchStrategy):
    """Projiziert vier Folgen zwischen Fourier- und Vorzeichenbedingung.

    Zweck: Sucht komplementäre Vier-Folgen-Kandidaten ohne kubische Matrixfaktorisierung.
    Mechanik: Wendet Douglas-Rachford auf die spektrale Leistungssphäre und den reellen Vorzeichenraum an.
    Grundlage: Pro Frequenz muss ``|Â|² + |B̂|² + |Ĉ|² + |D̂|² = 4K`` gelten.
    Pipeline: Kann nur eine Pipeline eröffnen, weil keine ``refine``-Methode existiert.
    Grenzen: Die abschließende Vorzeichenprojektion bleibt heuristisch und garantiert keine diskrete Lösung.
    """

    def __init__(self, inner_steps: int = 50, *, ORDER: int = 668) -> None:
        if ORDER < 4 or ORDER % 4:
            raise ValueError("spectral search requires an order divisible by four")
        if inner_steps < 1:
            raise ValueError("inner_steps must be positive")
        self.inner_steps = inner_steps
        self.ORDER = ORDER
        self.K = ORDER // 4

    @property
    def name(self) -> str:
        return "spectral"

    @staticmethod
    def _project_fourier(state: np.ndarray) -> np.ndarray:
        """Project each Fourier four-vector onto norm ``sqrt(4K)``."""
        if state.ndim != 2 or state.shape[0] != 4:
            raise ValueError("spectral state must have shape (4, K)")
        size = state.shape[1]
        spectrum = np.fft.fft(state, axis=1)
        norms = np.sqrt(np.sum(np.abs(spectrum) ** 2, axis=0))
        target = np.sqrt(4.0 * size)
        nonzero = norms > 1e-12
        spectrum[:, nonzero] *= target / norms[nonzero]
        spectrum[:, ~nonzero] = 0.0
        spectrum[0, ~nonzero] = target
        return np.fft.ifft(spectrum, axis=1).real.astype(np.float32)

    @staticmethod
    def _project_sign(state: np.ndarray) -> np.ndarray:
        return np.where(state >= 0, 1.0, -1.0).astype(np.float32)

    @classmethod
    def _douglas_rachford_step(cls, state: np.ndarray) -> np.ndarray:
        orthogonal = cls._project_fourier(state)
        reflected_orthogonal = 2.0 * orthogonal - state
        signed = cls._project_sign(reflected_orthogonal)
        reflected_sign = 2.0 * signed - reflected_orthogonal
        return (0.5 * (state + reflected_sign)).astype(np.float32)

    def search(self, steps: int, seed: int) -> tuple[np.ndarray, dict[str, int], float]:
        started = time.perf_counter()
        rng = np.random.default_rng(seed)
        state = rng.normal(size=(4, self.K)).astype(np.float32)
        best = self._project_sign(state).astype(np.int8)
        best_energy = periodic_autocorrelation_energy(tuple(best))
        best_at = 0

        with tqdm(total=steps, desc="spectral", unit="steps") as progress:
            for step in range(steps):
                for _ in range(self.inner_steps):
                    state = self._douglas_rachford_step(state)
                candidate = self._project_sign(
                    self._project_fourier(state)).astype(np.int8)
                energy = periodic_autocorrelation_energy(tuple(candidate))
                if energy < best_energy:
                    best, best_energy, best_at = candidate, energy, step
                    if energy == 0:
                        progress.update(1)
                        break
                progress.update(1)
                if step % 50 == 0:
                    progress.set_postfix(best=best_energy)

        matrix = build_goethals_seidel(*best)
        metrics = check_orthogonality(matrix)
        elapsed = time.perf_counter() - started
        print(
            f"  seed={seed} best_energy={best_energy} found@step={best_at} {elapsed:.1f}s")
        return matrix, metrics, elapsed
