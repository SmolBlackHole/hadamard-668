"""Zellularautomat mit lokalen oder spektralen Update-Regeln."""
from __future__ import annotations

import math
import time

import numpy as np
from scipy.ndimage import convolve

from gpu import check_orthogonality
from .base import SearchStrategy


class CASearch(SearchStrategy):
    """Optimiert Update-Regeln eines zellulaeren Kandidatenmodells.

    Zweck: Sucht Vorzeichenmatrizen ueber lokale Faltungsregeln oder globale Frequenzfilter.
    Mechanik: Mutiert einen Kernel beziehungsweise Frequenzgewichte, entwickelt die Matrix und akzeptiert Energieverbesserungen oder temperaturabhaengige Verschlechterungen.
    Grundlage: Bewertet wird die Off-Diagonal-Energie von ``H H^T - n I``; der Spektralmodus ist ein FFT-Filter mit anschliessender Vorzeichenprojektion.
    Pipeline: ``refine`` kann jede gueltige Vorzeichenmatrix derselben Ordnung weiterverarbeiten.
    Grenzen: Weder Faltung noch FFT-Filter sind eine Orthogonalitaetsprojektion oder ein Existenzbeweis; der Default-Modus ist ``spectral``.
    """

    def __init__(self, order: int = SearchStrategy.ORDER, ca_steps: int = 1,
                 rule_seed: int = 42, kernel_size: int = 5, mode: str = "spectral") -> None:
        if order < 1 or ca_steps < 1:
            raise ValueError("cellular search requires a positive order and ca_steps")
        if mode not in ("local", "spectral"):
            raise ValueError("mode must be 'local' or 'spectral'")
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        self.ORDER = order
        self.ca_steps = ca_steps
        self.rule_seed = rule_seed
        self.kernel_size = kernel_size
        self.mode = mode

    @property
    def name(self) -> str:
        return f"ca_{self.mode}"

    # ---- spectral mode ----

    def _apply_spectral(self, matrix: np.ndarray, weights: np.ndarray) -> np.ndarray:
        """FFT → gewichten → IFFT → sign als globaler Filter.

        weights: (self.ORDER,) Array von Frequenz-Gewichten in [0, 1].
        Jede Frequenzkomponente wird mit ihrem Gewicht multipliziert.
        """
        transformed = np.fft.fft2(matrix.astype(np.float64))
        w = np.fft.fftshift(np.outer(weights, weights))
        transformed *= w
        result = np.fft.ifft2(transformed).real
        return np.sign(result).astype(np.int8)

    def _random_spectral_weights(self, rng: np.random.Generator) -> np.ndarray:
        """Zufaellige Frequenz-Gewichte mit festem erstem Eintrag."""
        w = rng.uniform(0.3, 1.0, size=self.ORDER)
        w[0] = 1.0
        return w

    def _mutate_spectral_weights(self, w: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Flippe ein Frequenz-Gewicht."""
        new = w.copy()
        idx = rng.integers(1, self.ORDER)
        new[idx] = rng.uniform(0.1, 1.0)
        return new

    # ---- local mode (5x5) ----

    @staticmethod
    def _apply_ca(matrix: np.ndarray, rule: np.ndarray) -> np.ndarray:
        values = convolve(matrix.astype(np.float64), rule, mode="wrap")
        return np.where(values > 0, 1, -1).astype(np.int8)

    def _random_rule(self, rng: np.random.Generator) -> np.ndarray:
        return rng.choice((-1.0, 0.0, 1.0), size=(self.kernel_size, self.kernel_size))

    @staticmethod
    def _mutate_rule(rule: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        k = rule.shape[0]
        candidate = rule.copy()
        r, c = rng.integers(0, k), rng.integers(0, k)
        choices = np.array((-1.0, 0.0, 1.0))
        candidate[r, c] = rng.choice(choices[choices != candidate[r, c]])
        return candidate

    # ---- shared optimisation ----

    def _evolve(self, matrix: np.ndarray, param: np.ndarray) -> np.ndarray:
        candidate = matrix
        for _ in range(self.ca_steps):
            if self.mode == "spectral":
                candidate = self._apply_spectral(candidate, param)
            else:
                candidate = self._apply_ca(candidate, param)
        return candidate

    def _optimize(self, matrix: np.ndarray, steps: int, seed: int) -> tuple[np.ndarray, dict, float]:
        started = time.perf_counter()
        rng = np.random.default_rng(seed + self.rule_seed)
        current = matrix.copy()
        best_metrics = check_orthogonality(current)
        best = current.copy()
        metrics = best_metrics

        if self.mode == "spectral":
            param = self._random_spectral_weights(rng)
            mutate = self._mutate_spectral_weights
        else:
            param = self._random_rule(rng)
            mutate = self._mutate_rule

        energy_scale = max(metrics["energy"], 1)
        accepted = 0

        for step in range(steps):
            candidate_param = mutate(param, rng)
            candidate = self._evolve(current, candidate_param)
            candidate_metrics = check_orthogonality(candidate)
            delta = candidate_metrics["energy"] - metrics["energy"]
            T = 1.0 * 0.001 ** (step / max(steps - 1, 1))

            if delta <= 0 or rng.random() < math.exp(-delta / (T * energy_scale)):
                current, metrics, param = candidate, candidate_metrics, candidate_param
                accepted += 1
                if candidate_metrics["energy"] < best_metrics["energy"]:
                    best, best_metrics = candidate.copy(), candidate_metrics
                    if best_metrics["energy"] == 0:
                        break

            if step % 200 == 0 and step > 0:
                print(f"  [{step:>6d}/{steps}]  energy={metrics['energy']}  best={best_metrics['energy']}  "
                      f"acc={accepted}  T={T:.4f}", flush=True)

        elapsed = time.perf_counter() - started
        print(f"  seed={seed}  best_energy={best_metrics['energy']}  accepted={accepted}  {elapsed:.1f}s")
        return best, best_metrics, elapsed

    def search(self, steps: int, seed: int) -> tuple[np.ndarray, dict[str, int], float]:
        rng = np.random.default_rng(seed)
        matrix = rng.choice((-1, 1), size=(self.ORDER, self.ORDER)).astype(np.int8)
        return self._optimize(matrix, steps, seed)

    def refine(self, matrix: np.ndarray, steps: int, seed: int) -> tuple[np.ndarray, dict[str, int], float]:
        if matrix.shape != (self.ORDER, self.ORDER) or not np.all(np.isin(matrix, (-1, 1))):
            raise ValueError(f"cellular needs a {self.ORDER}x{self.ORDER} sign matrix")
        return self._optimize(matrix, steps, seed)
