"""Zellularautomat auf vier zyklischen Folgen mit lokalen oder spektralen Regeln."""
from __future__ import annotations

import math
import time

import numpy as np
from scipy.ndimage import convolve1d

from constructions import build_goethals_seidel, periodic_autocorrelation_energy
from gpu import check_orthogonality
from .base import SearchStrategy


class CASearch(SearchStrategy):
    """Optimiert vier periodische Folgen mit zellulären Update-Regeln.

    Zweck: Erprobt lokale Faltungsregeln und globale Frequenzfilter im kompakten Goethals-Seidel-Raum.
    Mechanik: Mutiert eine periodische 1D-Regel oder RFFT-Gewichte und akzeptiert Kandidaten per Autokorrelationsenergie.
    Grundlage: Komplementäre Folgen haben für jeden Nichtnull-Shift eine verschwindende summierte Autokorrelation.
    Pipeline: ``refine`` akzeptiert ausschließlich exakt extrahierbare Goethals-Seidel-Matrizen.
    Grenzen: Die Regeln sind heuristische Mutationen und keine Projektion auf die Hadamard-Bedingung.
    """

    def __init__(
        self,
        order: int = SearchStrategy.ORDER,
        ca_steps: int = 1,
        rule_seed: int = 42,
        kernel_size: int = 5,
        mode: str = "spectral",
    ) -> None:
        if order < 4 or order % 4 or ca_steps < 1:
            raise ValueError(
                "cellular search requires an order divisible by four and positive ca_steps")
        if mode not in ("local", "spectral"):
            raise ValueError("mode must be 'local' or 'spectral'")
        if kernel_size % 2 == 0:
            raise ValueError("kernel_size must be odd")
        self.ORDER = order
        self.K = order // 4
        self.ca_steps = ca_steps
        self.rule_seed = rule_seed
        self.kernel_size = kernel_size
        self.mode = mode

    @property
    def name(self) -> str:
        return f"ca_{self.mode}"

    @staticmethod
    def _apply_spectral(sequences: np.ndarray, weights: np.ndarray) -> np.ndarray:
        spectrum = np.fft.rfft(sequences.astype(np.float32), axis=1)
        filtered = np.fft.irfft(
            spectrum * weights[None, :], n=sequences.shape[1], axis=1)
        return np.where(filtered >= 0, 1, -1).astype(np.int8)

    def _random_spectral_weights(self, rng: np.random.Generator) -> np.ndarray:
        weights = rng.uniform(0.3, 1.0, size=self.K // 2 + 1)
        weights[0] = 1.0
        return weights

    @staticmethod
    def _mutate_spectral_weights(
        weights: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        candidate = weights.copy()
        index = 0 if len(candidate) == 1 else int(rng.integers(1, len(candidate)))
        candidate[index] = rng.uniform(0.1, 1.0)
        return candidate

    @staticmethod
    def _apply_ca(sequences: np.ndarray, rule: np.ndarray) -> np.ndarray:
        values = convolve1d(
            sequences.astype(np.float32), rule, axis=1, mode="wrap")
        return np.where(values >= 0, 1, -1).astype(np.int8)

    def _random_rule(self, rng: np.random.Generator) -> np.ndarray:
        return rng.choice(
            (-1.0, 0.0, 1.0), size=self.kernel_size).astype(np.float32)

    @staticmethod
    def _mutate_rule(rule: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        candidate = rule.copy()
        index = int(rng.integers(0, len(candidate)))
        choices = np.array((-1.0, 0.0, 1.0), dtype=np.float32)
        candidate[index] = rng.choice(choices[choices != candidate[index]])
        return candidate

    def _evolve(self, sequences: np.ndarray, parameter: np.ndarray) -> np.ndarray:
        candidate = sequences
        for _ in range(self.ca_steps):
            candidate = (
                self._apply_spectral(candidate, parameter)
                if self.mode == "spectral"
                else self._apply_ca(candidate, parameter)
            )
        return candidate

    def _optimize(
        self,
        sequences: np.ndarray,
        steps: int,
        seed: int,
    ) -> tuple[np.ndarray, dict[str, int], float]:
        started = time.perf_counter()
        rng = np.random.default_rng(seed + self.rule_seed)
        current = np.asarray(sequences, dtype=np.int8).copy()
        energy = best_energy = periodic_autocorrelation_energy(tuple(current))
        best = current.copy()

        if self.mode == "spectral":
            parameter = self._random_spectral_weights(rng)
            mutate = self._mutate_spectral_weights
        else:
            parameter = self._random_rule(rng)
            mutate = self._mutate_rule

        energy_scale = max(energy, 1)
        accepted = 0
        for step in range(steps):
            candidate_parameter = mutate(parameter, rng)
            candidate = self._evolve(current, candidate_parameter)
            candidate_energy = periodic_autocorrelation_energy(tuple(candidate))
            delta = candidate_energy - energy
            temperature = 0.001 ** (step / max(steps - 1, 1))
            if delta <= 0 or rng.random() < math.exp(
                -delta / (temperature * energy_scale)
            ):
                current, energy = candidate, candidate_energy
                parameter = candidate_parameter
                accepted += 1
                if energy < best_energy:
                    best, best_energy = current.copy(), energy
                    if energy == 0:
                        break

        matrix = build_goethals_seidel(*best)
        metrics = check_orthogonality(matrix)
        elapsed = time.perf_counter() - started
        print(
            f"  seed={seed} best_energy={best_energy} accepted={accepted} {elapsed:.1f}s")
        return matrix, metrics, elapsed

    def _extract_sequences(self, matrix: np.ndarray) -> np.ndarray:
        if matrix.shape != (self.ORDER, self.ORDER) or not np.all(
            np.isin(matrix, (-1, 1))
        ):
            raise ValueError(
                f"cellular needs a {self.ORDER}x{self.ORDER} sign matrix")
        first_row = matrix[0]
        sequences = np.stack((
            first_row[:self.K],
            first_row[self.K:2 * self.K][::-1],
            first_row[2 * self.K:3 * self.K][::-1],
            first_row[3 * self.K:][::-1],
        )).astype(np.int8)
        if not np.array_equal(build_goethals_seidel(*sequences), matrix):
            raise ValueError("cellular refinement needs a Goethals-Seidel matrix")
        return sequences

    def search(self, steps: int, seed: int) -> tuple[np.ndarray, dict[str, int], float]:
        sequences = np.random.default_rng(seed).choice(
            (-1, 1), size=(4, self.K)).astype(np.int8)
        return self._optimize(sequences, steps, seed)

    def refine(self, matrix: np.ndarray, steps: int, seed: int) -> tuple[np.ndarray, dict[str, int], float]:
        return self._optimize(self._extract_sequences(matrix), steps, seed)
