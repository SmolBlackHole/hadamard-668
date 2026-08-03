"""Walsh-Mischheuristik auf vier gepaddeten zyklischen Folgen."""
from __future__ import annotations

import time

import numpy as np

from constructions import build_goethals_seidel, periodic_autocorrelation_energy
from gpu import check_orthogonality
from .base import SearchStrategy


def _next_power_of_two(value: int) -> int:
    return 1 << (value - 1).bit_length()


def _fwht(values: np.ndarray) -> np.ndarray:
    transformed = values.copy()
    width = 1
    while width < transformed.shape[1]:
        blocks = transformed.reshape(-1, 2 * width)
        left, right = blocks[:, :width].copy(), blocks[:, width:].copy()
        blocks[:, :width], blocks[:, width:] = left + right, left - right
        width *= 2
    return transformed


class WalshSearch(SearchStrategy):
    """Mutiert vier Folgen global im gepaddeten Walsh-Raum.

    Zweck: Nutzt globale Walsh-Koeffizientenmutationen im kompakten Goethals-Seidel-Suchraum.
    Mechanik: Füllt jede Folge auf die nächste Zweierpotenz auf, flippt einen Koeffizienten und projiziert nach der Rücktransformation auf Vorzeichen.
    Grundlage: Zweifache FWHT ergibt bis auf den Längenfaktor die Ausgangsfolge; für Zweierpotenzordnungen bleibt die Sylvester-Konstruktion exakt.
    Pipeline: Kann nur eine Pipeline eröffnen, weil keine ``refine``-Methode existiert.
    Grenzen: Das Padding von 167 auf 256 ist eine Suchheuristik und keine äquivalente Hadamard-Bedingung.
    """

    def __init__(self, order: int = SearchStrategy.ORDER) -> None:
        if order < 1:
            raise ValueError("walsh search requires a positive order")
        if _next_power_of_two(order) != order and order % 4:
            raise ValueError(
                "non-power-of-two Walsh search requires an order divisible by four")
        self.ORDER = order
        self.K = order // 4 if order % 4 == 0 else 0

    @property
    def name(self) -> str:
        return "walsh"

    @staticmethod
    def _mutate_sequences(
        sequences: np.ndarray,
        size: int,
        rng: np.random.Generator,
    ) -> np.ndarray:
        padded = np.zeros((4, size), dtype=np.float32)
        padded[:, :sequences.shape[1]] = sequences
        transformed = _fwht(padded)
        sequence = int(rng.integers(0, 4))
        frequency = int(rng.integers(0, size))
        transformed[sequence, frequency] *= -1
        restored = _fwht(transformed)[:, :sequences.shape[1]] / size
        return np.where(restored >= 0, 1, -1).astype(np.int8)

    def search(self, steps: int, seed: int) -> tuple[np.ndarray, dict[str, int], float]:
        started = time.perf_counter()
        if _next_power_of_two(self.ORDER) == self.ORDER:
            matrix = _fwht(np.eye(self.ORDER, dtype=np.float32))
            matrix = np.where(matrix >= 0, 1, -1).astype(np.int8)
            return matrix, check_orthogonality(matrix), time.perf_counter() - started

        rng = np.random.default_rng(seed)
        size = _next_power_of_two(self.K)
        current = rng.choice((-1, 1), size=(4, self.K)).astype(np.int8)
        energy = best_energy = periodic_autocorrelation_energy(tuple(current))
        best = current.copy()
        for _ in range(steps):
            candidate = self._mutate_sequences(current, size, rng)
            candidate_energy = periodic_autocorrelation_energy(tuple(candidate))
            if candidate_energy <= energy:
                current, energy = candidate, candidate_energy
                if energy < best_energy:
                    best, best_energy = current.copy(), energy
                    if energy == 0:
                        break

        matrix = build_goethals_seidel(*best)
        return matrix, check_orthogonality(matrix), time.perf_counter() - started
