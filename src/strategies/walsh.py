"""Walsh-Mischheuristik mit Padding auf die nächste Zweierpotenz."""
from __future__ import annotations

import time

import numpy as np

from gpu import check_orthogonality, to_numpy, xp
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
    """Mischt Kandidaten über eine gepaddete Fast-Walsh-Hadamard-Transformation.

    Zweck: Liefert eine Spektralheuristik und für Zweierpotenzordnungen die Sylvester-Konstruktion.
    Mechanik: Transformiert, rundet im Walsh-Raum zurück und akzeptiert nur bessere Vorzeichenmatrizen.
    Grundlage: Die Fast-Walsh-Hadamard-Transformation ist exakt für Längen ``2^m``; bei 668 ist das 1024-Padding nur eine Heuristik.
    Pipeline: Kann nur eine Pipeline eröffnen, weil keine ``refine``-Methode existiert.
    Grenzen: Das Padding auf 1024 ist bei Ordnung 668 keine äquivalente Hadamard-Bedingung.
    """

    def __init__(self, order: int = SearchStrategy.ORDER) -> None:
        if order < 1:
            raise ValueError("walsh search requires a positive order")
        self.ORDER = order

    @property
    def name(self) -> str:
        return "walsh"

    def search(self, steps: int, seed: int) -> tuple[np.ndarray, dict[str, int], float]:
        started = time.perf_counter()
        size = _next_power_of_two(self.ORDER)
        if size == self.ORDER:
            matrix = _fwht(xp.eye(size, dtype=xp.float64))
            matrix = xp.where(matrix >= 0, 1, -1).astype(xp.int8)
            return to_numpy(matrix), check_orthogonality(matrix), time.perf_counter() - started
        rng = np.random.default_rng(seed)
        matrix = xp.asarray(rng.choice(
            (-1, 1), size=(self.ORDER, self.ORDER)), dtype=xp.int8)
        best = matrix.copy()
        metrics = best_metrics = check_orthogonality(matrix)
        for _ in range(steps):
            padded = xp.zeros((self.ORDER, size), dtype=xp.float64)
            padded[:, :self.ORDER] = matrix
            mixed = _fwht(padded) / np.sqrt(size)
            candidate = xp.where(_fwht(xp.sign(mixed))[
                                 :, :self.ORDER] >= 0, 1, -1).astype(xp.int8)
            candidate_metrics = check_orthogonality(candidate)
            if candidate_metrics["energy"] <= metrics["energy"]:
                matrix, metrics = candidate, candidate_metrics
                if metrics["energy"] == 0:
                    break
            if metrics["energy"] < best_metrics["energy"]:
                best = matrix.copy()
                best_metrics = metrics
        return to_numpy(best), best_metrics, time.perf_counter() - started
