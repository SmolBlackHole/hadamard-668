"""Deterministische Kandidatensuche mit primitiven GF(2)-LFSRs."""
from __future__ import annotations

import math
import time

import numpy as np

from gpu import check_orthogonality
from .base import SearchStrategy


def _multiply_mod(left: int, right: int, polynomial: int, degree: int) -> int:
    result = 0
    while right:
        if right & 1:
            result ^= left
        right >>= 1
        left <<= 1
        if left & (1 << degree):
            left ^= polynomial
    return result


def _power_mod(base: int, exponent: int, polynomial: int, degree: int) -> int:
    result = 1
    while exponent:
        if exponent & 1:
            result = _multiply_mod(result, base, polynomial, degree)
        base = _multiply_mod(base, base, polynomial, degree)
        exponent >>= 1
    return result


def _prime_factors(value: int) -> set[int]:
    factors: set[int] = set()
    divisor = 2
    while divisor * divisor <= value:
        while value % divisor == 0:
            factors.add(divisor)
            value //= divisor
        divisor += 1
    if value > 1:
        factors.add(value)
    return factors


class GoldSearch(SearchStrategy):
    """Enumeriert Matrizen aus primitiven binären LFSR-Sequenzen.

    Zweck: Prüft deterministische Shift-Matrizen aus endlichen Körpern als Kandidatenquelle.
    Mechanik: Findet primitive Polynome, erzeugt ihre Perioden und bewertet begrenzt viele Shifts.
    Grundlage: Ein primitives Polynom vom Grad ``d`` erzeugt eine LFSR-Periode der Länge ``2^d - 1``.
    Pipeline: Kann nur eine Pipeline eröffnen, weil keine ``refine``-Methode existiert.
    Grenzen: Kleine Kreuzkorrelationen oder ein primitives Polynom implizieren keine Hadamard-Matrix.
    """

    def __init__(self, order: int = SearchStrategy.ORDER, degree: int | None = None) -> None:
        if order < 2:
            raise ValueError("gold search requires an order of at least two")
        self.ORDER = order
        self.degree = degree or math.ceil(math.log2(order + 1))

    @property
    def name(self) -> str:
        return "gold"

    def _primitive_polynomials(self) -> list[int]:
        order = (1 << self.degree) - 1
        factors = _prime_factors(order)
        polynomials = []
        for lower in range(1, 1 << self.degree, 2):
            polynomial = (1 << self.degree) | lower
            if _power_mod(2, order, polynomial, self.degree) != 1:
                continue
            if all(_power_mod(2, order // factor, polynomial, self.degree) != 1
                   for factor in factors):
                polynomials.append(polynomial)
        return polynomials

    def _candidate(self, polynomial: int) -> np.ndarray:
        period = (1 << self.degree) - 1
        values = np.empty(period, dtype=np.int8)
        state = 1
        for index in range(period):
            values[index] = 1 if state & 1 else -1
            state = _multiply_mod(state, 2, polynomial, self.degree)
        sequence = np.resize(values, self.ORDER)
        return np.array([np.roll(sequence, shift) for shift in range(self.ORDER)], dtype=np.int8)

    def search(self, steps: int, seed: int) -> tuple[np.ndarray, dict[str, int], float]:
        del seed
        started = time.perf_counter()
        candidates = self._primitive_polynomials()[:max(1, steps)]
        best_matrix = self._candidate(candidates[0])
        best_metrics = check_orthogonality(best_matrix)
        for polynomial in candidates[1:]:
            matrix = self._candidate(polynomial)
            metrics = check_orthogonality(matrix)
            if metrics["energy"] < best_metrics["energy"]:
                best_matrix, best_metrics = matrix, metrics
            if best_metrics["energy"] == 0:
                break
        return best_matrix, best_metrics, time.perf_counter() - started
