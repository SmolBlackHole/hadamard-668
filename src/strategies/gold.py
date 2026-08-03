"""Deterministische Kandidatensuche mit primitiven GF(2)-LFSRs."""
from __future__ import annotations

import math
import time

import numpy as np

from constructions import build_goethals_seidel, periodic_autocorrelation_energy
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
    """Erzeugt deterministische Vier-Folgen-Kandidaten aus binären LFSRs.

    Zweck: Nutzt Gold-artige Kombinationen primitiver LFSR-Folgen als kompakte GS-Seeder.
    Mechanik: Kombiniert zwei m-Sequenzen mit deterministischen relativen Shifts und bewertet nur ihre Autokorrelation.
    Grundlage: Ein primitives Polynom vom Grad ``d`` erzeugt eine Periode der Länge ``2^d - 1``; Produkte entsprechen XOR-Kombinationen.
    Pipeline: Kann nur eine Pipeline eröffnen, weil keine ``refine``-Methode existiert.
    Grenzen: Günstige Gold-Kreuzkorrelationen implizieren keine komplementäre Vier-Folgen-Lösung.
    """

    def __init__(self, order: int = SearchStrategy.ORDER, degree: int | None = None) -> None:
        if order < 4 or order % 4:
            raise ValueError("gold search requires an order divisible by four")
        self.ORDER = order
        self.K = order // 4
        self.degree = degree or math.ceil(math.log2(self.K + 1))

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

    def _lfsr_sequence(self, polynomial: int) -> np.ndarray:
        period = (1 << self.degree) - 1
        values = np.empty(period, dtype=np.int8)
        state = 1
        for index in range(period):
            values[index] = 1 if state & 1 else -1
            state = _multiply_mod(state, 2, polynomial, self.degree)
        return np.resize(values, self.K)

    def _candidate_sequences(
        self,
        first_polynomial: int,
        second_polynomial: int,
        shift: int,
    ) -> np.ndarray:
        first = self._lfsr_sequence(first_polynomial)
        second = self._lfsr_sequence(second_polynomial)
        return np.stack((
            first,
            second,
            first * np.roll(second, shift),
            first * np.roll(second, shift + 1),
        )).astype(np.int8)

    def search(self, steps: int, seed: int) -> tuple[np.ndarray, dict[str, int], float]:
        del seed
        started = time.perf_counter()
        polynomials = self._primitive_polynomials()
        if not polynomials:
            raise RuntimeError("no primitive polynomial found")
        pairs = [(first, second) for index, first in enumerate(polynomials)
                 for second in polynomials[index + 1:]]
        if not pairs:
            pairs = [(polynomials[0], polynomials[0])]

        limit = max(1, steps)
        best: np.ndarray | None = None
        best_energy = 2**63
        evaluated = 0
        for first, second in pairs:
            for shift in range(1, max(2, self.K)):
                candidate = self._candidate_sequences(first, second, shift)
                energy = periodic_autocorrelation_energy(tuple(candidate))
                evaluated += 1
                if energy < best_energy:
                    best, best_energy = candidate, energy
                    if energy == 0:
                        break
                if evaluated >= limit:
                    break
            if best_energy == 0 or evaluated >= limit:
                break

        assert best is not None
        matrix = build_goethals_seidel(*best)
        return matrix, check_orthogonality(matrix), time.perf_counter() - started
