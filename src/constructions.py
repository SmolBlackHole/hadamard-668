"""Exact sequence-level constructions for GS4 Hadamard matrices."""

from __future__ import annotations

from math import isqrt

import numpy as np

from .models import Int8Array

FieldElement = tuple[int, int]


def _is_prime(value: int) -> bool:
    if value < 2:
        return False
    if value % 2 == 0:
        return value == 2
    return all(value % divisor for divisor in range(3, isqrt(value) + 1, 2))


def _prime_factors(value: int) -> list[int]:
    factors: list[int] = []
    divisor = 2
    while divisor * divisor <= value:
        if value % divisor == 0:
            factors.append(divisor)
            while value % divisor == 0:
                value //= divisor
        divisor += 1 if divisor == 2 else 2
    if value > 1:
        factors.append(value)
    return factors


def supports_paley_ng(n: int) -> bool:
    """Return whether the implemented prime-field Paley/Ito case applies."""
    return n > 0 and n % 2 == 0 and _is_prime(2 * n - 1)


def paley_ng_sequences(n: int) -> Int8Array:
    """Construct the second Paley/Ito NG-pair, duplicated as a GS4 tuple."""
    if not supports_paley_ng(n):
        raise ValueError("Paley-NG requires even n with prime p=2n-1")

    prime = 2 * n - 1
    field_order = prime * prime
    factors = _prime_factors(field_order - 1)

    def multiply(left: FieldElement, right: FieldElement, c: int) -> FieldElement:
        a, b = left
        d, e = right
        return (
            (a * d - c * b * e) % prime,
            (a * e + b * d - b * e) % prime,
        )

    def power(base: FieldElement, exponent: int, c: int) -> FieldElement:
        result = (1, 0)
        while exponent:
            if exponent & 1:
                result = multiply(result, base, c)
            base = multiply(base, base, c)
            exponent >>= 1
        return result

    generator: FieldElement | None = None
    coefficient = 0
    for c in range(prime):
        discriminant = (1 - 4 * c) % prime
        if pow(discriminant, (prime - 1) // 2, prime) != prime - 1:
            continue
        candidate = (0, 1)
        if all(power(candidate, (field_order - 1) // factor, c) != (1, 0) for factor in factors):
            generator = candidate
            coefficient = c
            break
    if generator is None:
        raise RuntimeError(f"could not find a primitive GF({prime}^2) generator")

    residues = {value * value % prime for value in range(1, prime)}
    alpha = power(generator, n, coefficient)
    w = power(generator, n - 1, coefficient)

    def sign(exponent: int) -> np.int8:
        z = power(w, exponent, coefficient)
        trace_value, trace_x = multiply(alpha, z, coefficient)
        return np.int8(1 if (2 * trace_value - trace_x) % prime in residues else -1)

    a = np.fromiter((sign(2 * i) for i in range(n)), dtype=np.int8, count=n)
    b = np.fromiter((sign(2 * i + 1) for i in range(n)), dtype=np.int8, count=n)
    return np.stack((a, b, a, b))


def golay_pair_2() -> tuple[Int8Array, Int8Array]:
    """Return the canonical binary Golay pair of length two."""
    return np.asarray((1, -1), dtype=np.int8), np.asarray((1, 1), dtype=np.int8)


def turyn_pair(
    golay_a: Int8Array, golay_b: Int8Array, c: Int8Array, d: Int8Array
) -> tuple[Int8Array, Int8Array]:
    """Apply the Balonin-Djokovic Turyn product to one sequence pair."""
    if golay_a.ndim != 1 or golay_b.shape != golay_a.shape or c.ndim != 1 or d.shape != c.shape:
        raise ValueError("Turyn inputs must be equally sized one-dimensional pairs")

    left = golay_a.astype(np.int16, copy=False)
    right = golay_b.astype(np.int16, copy=False)
    plus = (left + right) // 2
    minus = (left - right) // 2
    c16 = c.astype(np.int16, copy=False)
    d16 = d.astype(np.int16, copy=False)
    e = plus[:, None] * c16 + minus[:, None] * d16[::-1]
    f = plus[:, None] * d16 - minus[:, None] * c16[::-1]
    if not np.all(np.abs(e) == 1) or not np.all(np.abs(f) == 1):
        raise ValueError("Turyn inputs did not produce binary sequences")
    return e.reshape(-1, order="F").astype(np.int8), f.reshape(-1, order="F").astype(np.int8)


def turyn_gs4(golay_a: Int8Array, golay_b: Int8Array, sequences: Int8Array) -> Int8Array:
    """Lift a GS4 tuple by applying one Turyn factor to both sequence pairs."""
    if sequences.ndim != 2 or sequences.shape[0] != 4:
        raise ValueError("Turyn GS4 input must have shape (4, n)")
    first = turyn_pair(golay_a, golay_b, sequences[0], sequences[1])
    second = turyn_pair(golay_a, golay_b, sequences[2], sequences[3])
    return np.stack((*first, *second))


def double_gs4(sequences: Int8Array) -> Int8Array:
    """Lift a GS4 tuple from length n to 2n using the Golay-2 factor."""
    return turyn_gs4(*golay_pair_2(), sequences)
