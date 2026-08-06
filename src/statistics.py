"""Statistical tools for solver benchmarks — Wilson CI, Z-test, McNemar."""

from __future__ import annotations

import math


def fmt_e(e: int) -> str:
    """Compact energy formatting: 1234 -> '1.2k'."""
    if e == 0:
        return "0"
    if e >= 1_000_000:
        return f"{e / 1_000_000:.1f}M"
    if e >= 1000:
        return f"{e / 1000:.0f}k"
    return str(e)


def wilson_ci(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for binomial proportion (95 % CI)."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    den = 1 + z * z / n
    center = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, center - half), min(1.0, center + half)


def z_test(k1: int, n1: int, k2: int, n2: int) -> tuple[float, float]:
    """Two-sample Z-test for proportions. Returns (z, p-value)."""
    if n1 == 0 or n2 == 0:
        return 0.0, 1.0
    p1 = k1 / n1
    p2 = k2 / n2
    p_pool = (k1 + k2) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se if se > 0 else 0.0
    p_val = 2 * (1 - 0.5 * (1 + math.erf(abs(z) / math.sqrt(2))))
    return z, p_val


def mcnemar(a: int, b: int, c: int, d: int) -> tuple[float, float]:
    """McNemar test for paired nominal data.

    Contingency table:
        Config2+  Config2-
    C1+     a         b
    C1-     c         d

    Returns (chi2, p-value) with continuity correction.
    """
    if b + c == 0:
        return 0.0, 1.0
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    p_val = 1 - math.erf(math.sqrt(chi2 / 2))
    return chi2, p_val
