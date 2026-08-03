"""Smoke tests for the Hadamard-668 search engine."""
import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "verifier"))

import numpy as np

from search import (
    ORDER, K, H,
    williamson_search,
    check_orthogonality,
    _build_williamson,
    _autocorrelation_energy,
    energy_off_diagonal,
)


def test_known_williamson_4x4():
    ones = np.array([1], dtype=np.int8)
    M = _build_williamson(ones, ones, ones, ones)
    assert M.shape == (4, 4)
    m = check_orthogonality(M)
    assert m["energy"] == 0, f"energy={m['energy']}"
    assert m["orthogonal_pairs"] == 6
    print("  test_known_williamson_4x4  PASS")


def test_autocorrelation_zero():
    ones = np.array([1], dtype=np.int8)
    e = _autocorrelation_energy((ones, ones, ones, ones))
    assert e == 0, f"energy={e}"
    print("  test_autocorrelation_zero  PASS")


def test_energy_on_known():
    ones = np.array([1], dtype=np.int8)
    M = _build_williamson(ones, ones, ones, ones)
    assert energy_off_diagonal(M) == 0
    print("  test_energy_on_known  PASS")


def test_search_order4():
    import search as s
    o, k, h = s.ORDER, s.K, s.H
    s.ORDER, s.K, s.H = 4, 1, 1
    try:
        _, met = williamson_search(steps=100, seed=0)
        assert met["energy"] == 0, f"energy={met['energy']}"
        assert met["orthogonal_pairs"] == 6
        print("  test_search_order4  PASS")
    finally:
        s.ORDER, s.K, s.H = o, k, h


def test_constants():
    assert ORDER == 668 and K == 167 and H == 84
    print("  test_constants  PASS")


if __name__ == "__main__":
    print("Smoke tests ...\n")
    test_known_williamson_4x4()
    test_autocorrelation_zero()
    test_energy_on_known()
    test_search_order4()
    test_constants()
    print("\nAll PASS.")
