"""Tests against known Hadamard matrices and search engine verification.

Run from project root: PYTHONPATH="src" python tests/test_search.py
"""

import numpy as np

from search import (
    ORDER, K, HALF,
    _gram_off_diagonal_energy, check_orthogonality, _autocorrelation_energy,
    _symmetric_circulant, _build_williamson,
    williamson_search,
)
from verifier.known import sylvester, paley, get_known


def test_sym_circulant_order5():
    """Symmetric circulant: first half [1,-1,1] -> full row [1,-1,1,-1,1]."""
    half = np.array([1, -1, 1], dtype=np.int8)
    full = _symmetric_circulant(half)
    expected = np.array([1, -1, 1, -1, 1], dtype=np.int8)
    assert len(full) == 5
    assert np.array_equal(full, expected)
    print("  test_sym_circulant_order5  PASS")


def test_sylvester_order_4():
    H = sylvester(4)
    m = check_orthogonality(H)
    assert m["energy"] == 0, f"energy={m['energy']}"
    assert m["orthogonal_pairs"] == 6
    assert m["max_abs_correlation"] == 0
    print("  test_sylvester_order_4  PASS")


def test_sylvester_order_8():
    H = sylvester(8)
    m = check_orthogonality(H)
    assert m["energy"] == 0, f"energy={m['energy']}"
    print("  test_sylvester_order_8  PASS")


def test_paley_order_8():
    H = paley(8)
    m = check_orthogonality(H)
    assert m["energy"] == 0, f"energy={m['energy']}"
    print("  test_paley_order_8  PASS")


def test_paley_order_12():
    H = paley(12)
    m = check_orthogonality(H)
    assert m["energy"] == 0, f"energy={m['energy']}"
    print("  test_paley_order_12  PASS")


def test_get_known():
    for order in [1, 2, 4, 8, 12, 16, 20]:
        H = get_known(order)
        assert H.shape == (order, order), f"order={order} shape={H.shape}"
        m = check_orthogonality(H)
        assert m["energy"] == 0, f"order={order} energy={m['energy']}"
    print("  test_get_known[1,2,4,8,12,16,20]  PASS")


def test_williamson_builds_known_order4():
    ones = np.array([1], dtype=np.int8)
    H = _build_williamson(ones, ones, ones, ones)
    m = check_orthogonality(H)
    assert m["energy"] == 0
    assert m["orthogonal_pairs"] == 6
    print("  test_williamson_builds_known_order4  PASS")


def test_autocorrelation_energy_zero_for_ones():
    ones = np.array([1], dtype=np.int8)
    e = _autocorrelation_energy((ones, ones, ones, ones))
    assert e == 0
    print("  test_autocorrelation_energy_zero_for_ones  PASS")


def test_gram_energy_on_sylvester():
    H = sylvester(4)
    assert _gram_off_diagonal_energy(H) == 0
    print("  test_gram_energy_on_sylvester  PASS")


def test_williamson_search_order4():
    import search as s
    saved = s.ORDER, s.K, s.HALF
    s.ORDER, s.K, s.HALF = 4, 1, 1
    try:
        _, met = williamson_search(steps=100, seed=0)
        assert met["energy"] == 0, f"energy={met['energy']}"
        assert met["orthogonal_pairs"] == 6
        print("  test_williamson_search_order4  PASS")
    finally:
        s.ORDER, s.K, s.HALF = saved


def test_constants():
    assert ORDER == 668 and K == 167 and HALF == 84
    print("  test_constants  PASS")


if __name__ == "__main__":
    print("Tests ...\n")
    test_sym_circulant_order5()
    test_sylvester_order_4()
    test_sylvester_order_8()
    test_paley_order_8()
    test_paley_order_12()
    test_get_known()
    test_williamson_builds_known_order4()
    test_autocorrelation_energy_zero_for_ones()
    test_gram_energy_on_sylvester()
    test_williamson_search_order4()
    test_constants()
    print("\nAll PASS.")
