"""Independent checks for bounded research diagnostics."""

from itertools import combinations
from math import gcd

import numpy as np
import pytest

from lab.neighborhoods import (
    decimate,
    extended_hash,
    main,
    repair_witnesses,
    repeated_row_triple,
    residual_rows,
)
from src.models import Int8Array
from src.verify import verify_candidate


@pytest.mark.parametrize("n", [3, 4, 5, 6, 7, 8, 11, 12])
def test_decimation_preserves_energy_and_generated_orbit(n: int) -> None:
    rng = np.random.default_rng(n)
    state = rng.choice(np.array([-1, 1], dtype=np.int8), size=(4, n))
    reference = residual_rows(state).sum(axis=0)
    identity = extended_hash(state)
    for k in range(1, 2 * n):
        if gcd(k, 2 * n) != 1:
            continue
        transformed = decimate(state, k)
        residual = residual_rows(transformed).sum(axis=0)
        assert residual @ residual == reference @ reference
        assert extended_hash(transformed) == identity
    transformed = state[[2, 0, 3, 1]].copy()
    transformed[0] = transformed[0, ::-1]
    transformed[1] *= -1
    transformed[2] = np.concatenate((-transformed[2, -2:], transformed[2, :-2]))
    assert extended_hash(transformed) == identity


def test_repair_search_matches_brute_force() -> None:
    rng = np.random.default_rng(2026)
    for _ in range(4):
        state: Int8Array = rng.choice(np.array([-1, 1], dtype=np.int8), size=(4, 3))
        actual = repair_witnesses(state)
        for family, size in {
            "single": 1,
            "pair": 2,
            "triple_distinct_rows": 3,
            "triple_all": 3,
            "quadruple_one_per_row": 4,
        }.items():
            found = False
            for positions in combinations(range(12), size):
                if (
                    family in ("triple_distinct_rows", "quadruple_one_per_row")
                    and len({p // 3 for p in positions}) != size
                ):
                    continue
                candidate = state.copy()
                for position in positions:
                    candidate.flat[position] *= -1
                if np.all(residual_rows(candidate).sum(axis=0) == 0):
                    verify_candidate(candidate)
                    found = True
                    break
            assert (actual[family] is not None) == found


def test_invalid_decimation() -> None:
    with pytest.raises(ValueError, match="unit"):
        decimate(np.ones((4, 6), dtype=np.int8), 3)


@pytest.mark.parametrize("n", [3, 4, 5])
def test_repeated_row_triples_match_brute_force(n: int) -> None:
    state = np.random.default_rng(n + 20).choice(np.array([-1, 1], dtype=np.int8), size=(4, n))
    original = state.copy()
    actual = repeated_row_triple(state)
    found = False
    for positions in combinations(range(4 * n), 3):
        if len({p // n for p in positions}) == 3:
            continue
        candidate = state.copy()
        for position in positions:
            candidate.flat[position] *= -1
        if np.all(residual_rows(candidate).sum(axis=0) == 0):
            found = True
            break
    assert (actual is not None) == found
    np.testing.assert_array_equal(state, original)
    if actual is not None:
        for row, column in actual:
            state[row, column] *= -1
        verify_candidate(state)


@pytest.mark.parametrize("first,last", [(0, 10), (10, 9)])
def test_invalid_run_range(first: int, last: int, capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit) as error:
        main(["--first-run-id", str(first), "--last-run-id", str(last)])
    assert error.value.code == 2
    assert "1 <= first-run-id <= last-run-id" in capsys.readouterr().err
