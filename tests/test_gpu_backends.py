"""Backend contracts for full-matrix GPU strategies."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from gpu import gram_backend, gram_matrix, to_numpy, xp
from strategies.ising import IsingSearch
from strategies.repair import RepairSearch
from strategies.spectral import SpectralSearch


@pytest.mark.parametrize("strategy", [
    RepairSearch(4), IsingSearch(92), SpectralSearch(ORDER=92, inner_steps=1),
])
def test_full_matrix_strategies_return_host_sign_matrices(strategy) -> None:
    matrix, metrics, _ = strategy.search(steps=1, seed=1)
    assert isinstance(matrix, np.ndarray)
    assert matrix.shape == (strategy.ORDER, strategy.ORDER)
    assert matrix.dtype == np.int8
    assert np.all(np.isin(matrix, (-1, 1)))
    assert isinstance(metrics["energy"], int)


def test_numpy_backend_can_be_forced_for_benchmarks() -> None:
    root = Path(__file__).parents[1]
    environment = os.environ | {"HADAMARD_BACKEND": "numpy"}
    result = subprocess.run(
        [sys.executable, "-c",
            "import sys; sys.path.insert(0, 'src'); from gpu import gram_backend, xp; print(xp.__name__, gram_backend())"],
        cwd=root, env=environment, capture_output=True, text=True, check=True,
    )
    assert result.stdout.strip() == "numpy numpy-float32-to-int32"


def test_numpy_float32_gram_is_exact_at_target_order() -> None:
    matrix = np.random.default_rng(42).choice(
        [-1, 1], size=(668, 668)).astype(np.int8)
    expected = matrix.astype(np.int64) @ matrix.astype(np.int64).T
    np.fill_diagonal(expected, 0)
    assert np.array_equal(gram_matrix(matrix, backend=np), expected)


@pytest.mark.skipif(xp is np, reason="CuPy backend is not active")
def test_cupy_float32_gram_is_exact_at_target_order() -> None:
    matrix = np.random.default_rng(42).choice(
        [-1, 1], size=(668, 668)).astype(np.int8)
    expected = matrix.astype(np.int64) @ matrix.astype(np.int64).T
    np.fill_diagonal(expected, 0)
    assert gram_backend() == "cupy-float32-to-int32"
    assert np.array_equal(to_numpy(gram_matrix(matrix)), expected)
