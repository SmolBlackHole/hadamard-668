"""Backend contracts for full-matrix GPU strategies."""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from strategies.ca import CASearch
from strategies.direct import DirectSearch
from strategies.ising import IsingSearch
from strategies.repair import RepairSearch
from strategies.spectral import SpectralSearch
from strategies.walsh import WalshSearch


@pytest.mark.parametrize("strategy", [
    RepairSearch(4), DirectSearch(4), IsingSearch(4),
    SpectralSearch(ORDER=4, inner_steps=1), CASearch(order=4), WalshSearch(4),
])
def test_full_matrix_strategies_return_host_sign_matrices(strategy) -> None:
    matrix, metrics, _ = strategy.search(steps=1, seed=1)
    assert isinstance(matrix, np.ndarray)
    assert matrix.shape == (4, 4)
    assert np.all(np.isin(matrix, (-1, 1)))
    assert isinstance(metrics["energy"], int)


def test_numpy_backend_can_be_forced_for_benchmarks() -> None:
    root = Path(__file__).parents[1]
    environment = os.environ | {"HADAMARD_BACKEND": "numpy"}
    result = subprocess.run(
        [sys.executable, "-c",
            "import sys; sys.path.insert(0, 'src'); from gpu import xp; print(xp.__name__)"],
        cwd=root, env=environment, capture_output=True, text=True, check=True,
    )
    assert result.stdout.strip() == "numpy"
