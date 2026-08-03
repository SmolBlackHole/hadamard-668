"""The shared GPU backend and Hadamard matrix metrics."""
from __future__ import annotations

import os
from typing import Any

import numpy as np

xp: Any = np


def _setup() -> None:
    """Use CuPy when its CUDA DLLs are available, otherwise keep NumPy."""
    global xp
    try:
        import glob
        import site
        import warnings

        os.environ.setdefault(
            "CUPY_CACHE_DIR", os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", ".cupy-cache")))
        nvidia = os.path.join(site.getsitepackages()[0], "nvidia")
        if os.path.isdir(nvidia):
            for directory in {os.path.dirname(path) for path in glob.glob(os.path.join(nvidia, "**", "*.dll"), recursive=True)}:
                os.environ["PATH"] = directory + \
                    ";" + os.environ.get("PATH", "")
            os.environ.setdefault("CUDA_PATH", nvidia)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import cupy as cp
        cp.dot(cp.array([1]), cp.array([1]))
        xp = cp
    except Exception:
        pass


_setup()


def check_orthogonality(matrix: np.ndarray) -> dict[str, int]:
    """Return off-diagonal Gram-matrix metrics for a sign matrix."""
    size = matrix.shape[0]
    pair_count = size * (size - 1) // 2
    if pair_count == 0:
        return {"energy": 0, "orthogonal_pairs": 0, "max_abs_correlation": 0}
    gram = xp.asarray(matrix, dtype=xp.int64)
    gram = gram @ gram.T
    upper = gram[xp.triu_indices(size, k=1)]
    energy = xp.dot(upper, upper)
    orthogonal_pairs = xp.count_nonzero(upper == 0)
    maximum = xp.abs(upper).max()
    if xp is not np:
        energy, orthogonal_pairs, maximum = energy.get(
        ), orthogonal_pairs.get(), maximum.get()
    return {
        "energy": int(energy),
        "orthogonal_pairs": int(orthogonal_pairs),
        "max_abs_correlation": int(maximum),
    }
