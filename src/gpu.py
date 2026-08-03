"""The shared GPU backend and Hadamard matrix metrics."""
from __future__ import annotations

import os
from typing import Any

import numpy as np

xp: Any = np


def _setup() -> None:
    """Use CuPy when its CUDA DLLs are available, otherwise keep NumPy."""
    global xp
    if os.environ.get("HADAMARD_BACKEND", "auto").lower() == "numpy":
        return
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


def to_numpy(values: Any) -> np.ndarray:
    """Return a host array without copying when the NumPy backend is active."""
    return values if isinstance(values, np.ndarray) else xp.asnumpy(values)


def _scalar(value: Any) -> int:
    return int(value.item())


def gram_backend() -> str:
    """Describe the exact Gram-matrix computation selected for this process."""
    return "numpy-float32-to-int32" if xp is np else "cupy-float32-to-int32"


def gram_matrix(matrix: Any, *, backend=None):
    """Return a zero-diagonal Gram matrix on the active or requested backend."""
    module = xp if backend is None else backend
    values = module.asarray(matrix, dtype=module.float32)
    gram = module.rint(values @ values.T).astype(module.int32)
    module.fill_diagonal(gram, 0)
    return gram


def metrics_from_gram(gram: Any) -> dict[str, int]:
    """Return exact off-diagonal metrics from a symmetric zero-diagonal Gram matrix."""
    module = np if isinstance(gram, np.ndarray) else xp
    size = gram.shape[0]
    pair_count = size * (size - 1) // 2
    if pair_count == 0:
        return {"energy": 0, "orthogonal_pairs": 0, "max_abs_correlation": 0}
    wide = gram.astype(module.int64, copy=False)
    energy = module.sum(wide * wide) // 2
    nonzero = module.count_nonzero(gram)
    orthogonal_pairs = pair_count - nonzero // 2
    maximum = module.abs(gram).max()
    return {
        "energy": _scalar(energy),
        "orthogonal_pairs": _scalar(orthogonal_pairs),
        "max_abs_correlation": _scalar(maximum),
    }


def correlation_histogram(gram: Any) -> dict[str, int]:
    """Count absolute off-diagonal correlations once per unordered row pair."""
    module = np if isinstance(gram, np.ndarray) else xp
    rows, columns = module.triu_indices(gram.shape[0], k=1)
    values = to_numpy(module.abs(gram[rows, columns]))
    magnitudes, counts = np.unique(values, return_counts=True)
    return {str(int(magnitude)): int(count)
            for magnitude, count in zip(magnitudes, counts)}


def check_orthogonality(matrix: np.ndarray) -> dict[str, int]:
    """Return exact off-diagonal Gram-matrix metrics for a sign matrix."""
    return metrics_from_gram(gram_matrix(matrix))


def entry_flip_deltas(
    matrix: Any,
    gram: Any,
    row: int,
    columns: Any,
) -> np.ndarray:
    """Return exact energy deltas for flipping ``matrix[row, columns]``."""
    module = np if isinstance(matrix, np.ndarray) else xp
    selected = module.asarray(columns, dtype=module.int64)
    if selected.ndim != 1:
        raise ValueError("columns must be one-dimensional")
    if selected.size == 0:
        return np.empty(0, dtype=np.int64)
    gram_row = gram[row].astype(module.int64, copy=False)
    column_values = matrix[:, selected].astype(module.int64, copy=False)
    row_values = matrix[row, selected].astype(module.int64, copy=False)
    projections = gram_row @ column_values
    deltas = 4 * (matrix.shape[0] - 1) - 4 * row_values * projections
    return np.asarray(to_numpy(deltas), dtype=np.int64)


def entry_flip_delta(matrix: Any, gram: Any, row: int, column: int) -> int:
    """Return the exact energy delta for one sign-entry flip."""
    return int(entry_flip_deltas(matrix, gram, row, [column])[0])


def apply_entry_flip(
    matrix: Any,
    gram: Any,
    row: int,
    column: int,
    *,
    known_delta: int | None = None,
) -> int:
    """Flip one sign entry and update the affected Gram row and column in O(n)."""
    delta = (entry_flip_delta(matrix, gram, row, column)
             if known_delta is None else known_delta)
    old_value = matrix[row, column]
    change = (-2 * old_value * matrix[:, column]).astype(
        gram.dtype, copy=False)
    change[row] = 0
    updated = gram[row] + change
    matrix[row, column] = -old_value
    gram[row, :] = updated
    gram[:, row] = updated
    gram[row, row] = 0
    return int(delta)
