"""Hadamard matrix metrics and orthogonality checks."""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass
from typing import Any, cast

import numpy as np


@dataclass(frozen=True)
class Metrics:
    energy: int
    orthogonal_pairs: int
    max_abs_correlation: int


xp: Any = np


def _setup() -> None:
    global xp
    if os.environ.get("HADAMARD_BACKEND", "auto").lower() == "numpy":
        return
    try:
        import glob
        import site
        import warnings

        os.environ.setdefault(
            "CUPY_CACHE_DIR",
            os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".cupy-cache")),
        )

        nvidia_root = None
        for sp in site.getsitepackages():
            candidate = os.path.join(sp, "nvidia")
            if os.path.isdir(candidate):
                nvidia_root = candidate
                break

        if nvidia_root is not None:
            for directory in sorted(
                {
                    os.path.dirname(p)
                    for p in glob.glob(os.path.join(nvidia_root, "**", "*.dll"), recursive=True)
                }
            ):
                with contextlib.suppress(OSError):
                    os.add_dll_directory(directory)
            os.environ["CUDA_PATH"] = nvidia_root

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import cupy as cp  # pyright: ignore[reportMissingImports]

        backend: Any = cp
        _t: Any = backend.array([1.0, 2.0])
        _t = _t * _t
        _t = backend.fft.fft(_t)
        xp = backend
    except Exception:
        pass


_setup()


def _scalar(value: Any) -> int:
    return int(value.item())


def gram_matrix(matrix: Any) -> Any:
    module: Any = xp
    values: Any = module.asarray(matrix, dtype=module.float32)
    gram: Any = module.rint(values @ values.T).astype(module.int32)
    module.fill_diagonal(gram, 0)
    return gram


def metrics_from_gram(gram: Any) -> Metrics:
    module: Any = np if isinstance(gram, np.ndarray) else xp
    array: Any = cast(Any, gram)
    size = int(array.shape[0])
    pair_count: int = size * (size - 1) // 2
    if pair_count == 0:
        return Metrics(0, 0, 0)
    wide: Any = array.astype(module.int64, copy=False)
    energy: Any = module.sum(wide * wide) // 2
    nonzero: Any = module.count_nonzero(array)
    orthogonal_pairs: Any = pair_count - nonzero // 2
    maximum: Any = module.abs(array).max()
    return Metrics(_scalar(energy), _scalar(orthogonal_pairs), _scalar(maximum))


def check_orthogonality(matrix: np.ndarray) -> Metrics:
    return metrics_from_gram(gram_matrix(matrix))
