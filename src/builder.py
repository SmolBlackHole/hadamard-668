"""Goethals-Seidel matrix construction from four binary sequences."""

from __future__ import annotations

from functools import lru_cache

import numpy as np
import numpy.typing as npt

from .models import Int8Array


@lru_cache(maxsize=8)
def _circulant_indices(n: int) -> npt.NDArray[np.int64]:
    return (np.arange(n)[None, :] - np.arange(n)[:, None]) % n


@lru_cache(maxsize=8)
def _negacirculant_signs(n: int) -> Int8Array:
    return np.where(np.arange(n)[None, :] >= np.arange(n)[:, None], np.int8(1), np.int8(-1))


def _negacirculant(values: Int8Array) -> Int8Array:
    circulant = values[_circulant_indices(values.size)]
    return _negacirculant_signs(values.size) * circulant


def build_gs4(sequences: Int8Array) -> Int8Array:
    if sequences.ndim != 2 or sequences.shape[0] != 4:
        raise ValueError("GS4 requires an array with shape (4, n)")
    a, b, c, d = (_negacirculant(sequence) for sequence in sequences)
    br, cr, dr = b[:, ::-1], c[:, ::-1], d[:, ::-1]
    btr, ctr, dtr = b.T[:, ::-1], c.T[:, ::-1], d.T[:, ::-1]
    return np.block(
        [
            [a, br, cr, dr],
            [-br, a, -dtr, ctr],
            [-cr, dtr, a, -btr],
            [-dr, -ctr, btr, a],
        ]
    )
