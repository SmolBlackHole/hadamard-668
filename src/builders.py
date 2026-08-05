"""Goethals-Seidel: circulant, group-circulant, and negacyclic variants."""

from __future__ import annotations

import numpy as np


def _gs4_block(A, B, C, D):
    BR, CR, DR = B[:, ::-1], C[:, ::-1], D[:, ::-1]
    BtR, CtR, DtR = B.T[:, ::-1], C.T[:, ::-1], D.T[:, ::-1]
    return np.block(
        [
            [A, BR, CR, DR],
            [-BR, A, -DtR, CtR],
            [-CR, DtR, A, -BtR],
            [-DR, -CtR, BtR, A],
        ]
    ).astype(np.int8)


def _circulant(values):
    values = np.asarray(values, dtype=np.int8)
    n = values.size
    return values[(np.arange(n)[None, :] - np.arange(n)[:, None]) % n]


def _negacirculant(values):
    values = np.asarray(values, dtype=np.int8)
    n = values.size
    idx = (np.arange(n)[None, :] - np.arange(n)[:, None]) % n
    signs = np.where(np.arange(n)[None, :] >= np.arange(n)[:, None], 1, -1)
    return (signs * values[idx]).astype(np.int8)


_DIFF_CACHE: dict[tuple[int, ...], np.ndarray] = {}


def _diff_table(dims):
    if dims not in _DIFF_CACHE:
        n = int(np.prod(dims))
        table = np.zeros((n, n), dtype=np.int32)
        for i in range(n):
            rem = i
            gi = []
            for d in reversed(dims):
                gi.insert(0, rem % d)
                rem //= d
            for j in range(n):
                rem = j
                gj = []
                for d in reversed(dims):
                    gj.insert(0, rem % d)
                    rem //= d
                diff = [(b - a) % d for a, b, d in zip(gi, gj, dims)]
                idx = 0
                stride = 1
                for dv, ds in zip(reversed(diff), reversed(dims)):
                    idx += dv * stride
                    stride *= ds
                table[i, j] = idx
        _DIFF_CACHE[dims] = table
    return _DIFF_CACHE[dims]


def build_goethals_seidel(a, b, c, d):
    return _gs4_block(_circulant(a), _circulant(b), _circulant(c), _circulant(d))


def build_negacyclic_gs4(a, b, c, d):
    return _gs4_block(_negacirculant(a), _negacirculant(b), _negacirculant(c), _negacirculant(d))


def build_group_gs4(seqs, dims):
    n = int(np.prod(dims))
    dt = _diff_table(tuple(dims))

    def circ(flat):
        return flat[dt].astype(np.int8)

    return _gs4_block(circ(seqs[0, :n]), circ(seqs[1, :n]), circ(seqs[2, :n]), circ(seqs[3, :n]))
