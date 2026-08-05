"""Energy functions for Hadamard search."""

from __future__ import annotations

import numpy as np
from numba import njit


@njit
def _gram_sq_energy(G):
    N = G.shape[0]
    e = np.int64(0)
    for i in range(N):
        for j in range(N):
            if i != j:
                v = G[i, j]
                e += np.int64(v) * v
    return e


@njit
def _off_diag_gram_energy(M):
    N = M.shape[0]
    e = np.int64(0)
    for i in range(N):
        for j in range(N):
            if i != j:
                dot = np.int32(0)
                for k in range(N):
                    dot += M[i, k] * M[j, k]
                e += np.int64(dot) * np.int64(dot)
    return e


def _build_to_energy(builder_fn, *args):
    M = builder_fn(*args)
    return int(_off_diag_gram_energy(M.astype(np.int32)))


def negacyclic_gs4_energy(seqs):
    from builders import build_negacyclic_gs4

    n = seqs.shape[1]
    return _build_to_energy(
        build_negacyclic_gs4,
        seqs[0, :n].copy(),
        seqs[1, :n].copy(),
        seqs[2, :n].copy(),
        seqs[3, :n].copy(),
    )


def cyclic_gs4_energy(seqs):
    from builders import build_goethals_seidel

    n = seqs.shape[1]
    return _build_to_energy(
        build_goethals_seidel,
        seqs[0, :n].copy(),
        seqs[1, :n].copy(),
        seqs[2, :n].copy(),
        seqs[3, :n].copy(),
    )


def group_gs4_energy(seqs, dims):
    from builders import build_group_gs4

    return _build_to_energy(build_group_gs4, seqs, dims)
