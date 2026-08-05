"""Iterated local search — singles scan + 2/3-bit rescue + kick, energy-budgeted."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def search(
    seqs: np.ndarray,
    energy_fn: Callable[[np.ndarray], int | float],
    rng: np.random.Generator,
    *,
    budget: int | None = None,
) -> tuple[np.ndarray, int]:
    """Iterated local search. Stops when all phases fail to improve, or budget exhausted."""
    n_seqs, n_cols = seqs.shape
    positions = [(s, c) for s in range(n_seqs) for c in range(n_cols)]
    B = len(positions)

    if budget is None:
        budget = B * 200

    best_seq = seqs.copy()
    best_e = int(energy_fn(seqs))
    budget -= 1

    while budget > 0 and best_e > 0:
        singles_e = np.full(B, np.inf)
        improved = False
        for idx, (s, c) in enumerate(positions):
            if budget <= 0:
                break
            best_seq[s, c] *= -1
            e = int(energy_fn(best_seq))
            budget -= 1
            singles_e[idx] = e
            if e < best_e:
                best_e = e
                improved = True
                break
            best_seq[s, c] *= -1

        if improved:
            continue

        K = min(B, max(12, B // 4))
        top = np.argsort(singles_e)[:K]
        for i in range(K):
            si, ci = positions[top[i]]
            best_seq[si, ci] *= -1
            for j in range(i + 1, K):
                if budget <= 0:
                    break
                sj, cj = positions[top[j]]
                best_seq[sj, cj] *= -1
                e = int(energy_fn(best_seq))
                budget -= 1
                if e < best_e:
                    best_e = e
                    improved = True
                    break
                best_seq[sj, cj] *= -1
            if improved:
                break
            best_seq[si, ci] *= -1

        if improved:
            continue

        if best_e > 0:
            K3 = min(K // 2, 10)
            for i in range(K3):
                si, ci = positions[top[i]]
                best_seq[si, ci] *= -1
                for j in range(i + 1, K3):
                    sj, cj = positions[top[j]]
                    best_seq[sj, cj] *= -1
                    for k in range(j + 1, K3):
                        if budget <= 0:
                            break
                        sk, ck = positions[top[k]]
                        best_seq[sk, ck] *= -1
                        e = int(energy_fn(best_seq))
                        budget -= 1
                        if e < best_e:
                            best_e = e
                            improved = True
                            break
                        best_seq[sk, ck] *= -1
                    if improved:
                        break
                    best_seq[sj, cj] *= -1
                if improved:
                    break
                best_seq[si, ci] *= -1

        if improved:
            continue

        if budget > 0 and best_e > 0:
            for s in range(n_seqs):
                c = int(rng.integers(0, n_cols))
                best_seq[s, c] *= -1

    return best_seq, best_e
