"""Iterated local search — singles scan + 2/3-bit rescue + kick, energy-budgeted."""

from __future__ import annotations

import numpy as np


def _update_best(cur_seq, cur_e, best_seq, best_e):
    if cur_e < best_e:
        return cur_seq.copy(), cur_e
    return best_seq, best_e


def search(
    seqs: np.ndarray,
    tracker,
    rng: np.random.Generator,
    *,
    budget: int | None = None,
) -> tuple[np.ndarray, int]:
    """Iterated local search with EnergyTracker. Returns (best_seq, best_energy).

    The tracker holds M and G state. flip/flip_batch test changes without
    modifying state; accept/accept_many make them permanent.
    """
    n_seqs, n_cols = seqs.shape
    positions = [(s, c) for s in range(n_seqs) for c in range(n_cols)]
    B = len(positions)

    if budget is None:
        budget = B * 200

    cur_seq = seqs.copy()
    tracker.build(cur_seq)
    cur_e = tracker.energy()
    budget -= 1

    best_seq = cur_seq.copy()
    best_e = cur_e

    singles_e = np.empty(B)
    while budget > 0 and best_e > 0:
        # Phase 1: greedy singles scan
        singles_e.fill(np.inf)
        improved = False
        for idx, (s, c) in enumerate(positions):
            if budget <= 0:
                break
            e = tracker.flip(cur_seq, s, c)
            budget -= 1
            singles_e[idx] = e
            if e < cur_e:
                cur_e = tracker.accept(cur_seq, s, c)
                improved = True
                break

        if improved:
            best_seq, best_e = _update_best(cur_seq, cur_e, best_seq, best_e)
            continue

        # Phase 2: 2-bit rescue among top-K singles
        K = min(B, max(12, B // 4))
        top = np.argsort(singles_e)[:K]
        for i in range(K):
            if budget <= 0:
                break
            si, ci = positions[top[i]]
            for j in range(i + 1, K):
                if budget <= 0:
                    break
                sj, cj = positions[top[j]]
                e = tracker.flip_batch(cur_seq, [(si, ci), (sj, cj)])
                budget -= 1
                if e < cur_e:
                    cur_e = tracker.accept_many(cur_seq, [(si, ci), (sj, cj)])
                    improved = True
                    break
            if improved:
                break

        if improved:
            best_seq, best_e = _update_best(cur_seq, cur_e, best_seq, best_e)
            continue

        # Phase 3: 3-bit rescue among narrower pool
        K3 = min(K // 2, 10)
        for i in range(K3):
            if budget <= 0:
                break
            si, ci = positions[top[i]]
            for j in range(i + 1, K3):
                if budget <= 0:
                    break
                sj, cj = positions[top[j]]
                for k in range(j + 1, K3):
                    if budget <= 0:
                        break
                    sk, ck = positions[top[k]]
                    e = tracker.flip_batch(cur_seq, [(si, ci), (sj, cj), (sk, ck)])
                    budget -= 1
                    if e < cur_e:
                        cur_e = tracker.accept_many(cur_seq, [(si, ci), (sj, cj), (sk, ck)])
                        improved = True
                        break
                if improved:
                    break
            if improved:
                break

        if improved:
            best_seq, best_e = _update_best(cur_seq, cur_e, best_seq, best_e)
            continue

        # Phase 4: Kick
        if budget > 0 and cur_e > 0:
            kicks = [(s, int(rng.integers(0, n_cols))) for s in range(n_seqs)]
            for s, c in kicks:
                cur_seq[s, c] *= -1
            tracker.rebuild(cur_seq)
            cur_e = tracker.energy()
            budget -= 1
            best_seq, best_e = _update_best(cur_seq, cur_e, best_seq, best_e)

    return best_seq, best_e
