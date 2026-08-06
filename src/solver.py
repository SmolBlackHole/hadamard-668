"""Iterated local search — singles scan + 2/3-bit rescue + kick, energy-budgeted."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from tracker import GramTracker


def _update_best(
    cur_seq: npt.NDArray[np.int8],
    cur_e: int,
    best_seq: npt.NDArray[np.int8],
    best_e: int,
) -> tuple[npt.NDArray[np.int8], int]:
    if cur_e < best_e:
        return cur_seq.copy(), cur_e
    return best_seq, best_e


def search(
    seqs: npt.NDArray[np.int8],
    tracker: GramTracker,
    rng: np.random.Generator,
    *,
    steps: int,
) -> tuple[npt.NDArray[np.int8], int, int]:
    """Iterated local search. Returns (best_seq, best_energy, iterations)."""
    n_seqs, n_cols = seqs.shape
    positions: list[tuple[int, int]] = [(s, c) for s in range(n_seqs) for c in range(n_cols)]
    B = len(positions)

    cur_seq = seqs.copy()
    tracker.build(cur_seq)
    cur_e = tracker.energy()
    steps -= 1
    total_budget = steps
    iters: int = 0

    best_seq = cur_seq.copy()
    best_e = cur_e

    singles_e = np.empty(B)
    while steps > 0 and best_e > 0:
        iters += 1
        # Phase 1: greedy singles scan
        singles_e.fill(np.inf)
        improved = False
        for idx, (s, c) in enumerate(positions):
            if steps <= 0:
                break
            e = tracker.flip(cur_seq, s, c)
            steps -= 1
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
            if steps <= 0:
                break
            si, ci = positions[top[i]]
            for j in range(i + 1, K):
                if steps <= 0:
                    break
                sj, cj = positions[top[j]]
                e = tracker.flip_batch(cur_seq, [(si, ci), (sj, cj)])
                steps -= 1
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
            if steps <= 0:
                break
            si, ci = positions[top[i]]
            for j in range(i + 1, K3):
                if steps <= 0:
                    break
                sj, cj = positions[top[j]]
                for k in range(j + 1, K3):
                    if steps <= 0:
                        break
                    sk, ck = positions[top[k]]
                    e = tracker.flip_batch(cur_seq, [(si, ci), (sj, cj), (sk, ck)])
                    steps -= 1
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
        if steps > 0 and cur_e > 0:
            kicks = [(s, int(rng.integers(0, n_cols))) for s in range(n_seqs)]
            for s, c in kicks:
                cur_seq[s, c] *= -1
            tracker.build(cur_seq)
            cur_e = tracker.energy()
            steps -= 1
            best_seq, best_e = _update_best(cur_seq, cur_e, best_seq, best_e)

    return best_seq, best_e, total_budget - max(steps, 0)
