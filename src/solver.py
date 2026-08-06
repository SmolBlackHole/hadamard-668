"""Iterated local search — singles scan + 2/3-bit rescue + kick, energy-budgeted."""

from __future__ import annotations

from functools import lru_cache
from itertools import combinations

import numpy as np
import numpy.typing as npt

from tracker import GramTracker


@lru_cache(maxsize=16)
def _positions(n_seqs: int, n_cols: int) -> tuple[tuple[int, int], ...]:
    return tuple((s, c) for s in range(n_seqs) for c in range(n_cols))


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
    positions = _positions(n_seqs, n_cols)
    B = len(positions)

    cur_seq = seqs.copy()
    _rows_band, _cols_band = tracker._rows_band, tracker._cols_band
    tracker.build(cur_seq, band_rows=_rows_band, band_cols=_cols_band)
    cur_e = tracker.energy()
    steps -= 1
    total_budget = steps
    iters: int = 0

    best_seq = cur_seq.copy()
    best_e = cur_e

    # adaptive K: widen when rescue often finds improvements, shrink when kick happens
    K = min(B, max(12, B // 4))
    K3 = min(K // 2, 8)
    rescue_skip: int = 0  # cooldown after kick
    improved: bool = False

    singles_e = np.empty(B)
    top: npt.NDArray[np.int64] = np.empty(0, dtype=np.int64)
    top_candidates: list[tuple[int, int]] = []
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

        if steps <= 0:
            break

        if improved:
            K = min(B, max(12, B // 4))  # singles work — narrow rescue
            K3 = min(K // 2, 8)
        elif rescue_skip > 0:
            rescue_skip -= 1
        else:
            top = np.argpartition(singles_e, K)[:K]
            top_candidates = [positions[t] for t in top]

            # 2-bit rescue
            result = _rescue(cur_seq, tracker, top_candidates, 2, cur_e)
            if result is not None:
                cur_e = result
                improved = True
                K = min(B, max(16, B // 2))  # rescue hit — widen
                K3 = min(K // 2, 12)
            else:
                # 3-bit rescue among narrower pool
                result = _rescue(cur_seq, tracker, top_candidates[:K3], 3, cur_e)
                if result is not None:
                    cur_e = result
                    improved = True
                    K = min(B, max(16, B // 2))  # rescue hit — widen
                    K3 = min(K // 2, 12)

        # Phase 4: Kick
        if not improved and steps > 0 and cur_e > 0:
            cols = rng.integers(0, n_cols, size=n_seqs)
            for s in range(n_seqs):
                cur_seq[s, int(cols[s])] *= -1
            tracker.build(cur_seq, band_rows=_rows_band, band_cols=_cols_band)
            cur_e = tracker.energy()
            steps -= 1
            K = min(B, max(12, B // 4))  # reset after kick
            K3 = min(K // 2, 8)
            rescue_skip = 3  # skip rescue for 3 iterations after kick

        best_seq, best_e = _update_best(cur_seq, cur_e, best_seq, best_e)

    return best_seq, best_e, total_budget - max(steps, 0)


def _rescue(
    cur_seq: npt.NDArray[np.int8],
    tracker: GramTracker,
    candidates: list[tuple[int, int]],
    width: int,
    cur_e: int,
) -> int | None:
    """Try all width-bit combinations. Returns best improved cur_e or None."""
    assert tracker._rows_band and tracker._cols_band, "_rescue requires band tables"

    best_e = cur_e
    best_combo: tuple | None = None

    for combo in combinations(candidates, width):
        row_parts = [tracker._rows_band[s][c] for s, c in combo]
        col_parts = [tracker._cols_band[s][c] for s, c in combo]
        rows = np.concatenate(row_parts)
        cols = np.concatenate(col_parts)

        e = tracker._compute_delta(rows.astype(np.int16), cols.astype(np.int16))
        if e < best_e:
            best_e = e
            best_combo = combo

    if best_combo is None:
        return None

    for s, c in best_combo:
        cur_seq[s, c] *= -1
    tracker.build(cur_seq, band_rows=tracker._rows_band, band_cols=tracker._cols_band)
    return tracker.energy()
