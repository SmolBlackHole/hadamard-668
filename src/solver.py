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

    # K cap: B//8 (=84 at n=167) is enough candidates without combinatorial explosion
    K = min(B, max(12, B // 8))
    K3 = min(K // 2, 8)
    rescue_mode: bool = False  # best-mode after consecutive hits
    rescue_streak: int = 0

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
            rescue_streak = 0
            rescue_mode = False
        else:
            top = np.argpartition(singles_e, K)[:K]
            top_candidates = [positions[t] for t in top]

            mode = "best" if rescue_mode else "first"
            narrowed = top_candidates if rescue_mode else None

            # 2-bit rescue
            result = _rescue(
                cur_seq, tracker, top_candidates, 2, cur_e, mode=mode, narrowed=narrowed
            )
            if result is not None:
                cur_e = result
                improved = True
                rescue_streak += 1
                if rescue_streak >= 2:
                    rescue_mode = True  # productive valley, go deep
            else:
                rescue_streak = 0
                rescue_mode = False

                # 3-bit rescue among narrower pool
                result = _rescue(cur_seq, tracker, top_candidates[:K3], 3, cur_e, mode=mode)
                if result is not None:
                    cur_e = result
                    improved = True

        # Phase 4: Kick
        if not improved and steps > 0 and cur_e > 0:
            cols = rng.integers(0, n_cols, size=n_seqs)
            for s in range(n_seqs):
                cur_seq[s, int(cols[s])] *= -1
            tracker.build(cur_seq, band_rows=_rows_band, band_cols=_cols_band)
            cur_e = tracker.energy()
            steps -= 1
            rescue_mode = False
            rescue_streak = 0

        best_seq, best_e = _update_best(cur_seq, cur_e, best_seq, best_e)

    return best_seq, best_e, total_budget - max(steps, 0)


def _rescue(
    cur_seq: npt.NDArray[np.int8],
    tracker: GramTracker,
    candidates: list[tuple[int, int]],
    width: int,
    cur_e: int,
    *,
    mode: str = "first",
    narrowed: list[tuple[int, int]] | None = None,
) -> int | None:
    """Try width-bit combinations. Returns improved energy or None.

    mode="first": stop at first improvement (fast descent).
    mode="best":  scan all, take best (for deep plateaus).
    narrowed:     optional pre-filtered candidate list for "best" mode.
    """
    band_rows = tracker._rows_band
    band_cols = tracker._cols_band
    if not band_rows or not band_cols:
        return None

    pool = narrowed if narrowed is not None else candidates
    best_e = cur_e
    best_combo: tuple | None = None

    for combo in combinations(pool, width):
        row_parts = [band_rows[s][c] for s, c in combo]
        col_parts = [band_cols[s][c] for s, c in combo]
        rows = np.concatenate(row_parts)
        cols = np.concatenate(col_parts)

        e = tracker._compute_delta(rows.astype(np.int16), cols.astype(np.int16))
        if e < cur_e:
            if mode == "first":
                for s, c in combo:
                    cur_seq[s, c] *= -1
                tracker.build(cur_seq, band_rows=band_rows, band_cols=band_cols)
                return tracker.energy()
            if e < best_e:
                best_e = e
                best_combo = combo

    if best_combo is None:
        return None

    for s, c in best_combo:
        cur_seq[s, c] *= -1
    tracker.build(cur_seq, band_rows=band_rows, band_cols=band_cols)
    return tracker.energy()
