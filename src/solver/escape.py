"""Targeted proposals, residual ranking and random escape moves."""

from __future__ import annotations

from itertools import product

import numpy as np
import numpy.typing as npt

from ..models import CandidateBudget, Int8Array
from ..tracker import Tracker


def targeted_candidates(
    delta: Int8Array,
    u: npt.NDArray[np.int32],
    n: int,
    rng: np.random.Generator | None = None,
) -> list[tuple[int, int, int, int]]:
    """Spread at most 625 four-sequence starts over feasible residual lags.

    Without ``rng``, preserve the legacy first-five-column proposal order.
    Otherwise sample up to five matching columns uniformly without replacement
    for each sequence and lag, including random order within the sample.
    """
    per_lag: list[list[tuple[int, int, int, int]]] = []
    for k in np.flatnonzero(u):
        target = -u[k]
        columns: list[list[int]] = []
        for s in range(4):
            matching = [c for c in range(n) if delta[s * n + c, k] == target]
            if rng is None:
                columns.append(matching[:5])
            else:
                columns.append(
                    [
                        int(c)
                        for c in rng.choice(matching, size=min(5, len(matching)), replace=False)
                    ]
                )
        if all(columns):
            per_lag.append([(c0, c1, c2, c3) for c0, c1, c2, c3 in product(*columns)])
    if not per_lag:
        return []

    limit = 625
    base, extra = divmod(limit, len(per_lag))
    selected: list[tuple[int, int, int, int]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for lag_index, candidates in enumerate(per_lag):
        take = min(len(candidates), base + (lag_index < extra))
        if take == 1:
            ranks = [len(candidates) // 2]
        elif take:
            ranks = [i * (len(candidates) - 1) // (take - 1) for i in range(take)]
        else:
            ranks = []
        for rank in ranks:
            candidate = candidates[rank]
            if candidate not in seen:
                seen.add(candidate)
                selected.append(candidate)

    offsets = [0] * len(per_lag)
    while len(selected) < limit:
        advanced = False
        for lag_index, candidates in enumerate(per_lag):
            if offsets[lag_index] == len(candidates):
                continue
            candidate = candidates[offsets[lag_index]]
            offsets[lag_index] += 1
            advanced = True
            if candidate in seen:
                continue
            seen.add(candidate)
            selected.append(candidate)
            if len(selected) == limit:
                break
        if not advanced:
            break
    return selected


def rank_targeted_candidates(
    candidates: list[tuple[int, int, int, int]],
    delta: Int8Array,
    u: npt.NDArray[np.int32],
    n: int,
    budget: CandidateBudget,
) -> list[tuple[int, int, int, int]]:
    """Stably rank the affordable proposal prefix by exact post-flip ``Q``.

    Each proposal flips different sequences, so singleton residual deltas add
    without interaction corrections. Reserve one evaluation per scored proposal
    before computing its score. Initializing a later quench is separate work.
    """
    count = budget.take(len(candidates))
    scored: list[tuple[int, tuple[int, int, int, int]]] = []
    for candidate in candidates[:count]:
        residual = u.astype(np.int64)
        for s, c in enumerate(candidate):
            residual += delta[s * n + c]
        scored.append((int(np.dot(residual, residual)), candidate))
    scored.sort(key=lambda item: item[0])
    return [candidate for _, candidate in scored]


def random_kick(
    cur_seq: Int8Array,
    tracker: Tracker,
    rng: np.random.Generator,
) -> int:
    """Mutate the current state with one flip in two random sequences.

    Returns:
        The exact energy after both accepted flips.
    """
    n_seqs, n_cols = cur_seq.shape
    n_flips = 2
    cols = np.asarray(rng.integers(0, n_cols, size=n_seqs), dtype=np.intp)
    cur_e = 0
    for s in rng.choice(n_seqs, size=min(n_flips, n_seqs), replace=False):
        cur_e = tracker.accept(cur_seq, s, int(cols[s]))
    return cur_e
