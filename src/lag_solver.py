"""Lag-scored GS4 solver (CPU-only).

Scored by violated lags fixed minus created.
Pipeline: lag greedy -> tabu -> random kick.
No targeted kick, no GPU.
"""

from __future__ import annotations

import time

import numpy as np
import numpy.typing as npt

from .navigator import (
    SearchStats,
    SolverConfig,
    _positions,
    _random_kick,
    _tabu_phase,
    _update_best,
)
from .tracker import Tracker

__all__ = ["SearchStats", "SolverConfig", "lag_greedy", "lag_score", "search"]


def lag_score(
    u: npt.NDArray[np.int32],
    delta: npt.NDArray[np.int8],
) -> npt.NDArray[np.int64]:
    fixable = np.abs(u) == 1
    fixes = np.sum((delta == -u) & fixable, axis=1)
    creates = np.sum((u == 0) & (delta != 0), axis=1)
    return fixes - creates


def lag_greedy(
    seqs: npt.NDArray[np.int8],
    tracker: Tracker,
    max_steps: int = 2000,
) -> tuple[npt.NDArray[np.int8], int]:
    n = seqs.shape[1]
    positions = _positions(4, n)
    assert tracker._u is not None and tracker._delta is not None

    cur_e = tracker.energy()
    last_idx = -1
    for _ in range(max_steps):
        if cur_e == 0:
            break
        scores = lag_score(tracker._u, tracker._delta)
        qs = tracker.flip_qs()
        pos = np.flatnonzero(scores > 0)
        if last_idx >= 0:
            pos = pos[pos != last_idx]
        if pos.size == 0:
            if last_idx >= 0 and qs.size > 1:
                qs[last_idx] = np.iinfo(np.int64).max
            best = int(np.argmin(qs))
            if qs[best] >= tracker._q:
                break
            s, c = positions[best]
        else:
            order = np.lexsort((qs[pos], -scores[pos]))
            s, c = positions[int(pos[order[0]])]
        last_idx = s * n + c
        cur_e = tracker.accept(seqs, s, c)
    return seqs, cur_e


def search(
    seqs: npt.NDArray[np.int8],
    tracker: Tracker,
    rng: np.random.Generator,
    *,
    steps: int,
    config: SolverConfig | None = None,
) -> tuple[npt.NDArray[np.int8], int, int, SearchStats]:
    cfg = config if config is not None else SolverConfig()
    stats = SearchStats()

    cur = seqs.copy()
    tracker.build(cur)
    cur_e = tracker.energy()
    best_seq, best_e = cur.copy(), cur_e
    used = 0

    tabu_cfg = SolverConfig(
        tabu=True, kick=False, targeted_escape=False,
        tabu_steps=cfg.tabu_steps, tabu_tenure=cfg.tabu_tenure,
        tabu_decay=cfg.tabu_decay, tabu_noise=cfg.tabu_noise,
    )

    while used < steps and best_e > 0:
        t_phase = time.perf_counter()
        cur, cur_e = lag_greedy(cur, tracker)
        stats.single_time_s += time.perf_counter() - t_phase
        best_seq, best_e = _update_best(cur, cur_e, best_seq, best_e)
        if best_e == 0:
            break

        if cur_e > 0 and cfg.tabu:
            _tabu_phase(cur, tracker, cur_e, rng, tabu_cfg, stats)
            cur_e = tracker.energy()
            best_seq, best_e = _update_best(cur, cur_e, best_seq, best_e)
            if best_e == 0:
                break

        if cur_e > 0 and cfg.kick:
            t_kick = time.perf_counter()
            prev_e = cur_e
            cur_e = _random_kick(cur, tracker, rng)
            stats.kicks += 1
            stats.kick_evals += 1
            stats.energy_saved_kicks += prev_e - cur_e
            stats.kick_time_s += time.perf_counter() - t_kick
            stats._hit_other()
            used += 1
            best_seq, best_e = _update_best(cur, cur_e, best_seq, best_e)
        else:
            break

    return best_seq, best_e, used, stats
