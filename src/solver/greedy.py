"""Budgeted single-flip scoring and acceptance."""

from __future__ import annotations

import numpy as np

from ..models import CandidateBudget, Int8Array, SearchStats
from ..tracker import Tracker
from .config import SolverConfig

SINGLE_BATCH_SIZE = 64


def greedy_descent(
    cur_seq: Int8Array,
    tracker: Tracker,
    positions: tuple[tuple[int, int], ...],
    cur_e: int,
    budget: CandidateBudget,
    q_scale: int,
    cfg: SolverConfig,
    stats: SearchStats,
) -> tuple[bool, int, int]:
    """Score all affordable singles and prioritize a directly solving flip.

    Returns:
        Whether a flip was accepted, its resulting energy, and the number of
        single-flip scores charged to the candidate budget.

    Note:
        Charge every scored flip. Without a direct solution, retain the first
        improving position (or the existing geometry-weighted batch selection).
        A partial final budget permits only a prefix of the neighborhood.
    """
    used = budget.take(len(positions))
    stats.greedy_candidate_evals += used
    energies = tracker.flip_batch(0, used)
    zero = np.flatnonzero(energies == 0)
    if zero.size:
        idx = int(zero[0])
    elif cfg.geo_weight > 0:
        assert tracker._norm2 is not None
        scored = energies.astype(np.float64) + cfg.geo_weight * tracker._norm2[:used] * q_scale
        candidates = np.flatnonzero(scored < cur_e)
        if not candidates.size:
            return False, cur_e, used
        stop = (int(candidates[0]) // SINGLE_BATCH_SIZE + 1) * SINGLE_BATCH_SIZE
        batch = candidates[candidates < stop]
        idx = int(batch[np.argmin(scored[batch])])
    else:
        candidates = np.flatnonzero(energies < cur_e)
        if not candidates.size:
            return False, cur_e, used
        idx = int(candidates[0])
    updated = tracker.accept(cur_seq, *positions[idx])
    stats.greedy_energy_improvement += cur_e - updated
    stats.hit_greedy()
    if updated == 0:
        stats.record_solve("greedy", cur_e // q_scale)
    return True, updated, used
