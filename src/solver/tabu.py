"""Compiled soft-Tabu walks and exact snapshot adoption."""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import numpy.typing as npt
from numba import njit

from ..models import CandidateBudget, Int8Array, SearchPhase, SearchStats
from ..tracker import Tracker, TrackerSnapshot
from .config import SolverConfig
from .state import TabuWalkResult
from .tracing import record_phase

_jit: Any = njit


@_jit(cache=True)
def tabu_walk_kernel(
    seqs: Int8Array,
    delta: Int8Array,
    norm2: npt.NDArray[np.int32],
    u: npt.NDArray[np.int32],
    q: int,
    update_cols: npt.NDArray[np.intp],
    update_lags: npt.NDArray[np.intp],
    update_signs: Int8Array,
    noise: npt.NDArray[np.float64],
    tenure: float,
    decay: float,
    accept_equal: bool = False,
) -> tuple[
    Int8Array,
    Int8Array,
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
    int,
    int,
    int,
    int,
    int,
    int,
]:
    """Run exact Tabu steps over mutable tracker buffers.

    Args:
        seqs: Working sequence state with shape ``(4, n)``.
        delta: Reduced singleton-delta cache with shape ``(4n, m)``.
        norm2: Squared norm of each singleton delta.
        u: Current reduced residual with ``m = floor((n - 1) / 2)`` entries.
        q: Current reduced objective ``Q = ||u||^2``.
        update_cols: Columns affected by flipping each source column.
        update_lags: Reduced lag indices paired with ``update_cols``.
        update_signs: Negaperiodic signs paired with ``update_cols``.
        noise: Per-step multiplicative score noise; its first dimension limits
            the number of walk steps.
        tenure: Soft Tabu penalty assigned to the accepted flip.
        decay: Multiplicative Tabu-penalty decay per step.
        accept_equal: Save the latest equal-best visited state as well.

    Returns:
        Best sequence, delta, norm, and residual snapshots; best ``Q``; steps
        executed; pre-solve ``Q`` or ``-1``; and downhill, lateral, and uphill
        move counts.

    Note:
        The input state buffers are mutated in place. Best-state buffers are
        meaningful only when the walk visits a state below its initial ``Q``,
        unless ``accept_equal`` initializes them with the starting state.
    """
    n_seqs, n_cols = seqs.shape
    n_lags = u.size
    tabu = np.zeros((n_seqs, n_cols), dtype=np.float64)
    best_seq = seqs.copy() if accept_equal else np.empty_like(seqs)
    best_delta = delta.copy() if accept_equal else np.empty_like(delta)
    best_norm2 = norm2.copy() if accept_equal else np.empty_like(norm2)
    best_u = u.copy() if accept_equal else np.empty_like(u)
    best_q = q
    used = 0
    solve_q_before = -1
    downhill_moves = 0
    lateral_moves = 0
    uphill_moves = 0

    for step in range(noise.shape[0]):
        best_score = np.inf
        index = 0
        for candidate in range(n_seqs * n_cols):
            delta_q = int(norm2[candidate])
            for lag in range(n_lags):
                delta_q += 2 * int(delta[candidate, lag]) * int(u[lag])
            s = candidate // n_cols
            c = candidate - s * n_cols
            score = (q + delta_q) * (1.0 + tabu[s, c] + noise[step, s, c])
            if score < best_score:
                best_score = score
                index = candidate

        s = index // n_cols
        c = index - s * n_cols
        delta_q = int(norm2[index])
        for lag in range(n_lags):
            delta_q += 2 * int(delta[index, lag]) * int(u[lag])
        previous_q = q
        q += delta_q
        if q < previous_q:
            downhill_moves += 1
        elif q == previous_q:
            lateral_moves += 1
        else:
            uphill_moves += 1
        for lag in range(n_lags):
            u[lag] += delta[index, lag]

        value = seqs[s, c]
        for row_index in range(update_cols.shape[1]):
            col = update_cols[c, row_index]
            lag = update_lags[c, row_index]
            row = s * n_cols + col
            old = delta[row, lag]
            correction = update_signs[c, row_index] * value * seqs[s, col]
            new = old + correction
            norm2[row] += new * new - old * old
            delta[row, lag] = new
        for lag in range(n_lags):
            delta[index, lag] = -delta[index, lag]
        seqs[s, c] = -seqs[s, c]

        for tabu_s in range(n_seqs):
            for tabu_c in range(n_cols):
                tabu[tabu_s, tabu_c] *= decay
        tabu[s, c] = tenure
        used = step + 1

        if q < best_q or (accept_equal and q == best_q):
            best_q = q
            best_seq[:, :] = seqs
            best_delta[:, :] = delta
            best_norm2[:] = norm2
            best_u[:] = u
            if best_q == 0:
                solve_q_before = previous_q
                break

    return (
        best_seq,
        best_delta,
        best_norm2,
        best_u,
        best_q,
        used,
        solve_q_before,
        downhill_moves,
        lateral_moves,
        uphill_moves,
    )


def _tabu_walk(
    cur_seq: Int8Array,
    tracker: Tracker,
    cur_e: int,
    rng: np.random.Generator,
    config: SolverConfig,
    *,
    max_steps: int | None = None,
) -> TabuWalkResult:
    """Explore unrestricted single flips and adopt the best permitted snapshot.

    The compiled walk may move downhill, laterally, or uphill. If its best
    visited state beats the starting energy, ``cur_seq`` and ``tracker`` adopt
    the exact cached snapshot. ``tabu_accept_equal`` also allows the latest
    changed equal-best state. Otherwise both main-search objects remain unchanged.
    """
    steps = config.tabu_steps if max_steps is None else min(config.tabu_steps, max_steps)
    if steps <= 0:
        return TabuWalkResult(None, 0)

    n_seqs, n_cols = cur_seq.shape
    snapshot = tracker.snapshot()
    noise = config.tabu_noise * rng.random((steps, n_seqs, n_cols))
    (
        best_seq,
        best_delta,
        best_norm2,
        best_u,
        best_q,
        evaluations,
        solve_q_before,
        downhill_moves,
        lateral_moves,
        uphill_moves,
    ) = tabu_walk_kernel(
        snapshot.sequences,
        snapshot.delta,
        snapshot.norm2,
        snapshot.u,
        snapshot.q,
        snapshot.update_cols,
        snapshot.update_lags,
        snapshot.update_signs,
        noise,
        config.tabu_tenure,
        config.tabu_decay,
        config.tabu_accept_equal,
    )
    best_e = 64 * n_cols * best_q
    accept_equal = (
        config.tabu_accept_equal and best_e == cur_e and not np.array_equal(best_seq, cur_seq)
    )
    if best_e >= cur_e and not accept_equal:
        return TabuWalkResult(
            None,
            evaluations,
            lowest_q=best_q,
            downhill_moves=downhill_moves,
            lateral_moves=lateral_moves,
            uphill_moves=uphill_moves,
        )

    cur_seq[...] = best_seq
    tracker.adopt_snapshot(
        TrackerSnapshot(
            best_seq,
            best_u,
            best_q,
            best_delta,
            best_norm2,
            snapshot.update_cols,
            snapshot.update_lags,
            snapshot.update_signs,
        )
    )
    return TabuWalkResult(
        best_e,
        evaluations,
        solve_q_before if solve_q_before >= 0 else None,
        best_q,
        downhill_moves,
        lateral_moves,
        uphill_moves,
    )


def tabu_phase(
    cur_seq: Int8Array,
    tracker: Tracker,
    cur_e: int,
    rng: np.random.Generator,
    budget: CandidateBudget,
    cfg: SolverConfig,
    stats: SearchStats,
) -> tuple[bool, int, int]:
    """Run one Tabu walk and update its phase statistics.

    Returns:
        Whether the walk adopted a strict or enabled equal improvement, its energy,
        and the number of executed walk steps. The third value is not a count
        of candidate evaluations; each step costs ``4n`` evaluations.
    """
    n_cols = cur_seq.shape[1]
    t0 = time.perf_counter()
    q_start = cur_e // (64 * n_cols)
    prev_e = cur_e
    before = cur_seq.copy() if cfg.trace_phases else cur_seq
    max_steps = budget.remaining // cur_seq.size
    result = _tabu_walk(cur_seq, tracker, cur_e, rng, cfg, max_steps=max_steps)
    budget.take(result.steps * cur_seq.size)
    stats.tabu_time_s += time.perf_counter() - t0
    stats.tabu_walks += 1
    stats.tabu_moves += result.steps
    stats.tabu_candidate_evals += result.steps * cur_seq.size
    if q_start == 1:
        stats.tabu_walks_q1 += 1
    elif q_start == 2:
        stats.tabu_walks_q2 += 1
    else:
        stats.tabu_walks_q3plus += 1
    if result.energy is not None:
        stats.tabu_energy_improvement += prev_e - result.energy
        if result.energy < prev_e:
            stats.tabu_improvements += 1
            if q_start == 1:
                stats.tabu_improvements_q1 += 1
            elif q_start == 2:
                stats.tabu_improvements_q2 += 1
            else:
                stats.tabu_improvements_q3plus += 1
        if result.energy == 0:
            stats.tabu_solves += 1
            if q_start == 1:
                stats.tabu_solves_q1 += 1
            elif q_start == 2:
                stats.tabu_solves_q2 += 1
            else:
                stats.tabu_solves_q3plus += 1
            q_before = result.solve_q_before if result.solve_q_before is not None else q_start
            stats.record_solve("tabu", q_before)
        stats.hit_other()
        record_phase(
            stats,
            cfg,
            SearchPhase.TABU,
            before,
            cur_seq,
            q_start,
            result.energy // (64 * n_cols),
            result.lowest_q if result.lowest_q is not None else q_start,
            result.steps * cur_seq.size,
            result.steps,
            result.downhill_moves,
            result.lateral_moves,
            result.uphill_moves,
            "solved" if result.energy == 0 else "improved" if result.energy < prev_e else "equal",
        )
        return True, result.energy, result.steps
    record_phase(
        stats,
        cfg,
        SearchPhase.TABU,
        before,
        cur_seq,
        q_start,
        q_start,
        result.lowest_q if result.lowest_q is not None else q_start,
        result.steps * cur_seq.size,
        result.steps,
        result.downhill_moves,
        result.lateral_moves,
        result.uphill_moves,
        "no_improvement",
    )
    return False, cur_e, result.steps
