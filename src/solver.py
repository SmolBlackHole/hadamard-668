"""Iterated local search: greedy singles, Tabu walk, targeted/random kick.

The solver owns no CLI, construction, persistence, seed, or verification logic.
Its phases are greedy singles, a Tabu walk, and targeted or random kicks.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from typing import Any

import numpy as np
import numpy.typing as npt
from numba import njit  # pyright: ignore[reportMissingImports]

from .canonical import orbit_hash, validation_hash
from .models import (
    CandidateBudget,
    Int8Array,
    PhaseEvent,
    SearchPhase,
    SearchStats,
    SolverResult,
)
from .tracker import Tracker

SINGLE_BATCH_SIZE = 64
_jit: Any = njit


# --- Numba Tabu Kernel --------------------------------------------------------


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


# --- Config & Helpers ---------------------------------------------------------


@dataclass(frozen=True)
class SolverConfig:
    """Configure the iterated local search and its escape phases.

    Attributes:
        kick: Enable targeted and random escape phases after a stalled descent.
        tabu: Enable a Tabu walk after a stalled or low-``Q`` descent.
        tabu_steps: Maximum transitions in one Tabu walk.
        tabu_tenure: Initial soft penalty assigned to the accepted flip.
        tabu_decay: Multiplicative penalty decay applied after every step.
        tabu_noise: Scale of random multiplicative score noise.
        geo_weight: Optional weight for singleton-delta norm during greedy scans.
        targeted_escape: Allow one targeted escape per outer search.
        targeted_selection: Proposal selection: ``legacy`` first columns,
            ``random`` sampled columns, or ``residual`` sampled columns ranked
            by full post-flip ``Q``. Ranking consumes candidate evaluations.
        escape_quench_budget: Candidate limit for each nested targeted quench.
        qwindow_high: Empirical ``Q`` threshold for handing a greedy improvement
            directly to Tabu or escape handling. Zero disables the handoff.
        trace_phases: Record detailed state and cost events for every phase.
        tabu_accept_equal: Continue from a changed equal-best Tabu snapshot.
    """

    kick: bool = True
    tabu: bool = True
    tabu_steps: int = 400
    tabu_tenure: float = 5.0
    tabu_decay: float = 0.7
    tabu_noise: float = 0.2
    geo_weight: float = 0.0
    targeted_escape: bool = True
    escape_quench_budget: int = 10_000_000
    qwindow_high: int = 9  # empirical handoff from greedy to escape phases
    trace_phases: bool = False
    targeted_selection: str = "legacy"
    tabu_accept_equal: bool = False

    def __post_init__(self) -> None:
        if self.tabu_steps < 0:
            raise ValueError("tabu_steps cannot be negative")
        if self.tabu_tenure < 0 or not 0 <= self.tabu_decay <= 1:
            raise ValueError("tabu_tenure must be non-negative and tabu_decay must be in [0, 1]")
        if self.tabu_noise < 0 or self.geo_weight < 0:
            raise ValueError("tabu_noise and geo_weight cannot be negative")
        if self.escape_quench_budget <= 0:
            raise ValueError("escape_quench_budget must be positive")
        if self.qwindow_high < 0:
            raise ValueError("qwindow_high cannot be negative")
        if self.targeted_selection not in {"legacy", "random", "residual"}:
            raise ValueError("targeted_selection must be legacy, random, or residual")


@dataclass(frozen=True)
class TabuWalkResult:
    """Summarize a Tabu walk and its adopted best state.

    ``energy`` is ``None`` when the walk adopted no state. ``steps``
    counts executed walk transitions, not candidate evaluations; each step
    evaluates all ``4n`` single flips.

    Attributes:
        energy: Adopted best energy, possibly equal to the start, or ``None``.
        steps: Number of executed walk transitions.
        solve_q_before: ``Q`` immediately before a solving transition.
        lowest_q: Lowest ``Q`` visited by the walk.
        downhill_moves: Transitions that decreased ``Q``.
        lateral_moves: Transitions that preserved ``Q``.
        uphill_moves: Transitions that increased ``Q``.
    """

    energy: int | None
    steps: int
    solve_q_before: int | None = None
    lowest_q: int | None = None
    downhill_moves: int = 0
    lateral_moves: int = 0
    uphill_moves: int = 0


@dataclass(frozen=True)
class TargetedKickResult:
    """Summarize targeted quenches and the updated outer best state.

    Attributes:
        solved: Whether a nested quench reached zero energy.
        sequences: Solving state, or the unchanged outer current state.
        energy: Energy of ``sequences``.
        best_sequences: Best state known after all attempted quenches.
        best_energy: Energy of ``best_sequences``.
        accepted_moves: Targeted starts plus accepted nested-search moves.
    """

    solved: bool
    sequences: Int8Array
    energy: int
    best_sequences: Int8Array
    best_energy: int
    accepted_moves: int


@lru_cache(maxsize=16)
def _positions(n_seqs: int, n_cols: int) -> tuple[tuple[int, int], ...]:
    return tuple((s, c) for s in range(n_seqs) for c in range(n_cols))


def _update_best(
    cur_seq: Int8Array,
    cur_e: int,
    best_seq: Int8Array,
    best_e: int,
) -> tuple[Int8Array, int]:
    if cur_e < best_e:
        return cur_seq.copy(), cur_e
    return best_seq, best_e


def _basis_minus(sequences: Int8Array) -> int:
    return int(np.count_nonzero(np.prod(sequences, axis=0, dtype=np.int8) < 0))


def _record_phase(
    stats: SearchStats,
    config: SolverConfig,
    phase: SearchPhase,
    before: Int8Array,
    after: Int8Array,
    q_before: int,
    q_after: int,
    lowest_q: int,
    candidate_evals: int,
    accepted_moves: int,
    downhill_moves: int,
    lateral_moves: int,
    uphill_moves: int,
    outcome: str,
) -> None:
    """Append a complete phase event when tracing is enabled."""
    if not config.trace_phases:
        return
    stats.phase_events.append(
        PhaseEvent(
            phase=phase,
            q_before=q_before,
            q_after=q_after,
            lowest_q=lowest_q,
            candidate_evals=candidate_evals,
            accepted_moves=accepted_moves,
            downhill_moves=downhill_moves,
            lateral_moves=lateral_moves,
            uphill_moves=uphill_moves,
            basis_minus_before=_basis_minus(before),
            basis_minus_after=_basis_minus(after),
            state_hash_after=validation_hash(after),
            orbit_hash_after=orbit_hash(after),
            outcome=outcome,
        )
    )


def _targeted_candidates(
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


def _rank_targeted_candidates(
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


# --- Core Phases --------------------------------------------------------------


def _greedy_descent(
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


def _tabu_phase(
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
        _record_phase(
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
    _record_phase(
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


def _targeted_kick(
    cur_seq: Int8Array,
    tracker: Tracker,
    rng: np.random.Generator,
    budget: CandidateBudget,
    best_seq: Int8Array,
    best_e: int,
    cfg: SolverConfig,
    stats: SearchStats,
) -> TargetedKickResult:
    """Run nested quenches from targeted four-bit starts.

    Each start flips one bit in every sequence. Its nested search has a fresh
    local budget capped by ``escape_quench_budget``; all consumed work is then
    charged to the outer budget. Unsolved quenches do not replace the current
    outer state, but they may improve the global best state.
    """
    n = cur_seq.shape[1]
    assert tracker._delta is not None and tracker._u is not None
    _d = tracker._delta
    _u = tracker._u
    if cfg.targeted_selection == "legacy":
        candidates = _targeted_candidates(_d, _u, n)
    else:
        candidates = _targeted_candidates(_d, _u, n, rng)
    if cfg.targeted_selection == "residual":
        before_ranking = budget.used
        candidates = _rank_targeted_candidates(candidates, _d, _u, n, budget)
        stats.escape_candidate_evals += budget.used - before_ranking
    accepted_moves = 0

    if not candidates:
        return TargetedKickResult(False, cur_seq, tracker.energy(), best_seq, best_e, 0)
    quench_cfg = SolverConfig(targeted_escape=False)

    for c0, c1, c2, c3 in candidates:
        if budget.remaining <= 1:
            break
        budget.take(1)
        c0 %= n
        c1 %= n
        c2 %= n
        c3 %= n
        cand = cur_seq.copy()
        cand[0, c0] *= -1
        cand[1, c1] *= -1
        cand[2, c2] *= -1
        cand[3, c3] *= -1
        t2 = Tracker()
        quench = _search(
            cand,
            t2,
            rng,
            budget=CandidateBudget(min(cfg.escape_quench_budget, budget.remaining)),
            config=quench_cfg,
        )
        budget.take(quench.candidate_evals)
        stats.targeted_quenches += 1
        stats.quench_moves += quench.stats.total_accepted_moves
        stats.quench_candidate_evals += quench.stats.total_candidate_evals
        stats.escape_candidate_evals += 1
        accepted_moves += 1 + quench.stats.total_accepted_moves
        if quench.energy < best_e:
            best_seq, best_e = quench.sequences.copy(), quench.energy
        if quench.solved:
            phase = quench.stats.solve_phase or "unknown"
            q_before = quench.stats.solve_q_before if quench.stats.solve_q_before is not None else 0
            stats.record_solve(f"targeted:{phase}", q_before)
            return TargetedKickResult(True, quench.sequences, 0, best_seq, best_e, accepted_moves)

    return TargetedKickResult(False, cur_seq, tracker.energy(), best_seq, best_e, accepted_moves)


def _random_kick(
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


# --- Main Solver --------------------------------------------------------------


def search(
    seqs: Int8Array,
    tracker: Tracker,
    rng: np.random.Generator,
    *,
    candidate_budget: int,
    config: SolverConfig | None = None,
) -> SolverResult:
    """Search for a zero-residual GS4 state within a candidate budget.

    Args:
        seqs: Start state with shape ``(4, n)``. It is copied before search.
        tracker: Mutable tracker instance owned by this search.
        rng: Random source for Tabu noise and escape moves.
        candidate_budget: Maximum logical candidate evaluations.
        config: Search settings, or defaults when omitted.

    Returns:
        Best state seen, its exact matrix energy, consumed candidate work, and
        phase statistics.

    Note:
        A zero energy means the residual equations hold; final matrix
        acceptance remains the pipeline's responsibility. The supplied tracker
        follows the current working state and need not match the returned best
        state when the search exits.
    """
    return _search(seqs, tracker, rng, CandidateBudget(candidate_budget), config)


def _search(
    seqs: Int8Array,
    tracker: Tracker,
    rng: np.random.Generator,
    budget: CandidateBudget,
    config: SolverConfig | None = None,
) -> SolverResult:
    """Run the shared search loop against a caller-owned budget."""
    cfg = config if config is not None else SolverConfig()
    n_seqs, n_cols = seqs.shape
    positions = _positions(n_seqs, n_cols)
    stats = SearchStats()
    q_scale = 64 * n_cols

    cur_seq = seqs.copy()
    t0 = time.perf_counter()
    tracker.build(cur_seq)
    stats.rebuild_time_s += time.perf_counter() - t0
    cur_e = tracker.energy()
    best_seq, best_e = cur_seq.copy(), cur_e
    if cur_e == 0:
        stats.record_solve("initial", 0)
    lowest_q_seen = cur_e // q_scale  # niedrigstes Q bisher
    times_at_lowest = 0  # wie oft schon beim niedrigsten Q gestuckt
    early_escape_done = False
    _record_phase(
        stats,
        cfg,
        SearchPhase.INITIALIZE,
        cur_seq,
        cur_seq,
        cur_e // q_scale,
        cur_e // q_scale,
        cur_e // q_scale,
        0,
        0,
        0,
        0,
        0,
        "solved" if cur_e == 0 else "ready",
    )

    while budget.remaining > 0 and best_e > 0:
        # Phase 1: first-improvement greedy scan
        t_phase = time.perf_counter()
        phase_before = cur_seq.copy() if cfg.trace_phases else cur_seq
        q_before = cur_e // q_scale
        improved, cur_e, greedy_evals = _greedy_descent(
            cur_seq, tracker, positions, cur_e, budget, q_scale, cfg, stats
        )
        stats.greedy_time_s += time.perf_counter() - t_phase
        _record_phase(
            stats,
            cfg,
            SearchPhase.GREEDY,
            phase_before,
            cur_seq,
            q_before,
            cur_e // q_scale,
            cur_e // q_scale,
            greedy_evals,
            int(improved),
            int(improved),
            0,
            0,
            "solved" if cur_e == 0 else ("improved" if improved else "local_minimum"),
        )

        if cur_e == 0:
            best_seq, best_e = cur_seq.copy(), 0
            break

        if budget.remaining <= 0:
            best_seq, best_e = _update_best(cur_seq, cur_e, best_seq, best_e)
            break

        # Hand off low-Q improvements instead of starting another greedy scan.
        q_now = cur_e // q_scale
        if cfg.qwindow_high > 0 and q_now <= cfg.qwindow_high:
            improved = False  # force tabu/escape instead of more greedy

        # Phase 2: Tabu (if stuck)
        if not improved and cfg.tabu:
            improved, cur_e, _ = _tabu_phase(cur_seq, tracker, cur_e, rng, budget, cfg, stats)
            best_seq, best_e = _update_best(cur_seq, cur_e, best_seq, best_e)

        # Phase 3: Kick (if still stuck)
        if not improved and budget.remaining > 0 and cur_e > 0 and cfg.kick:
            t_kick = time.perf_counter()
            prev_e = cur_e
            q_now = cur_e // q_scale

            # Track niedrigstes Q
            if q_now < lowest_q_seen:
                lowest_q_seen = q_now
                times_at_lowest = 0  # reset: neues Rekordtief
            elif q_now == lowest_q_seen:
                times_at_lowest += 1  # schon wieder hier, kein Fortschritt

            # Trigger: wenn wir MIN. 2 MAL am Rekordtief gestuckt sind (Plateau erkannt)
            if (
                cfg.targeted_escape
                and not early_escape_done
                and q_now == lowest_q_seen
                and times_at_lowest >= 2
            ):
                early_escape_done = True
                phase_before = cur_seq.copy() if cfg.trace_phases else cur_seq
                q_before = cur_e // q_scale
                evals_before = budget.used
                targeted = _targeted_kick(
                    cur_seq, tracker, rng, budget, best_seq, best_e, cfg, stats
                )
                improved = targeted.solved
                cur_seq = targeted.sequences
                cur_e = targeted.energy
                best_seq = targeted.best_sequences
                best_e = targeted.best_energy
                _record_phase(
                    stats,
                    cfg,
                    SearchPhase.TARGETED,
                    phase_before,
                    cur_seq,
                    q_before,
                    cur_e // q_scale,
                    best_e // q_scale,
                    budget.used - evals_before,
                    targeted.accepted_moves,
                    0,
                    0,
                    0,
                    "solved" if targeted.solved else "exhausted",
                )
            else:
                phase_before = cur_seq.copy() if cfg.trace_phases else cur_seq
                q_before = cur_e // q_scale
                budget.take(1)
                cur_e = _random_kick(cur_seq, tracker, rng)
                improved = True
                stats.escape_candidate_evals += 1
                stats.random_kicks += 1
                if cur_e == 0:
                    stats.record_solve("random_kick", prev_e // q_scale)
                q_after = cur_e // q_scale
                _record_phase(
                    stats,
                    cfg,
                    SearchPhase.RANDOM_KICK,
                    phase_before,
                    cur_seq,
                    q_before,
                    q_after,
                    min(q_before, q_after),
                    1,
                    1,
                    int(q_after < q_before),
                    int(q_after == q_before),
                    int(q_after > q_before),
                    "solved" if cur_e == 0 else "accepted",
                )

            stats.escape_energy_improvement += prev_e - cur_e
            stats.escape_time_s += time.perf_counter() - t_kick
            if not improved:
                stats.hit_other()

        best_seq, best_e = _update_best(cur_seq, cur_e, best_seq, best_e)

    assert stats.total_candidate_evals == budget.used
    return SolverResult(best_seq, best_e, budget.used, stats)


# --- Tabu Walk Helper ---------------------------------------------------------


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
    assert (
        tracker._delta is not None
        and tracker._norm2 is not None
        and tracker._u is not None
        and tracker._update_cols is not None
        and tracker._update_lags is not None
        and tracker._update_signs is not None
    )
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
        cur_seq.copy(),
        tracker._delta.copy(),
        tracker._norm2.copy(),
        tracker._u.copy(),
        tracker._q,
        tracker._update_cols,
        tracker._update_lags,
        tracker._update_signs,
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
    tracker._adopt(best_seq, best_u, best_q, best_delta, best_norm2)
    return TabuWalkResult(
        best_e,
        evaluations,
        solve_q_before if solve_q_before >= 0 else None,
        best_q,
        downhill_moves,
        lateral_moves,
        uphill_moves,
    )
