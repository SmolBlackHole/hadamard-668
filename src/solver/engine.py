"""Orchestrate GS4 search phases, best states and nested quench budgets."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Protocol

import numpy as np
import numpy.typing as npt

from ..models import CandidateBudget, Int8Array, SearchPhase, SearchStats, SolverResult
from ..tracker import Tracker
from . import escape, greedy, tabu
from .config import SolverConfig
from .state import SearchState, TargetedKickResult
from .tracing import record_phase

GreedyOperator = Callable[
    [
        Int8Array,
        Tracker,
        tuple[tuple[int, int], ...],
        int,
        CandidateBudget,
        int,
        SolverConfig,
        SearchStats,
    ],
    tuple[bool, int, int],
]
TabuOperator = Callable[
    [Int8Array, Tracker, int, np.random.Generator, CandidateBudget, SolverConfig, SearchStats],
    tuple[bool, int, int],
]


class CandidateOperator(Protocol):
    """Generate targeted proposals without owning the search loop."""

    def __call__(
        self,
        delta: Int8Array,
        u: npt.NDArray[np.int32],
        n: int,
        rng: np.random.Generator | None = None,
    ) -> list[tuple[int, int, int, int]]:
        """Return candidate columns in reproducible evaluation order."""
        ...


@dataclass(frozen=True)
class SearchOperators:
    """Explicit per-search phase functions, shared with nested quenches.

    Defaults resolve the production functions when this object is constructed.
    Replacements must honor the same mutation, RNG and budget contracts.
    """

    greedy: GreedyOperator = field(default_factory=lambda: greedy.greedy_descent)
    tabu: TabuOperator = field(default_factory=lambda: tabu.tabu_phase)
    candidates: CandidateOperator = field(default_factory=lambda: escape.targeted_candidates)
    random_kick: Callable[[Int8Array, Tracker, np.random.Generator], int] = field(
        default_factory=lambda: escape.random_kick
    )


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


def _targeted_kick(
    cur_seq: Int8Array,
    tracker: Tracker,
    rng: np.random.Generator,
    budget: CandidateBudget,
    best_seq: Int8Array,
    best_e: int,
    cfg: SolverConfig,
    stats: SearchStats,
    operators: SearchOperators | None = None,
) -> TargetedKickResult:
    """Run nested quenches from targeted four-bit starts.

    Each start flips one bit in every sequence. Its nested search has a fresh
    local budget capped by ``escape_quench_budget``; all consumed work is then
    charged to the outer budget. Unsolved quenches do not replace the current
    outer state, but they may improve the global best state.
    """
    operators = operators or SearchOperators()
    n = cur_seq.shape[1]
    assert tracker._delta is not None and tracker._u is not None
    _d = tracker._delta
    _u = tracker._u
    if cfg.targeted_selection == "legacy":
        candidates = operators.candidates(_d, _u, n)
    else:
        candidates = operators.candidates(_d, _u, n, rng)
    if cfg.targeted_selection == "residual":
        before_ranking = budget.used
        candidates = escape.rank_targeted_candidates(candidates, _d, _u, n, budget)
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
            operators=operators,
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


def search(
    seqs: Int8Array,
    tracker: Tracker,
    rng: np.random.Generator,
    *,
    candidate_budget: int,
    config: SolverConfig | None = None,
    operators: SearchOperators | None = None,
) -> SolverResult:
    """Search for a zero-residual GS4 state within a candidate budget.

    Args:
        seqs: Start state with shape ``(4, n)``. It is copied before search.
        tracker: Mutable tracker instance owned by this search.
        rng: Random source for Tabu noise and escape moves.
        candidate_budget: Maximum logical candidate evaluations.
        config: Search settings, or defaults when omitted.
        operators: Optional per-run phase replacements, including nested quenches.

    Returns:
        Best state seen, its exact matrix energy, consumed candidate work, and
        phase statistics.

    Note:
        A zero energy means the residual equations hold; final matrix
        acceptance remains the pipeline's responsibility. The supplied tracker
        follows the current working state and need not match the returned best
        state when the search exits.
    """
    return _search(seqs, tracker, rng, CandidateBudget(candidate_budget), config, operators)


def _search(
    seqs: Int8Array,
    tracker: Tracker,
    rng: np.random.Generator,
    budget: CandidateBudget,
    config: SolverConfig | None = None,
    operators: SearchOperators | None = None,
) -> SolverResult:
    """Run the shared search loop against a caller-owned budget."""
    operators = operators or SearchOperators()
    cfg = config if config is not None else SolverConfig()
    n_seqs, n_cols = seqs.shape
    positions = _positions(n_seqs, n_cols)
    stats = SearchStats()
    q_scale = 64 * n_cols

    state = SearchState(seqs.copy(), tracker)
    t0 = time.perf_counter()
    state.tracker.build(state.sequences)
    stats.rebuild_time_s += time.perf_counter() - t0
    state.energy = state.tracker.energy()
    best_seq, best_e = state.sequences.copy(), state.energy
    if state.energy == 0:
        stats.record_solve("initial", 0)
    lowest_q_seen = state.energy // q_scale
    times_at_lowest = 0
    early_escape_done = False
    record_phase(
        stats,
        cfg,
        SearchPhase.INITIALIZE,
        state.sequences,
        state.sequences,
        state.energy // q_scale,
        state.energy // q_scale,
        state.energy // q_scale,
        0,
        0,
        0,
        0,
        0,
        "solved" if state.energy == 0 else "ready",
    )

    while budget.remaining > 0 and best_e > 0:
        # Phase 1: first-improvement greedy scan
        t_phase = time.perf_counter()
        phase_before = state.sequences.copy() if cfg.trace_phases else state.sequences
        q_before = state.energy // q_scale
        improved, state.energy, greedy_evals = operators.greedy(
            state.sequences, state.tracker, positions, state.energy, budget, q_scale, cfg, stats
        )
        stats.greedy_time_s += time.perf_counter() - t_phase
        record_phase(
            stats,
            cfg,
            SearchPhase.GREEDY,
            phase_before,
            state.sequences,
            q_before,
            state.energy // q_scale,
            state.energy // q_scale,
            greedy_evals,
            int(improved),
            int(improved),
            0,
            0,
            "solved" if state.energy == 0 else ("improved" if improved else "local_minimum"),
        )

        if state.energy == 0:
            best_seq, best_e = state.sequences.copy(), 0
            break

        if budget.remaining <= 0:
            best_seq, best_e = _update_best(state.sequences, state.energy, best_seq, best_e)
            break

        # Hand off low-Q improvements instead of starting another greedy scan.
        q_now = state.energy // q_scale
        if cfg.qwindow_high > 0 and q_now <= cfg.qwindow_high:
            improved = False  # force tabu/escape instead of more greedy

        # Phase 2: Tabu (if stuck)
        if not improved and cfg.tabu:
            improved, state.energy, _ = operators.tabu(
                state.sequences, state.tracker, state.energy, rng, budget, cfg, stats
            )
            best_seq, best_e = _update_best(state.sequences, state.energy, best_seq, best_e)

        # Phase 3: Kick (if still stuck)
        if not improved and budget.remaining > 0 and state.energy > 0 and cfg.kick:
            t_kick = time.perf_counter()
            prev_e = state.energy
            q_now = state.energy // q_scale

            # Count repeated stalls at the lowest Q visited.
            if q_now < lowest_q_seen:
                lowest_q_seen = q_now
                times_at_lowest = 0
            elif q_now == lowest_q_seen:
                times_at_lowest += 1

            # Try targeted quenches after two repeated stalls at this record low.
            if (
                cfg.targeted_escape
                and not early_escape_done
                and q_now == lowest_q_seen
                and times_at_lowest >= 2
            ):
                early_escape_done = True
                phase_before = state.sequences.copy() if cfg.trace_phases else state.sequences
                q_before = state.energy // q_scale
                evals_before = budget.used
                targeted = _targeted_kick(
                    state.sequences,
                    state.tracker,
                    rng,
                    budget,
                    best_seq,
                    best_e,
                    cfg,
                    stats,
                    operators,
                )
                improved = targeted.solved
                state.sequences = targeted.sequences
                state.energy = targeted.energy
                best_seq = targeted.best_sequences
                best_e = targeted.best_energy
                record_phase(
                    stats,
                    cfg,
                    SearchPhase.TARGETED,
                    phase_before,
                    state.sequences,
                    q_before,
                    state.energy // q_scale,
                    best_e // q_scale,
                    budget.used - evals_before,
                    targeted.accepted_moves,
                    0,
                    0,
                    0,
                    "solved" if targeted.solved else "exhausted",
                )
            else:
                phase_before = state.sequences.copy() if cfg.trace_phases else state.sequences
                q_before = state.energy // q_scale
                budget.take(1)
                state.energy = operators.random_kick(state.sequences, state.tracker, rng)
                improved = True
                stats.escape_candidate_evals += 1
                stats.random_kicks += 1
                if state.energy == 0:
                    stats.record_solve("random_kick", prev_e // q_scale)
                q_after = state.energy // q_scale
                record_phase(
                    stats,
                    cfg,
                    SearchPhase.RANDOM_KICK,
                    phase_before,
                    state.sequences,
                    q_before,
                    q_after,
                    min(q_before, q_after),
                    1,
                    1,
                    int(q_after < q_before),
                    int(q_after == q_before),
                    int(q_after > q_before),
                    "solved" if state.energy == 0 else "accepted",
                )

            stats.escape_energy_improvement += prev_e - state.energy
            stats.escape_time_s += time.perf_counter() - t_kick
            if not improved:
                stats.hit_other()

        best_seq, best_e = _update_best(state.sequences, state.energy, best_seq, best_e)

    assert stats.total_candidate_evals == budget.used
    return SolverResult(best_seq, best_e, budget.used, stats)
