"""Iterated local search — greedy singles, Tabu walk, targeted/random kick.

The solver owns no CLI, construction, persistence, seed, or verification logic.
Its phases are greedy singles, a Tabu walk, and targeted or random kicks."""

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from typing import Any

import numpy as np
import numpy.typing as npt
from numba import njit  # pyright: ignore[reportMissingImports]

from .models import Int8Array, SearchStats, SolverResult
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
) -> tuple[
    Int8Array,
    Int8Array,
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
    int,
    int,
    int,
]:
    """Run exact Tabu steps and return the best visited tracker state."""
    n_seqs, n_cols = seqs.shape
    n_lags = u.size
    tabu = np.zeros((n_seqs, n_cols), dtype=np.float64)
    best_seq = np.empty_like(seqs)
    best_delta = np.empty_like(delta)
    best_norm2 = np.empty_like(norm2)
    best_u = np.empty_like(u)
    best_q = q
    used = 0
    solve_q_before = -1

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

        if q < best_q:
            best_q = q
            best_seq[:, :] = seqs
            best_delta[:, :] = delta
            best_norm2[:] = norm2
            best_u[:] = u
            if best_q == 0:
                solve_q_before = previous_q
                break

    return best_seq, best_delta, best_norm2, best_u, best_q, used, solve_q_before


# --- Config & Helpers ---------------------------------------------------------


@dataclass(frozen=True)
class SolverConfig:
    kick: bool = True
    tabu: bool = True
    tabu_steps: int = 400
    tabu_tenure: float = 5.0
    tabu_decay: float = 0.7
    tabu_noise: float = 0.2
    geo_weight: float = 0.0
    targeted_escape: bool = True
    escape_quench_steps: int = 10_000
    qwindow_high: int = 9  # stop greedy at this Q (C-rich band, not floor)

    def __post_init__(self) -> None:
        if self.tabu_steps < 0:
            raise ValueError("tabu_steps cannot be negative")
        if self.tabu_tenure < 0 or not 0 <= self.tabu_decay <= 1:
            raise ValueError("tabu_tenure must be non-negative and tabu_decay must be in [0, 1]")
        if self.tabu_noise < 0 or self.geo_weight < 0:
            raise ValueError("tabu_noise and geo_weight cannot be negative")
        if self.escape_quench_steps <= 0:
            raise ValueError("escape_quench_steps must be positive")
        if self.qwindow_high < 0:
            raise ValueError("qwindow_high cannot be negative")


@dataclass(frozen=True)
class TabuWalkResult:
    energy: int | None
    steps: int
    solve_q_before: int | None = None


@dataclass(frozen=True)
class TargetedKickResult:
    solved: bool
    sequences: Int8Array
    energy: int
    best_sequences: Int8Array
    best_energy: int
    steps: int


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


def _targeted_candidates(
    delta: Int8Array, u: npt.NDArray[np.int32], n: int
) -> list[tuple[int, int, int, int]]:
    """Spread at most 625 targeted candidates over all feasible residual lags."""
    per_lag: list[list[tuple[int, int, int, int]]] = []
    for k in np.flatnonzero(u):
        target = -u[k]
        columns = [[c for c in range(n) if delta[s * n + c, k] == target][:5] for s in range(4)]
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


# --- Core Phases --------------------------------------------------------------


def _greedy_descent(
    cur_seq: Int8Array,
    tracker: Tracker,
    positions: tuple[tuple[int, int], ...],
    cur_e: int,
    steps: int,
    q_scale: int,
    cfg: SolverConfig,
    stats: SearchStats,
) -> tuple[bool, int, int]:
    """Single-flip scan. Returns (improved, new_cur_e, steps_used)."""
    B = len(positions)
    scan_count = min(B, steps)
    improved = False

    for start in range(0, scan_count, SINGLE_BATCH_SIZE):
        stop = min(start + SINGLE_BATCH_SIZE, scan_count)
        energies = tracker.flip_batch(start, stop)

        if cfg.geo_weight > 0:
            _n2 = tracker._norm2
            assert _n2 is not None
            n2 = _n2[start:stop].astype(np.float64)
            scored = energies.astype(np.float64) + cfg.geo_weight * n2 * q_scale
            cand = np.flatnonzero(scored < cur_e)
            if cand.size > 1:
                idx = start + int(cand[np.argmin(scored[cand])])
            elif cand.size == 1:
                idx = start + int(cand[0])
            else:
                continue
        else:
            cand = np.flatnonzero(energies < cur_e)
            if not cand.size:
                continue
            idx = start + int(cand[0])

        s, c = positions[idx]
        prev_e = cur_e
        cur_e = tracker.accept(cur_seq, s, c)
        stats.energy_saved_singles += prev_e - cur_e
        stats.hit_single()
        if cur_e == 0:
            stats.record_solve("greedy", prev_e // q_scale)
        improved = True
        scan_count = idx + 1
        break

    used = scan_count
    stats.single_evals += used
    return improved, cur_e, used


def _tabu_phase(
    cur_seq: Int8Array,
    tracker: Tracker,
    cur_e: int,
    rng: np.random.Generator,
    cfg: SolverConfig,
    stats: SearchStats,
) -> tuple[bool, int, int]:
    """Tabu walk. Tracks success per Q-level. Returns (improved, new_e, evals)."""
    n_cols = cur_seq.shape[1]
    t0 = time.perf_counter()
    q_start = cur_e // (64 * n_cols)
    prev_e = cur_e
    result = _tabu_walk(cur_seq, tracker, cur_e, rng, cfg)
    stats.tabu_time_s += time.perf_counter() - t0
    stats.tabu_walks += 1
    stats.tabu_evals += result.steps
    stats.tabu_candidate_evals += result.steps * cur_seq.size
    if q_start == 1:
        stats.tabu_walks_q1 += 1
    elif q_start == 2:
        stats.tabu_walks_q2 += 1
    else:
        stats.tabu_walks_q3plus += 1
    if result.energy is not None:
        stats.energy_saved_tabu += prev_e - result.energy
        stats.tabu_hits += 1
        if q_start == 1:
            stats.tabu_hits_q1 += 1
        elif q_start == 2:
            stats.tabu_hits_q2 += 1
        else:
            stats.tabu_hits_q3plus += 1
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
        return True, result.energy, result.steps
    return False, cur_e, result.steps


def _targeted_kick(
    cur_seq: Int8Array,
    tracker: Tracker,
    rng: np.random.Generator,
    budget: int,
    best_seq: Int8Array,
    best_e: int,
    cfg: SolverConfig,
    stats: SearchStats,
) -> TargetedKickResult:
    """Quench candidates spread across all feasible residual lags."""
    n = cur_seq.shape[1]
    assert tracker._delta is not None and tracker._u is not None
    _d = tracker._delta
    _u = tracker._u
    candidates = _targeted_candidates(_d, _u, n)

    if not candidates:
        return TargetedKickResult(False, cur_seq, tracker.energy(), best_seq, best_e, 0)
    quench_budget = min(cfg.escape_quench_steps, max(1, budget // 4))
    total_used = 0
    quench_cfg = SolverConfig(targeted_escape=False)

    for c0, c1, c2, c3 in candidates:
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
        quench = search(
            cand,
            t2,
            rng,
            steps=quench_budget,
            config=quench_cfg,
        )
        total_used += quench.steps
        stats.target_quenches += 1
        stats.quench_candidate_evals += quench.stats.total_candidate_evals
        stats.kicks += 1
        stats.kick_evals += 1
        if quench.energy < best_e:
            best_seq, best_e = quench.sequences.copy(), quench.energy
        if quench.solved:
            phase = quench.stats.solve_phase or "unknown"
            q_before = quench.stats.solve_q_before if quench.stats.solve_q_before is not None else 0
            stats.record_solve(f"targeted:{phase}", q_before)
            return TargetedKickResult(True, quench.sequences, 0, best_seq, best_e, total_used)

    return TargetedKickResult(False, cur_seq, tracker.energy(), best_seq, best_e, total_used)


def _random_kick(
    cur_seq: Int8Array,
    tracker: Tracker,
    rng: np.random.Generator,
) -> int:
    """Random kick: flip two randomly selected sequences."""
    n_seqs, n_cols = cur_seq.shape
    n_flips = 2  # small kicks keep state near C-rich band
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
    steps: int,
    config: SolverConfig | None = None,
) -> SolverResult:
    """Run iterated local search and return the best state found."""
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
    steps -= 1

    total_budget = steps
    best_seq, best_e = cur_seq.copy(), cur_e
    if cur_e == 0:
        stats.record_solve("initial", 0)
    lowest_q_seen = cur_e // q_scale  # niedrigstes Q bisher
    times_at_lowest = 0  # wie oft schon beim niedrigsten Q gestuckt
    early_escape_done = False

    while steps > 0 and best_e > 0:
        # Phase 1: Greedy descent — stop at QWindow (don't grind to floor)
        t_phase = time.perf_counter()
        improved, cur_e, used = _greedy_descent(
            cur_seq, tracker, positions, cur_e, steps, q_scale, cfg, stats
        )
        steps -= used
        stats.single_time_s += time.perf_counter() - t_phase

        if steps <= 0:
            best_seq, best_e = _update_best(cur_seq, cur_e, best_seq, best_e)
            break

        # QWindow: if greedy hit the C-rich band, stop further descent
        q_now = cur_e // q_scale
        if cfg.qwindow_high > 0 and q_now <= cfg.qwindow_high:
            improved = False  # force tabu/escape instead of more greedy

        # Phase 2: Tabu (if stuck)
        if not improved and cfg.tabu:
            improved, cur_e, _ = _tabu_phase(cur_seq, tracker, cur_e, rng, cfg, stats)
            best_seq, best_e = _update_best(cur_seq, cur_e, best_seq, best_e)

        # Phase 3: Kick (if still stuck)
        if not improved and steps > 0 and cur_e > 0 and cfg.kick:
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
                targeted = _targeted_kick(
                    cur_seq, tracker, rng, total_budget, best_seq, best_e, cfg, stats
                )
                improved = targeted.solved
                cur_seq = targeted.sequences
                cur_e = targeted.energy
                best_seq = targeted.best_sequences
                best_e = targeted.best_energy
                steps -= targeted.steps
            else:
                cur_e = _random_kick(cur_seq, tracker, rng)
                improved = True
                steps -= 1
                stats.kick_evals += 1
                if cur_e == 0:
                    stats.record_solve("random_kick", prev_e // q_scale)

            stats.kicks += 1
            stats.energy_saved_kicks += prev_e - cur_e
            stats.kick_time_s += time.perf_counter() - t_kick
            if not improved:
                stats.hit_other()

        best_seq, best_e = _update_best(cur_seq, cur_e, best_seq, best_e)

    return SolverResult(best_seq, best_e, total_budget - max(steps, 0), stats)


# --- Tabu Walk Helper ---------------------------------------------------------


def _tabu_walk(
    cur_seq: Int8Array,
    tracker: Tracker,
    cur_e: int,
    rng: np.random.Generator,
    config: SolverConfig,
) -> TabuWalkResult:
    """Explore uphill single flips and retain only the best visited state."""
    if config.tabu_steps <= 0:
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
    noise = config.tabu_noise * rng.random((config.tabu_steps, n_seqs, n_cols))
    best_seq, best_delta, best_norm2, best_u, best_q, evaluations, solve_q_before = (
        tabu_walk_kernel(
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
        )
    )
    best_e = 64 * n_cols * best_q
    if best_e >= cur_e:
        return TabuWalkResult(None, evaluations)

    cur_seq[...] = best_seq
    tracker._adopt(best_seq, best_u, best_q, best_delta, best_norm2)
    return TabuWalkResult(
        best_e,
        evaluations,
        solve_q_before if solve_q_before >= 0 else None,
    )
