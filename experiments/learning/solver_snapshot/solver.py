"""Iterated local search — greedy singles, Tabu walk, targeted/random kick.

Pipeline: Greedy -> Tabu -> Kick (targeted + Quench bei Q<=3, sonst random).
Targeted-Kick = 1 neg-Flip/Seq am dominanten Lag + Quench.
Quench-Budget = max(5000, original_budget // (q_now + 1))."""

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
import numpy.typing as npt
from numba import njit  # pyright: ignore[reportMissingImports]

from .benchmark_stats import fmt_e
from .tracker import Tracker

SINGLE_BATCH_SIZE = 64
_jit: Any = njit


# --- Numba Tabu Kernel --------------------------------------------------------


@_jit(cache=True)
def tabu_walk_kernel(
    seqs: npt.NDArray[np.int8],
    delta: npt.NDArray[np.int8],
    norm2: npt.NDArray[np.int32],
    u: npt.NDArray[np.int32],
    q: int,
    update_cols: npt.NDArray[np.intp],
    update_lags: npt.NDArray[np.intp],
    update_signs: npt.NDArray[np.int8],
    noise: npt.NDArray[np.float64],
    tenure: float,
    decay: float,
) -> tuple[
    npt.NDArray[np.int8],
    npt.NDArray[np.int8],
    npt.NDArray[np.int32],
    npt.NDArray[np.int32],
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
                break

    return best_seq, best_delta, best_norm2, best_u, best_q, used


# --- Config & Helpers ---------------------------------------------------------


@dataclass
class SolverConfig:
    kick: bool = True
    tabu: bool = True
    tabu_steps: int = 200
    tabu_tenure: float = 5.0
    tabu_decay: float = 0.7
    tabu_noise: float = 0.0
    geo_weight: float = 0.0
    targeted_escape: bool = True


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


class SearchStats:
    """Hit counters + streak histogram + phase timing + Q-level tracking."""

    __slots__ = (
        "_streak",
        "energy_saved_kicks",
        "energy_saved_singles",
        "energy_saved_tabu",
        "kick_evals",
        "kick_time_s",
        "kicks",
        "rebuild_time_s",
        "single_evals",
        "single_time_s",
        "singles",
        "singles_streaks",
        "tabu_evals",
        "tabu_hits",
        "tabu_hits_q1",
        "tabu_hits_q2",
        "tabu_hits_q3plus",
        "tabu_time_s",
        "tabu_walks",
        "tabu_walks_q1",
        "tabu_walks_q2",
        "tabu_walks_q3plus",
    )

    def __init__(self) -> None:
        self.singles: int = 0
        self.kicks: int = 0
        self.single_evals: int = 0
        self.kick_evals: int = 0
        self.tabu_evals: int = 0
        self.tabu_hits: int = 0
        self.tabu_hits_q1: int = 0
        self.tabu_hits_q2: int = 0
        self.tabu_hits_q3plus: int = 0
        self.tabu_walks: int = 0
        self.tabu_walks_q1: int = 0
        self.tabu_walks_q2: int = 0
        self.tabu_walks_q3plus: int = 0
        self.singles_streaks: list[int] = []
        self._streak: int = 0
        self.energy_saved_singles: int = 0
        self.energy_saved_kicks: int = 0
        self.energy_saved_tabu: int = 0
        self.single_time_s: float = 0.0
        self.kick_time_s: float = 0.0
        self.rebuild_time_s: float = 0.0
        self.tabu_time_s: float = 0.0

    def _hit_single(self) -> None:
        self.singles += 1
        self._streak += 1

    def _hit_other(self) -> None:
        if self._streak > 0:
            self.singles_streaks.append(self._streak)
            self._streak = 0

    def _flush(self) -> None:
        if self._streak > 0:
            self.singles_streaks.append(self._streak)
            self._streak = 0

    def to_dict(self) -> dict[str, object]:
        """Full stats for JSON persistence."""
        self._flush()
        return {
            "singles": self.singles,
            "kicks": self.kicks,
            "e_singles": self.energy_saved_singles,
            "e_kicks": self.energy_saved_kicks,
            "e_tabu": self.energy_saved_tabu,
            "single_evals": self.single_evals,
            "kick_evals": self.kick_evals,
            "tabu_evals": self.tabu_evals,
            "tabu_hits": self.tabu_hits,
            "tabu_walks": self.tabu_walks,
            "tabu_hits_q1": self.tabu_hits_q1,
            "tabu_hits_q2": self.tabu_hits_q2,
            "tabu_hits_q3plus": self.tabu_hits_q3plus,
            "tabu_walks_q1": self.tabu_walks_q1,
            "tabu_walks_q2": self.tabu_walks_q2,
            "tabu_walks_q3plus": self.tabu_walks_q3plus,
            "single_time_s": self.single_time_s,
            "kick_time_s": self.kick_time_s,
            "rebuild_time_s": self.rebuild_time_s,
            "tabu_time_s": self.tabu_time_s,
        }

    def display(self) -> str:
        """Compact one-line summary for CLI output."""
        self._flush()
        parts = [
            f"S={self.singles}({fmt_e(self.energy_saved_singles)})",
            f"TB={self.tabu_hits}/{self.tabu_walks}({fmt_e(self.energy_saved_tabu)})",
            f"K={self.kicks}({fmt_e(self.energy_saved_kicks)})",
            f"evals={self.single_evals}/{self.tabu_evals}/{self.kick_evals}",
            f"t={self.single_time_s:.1f}s/{self.tabu_time_s:.1f}s/{self.kick_time_s:.1f}s",
        ]
        if self.singles_streaks:
            s = self.singles_streaks
            parts.append(f"strk={sum(s) / len(s):.1f}/{max(s)}")
        else:
            parts.append("strk=-")
        return " ".join(parts)


# --- Core Phases --------------------------------------------------------------


def _greedy_descent(
    cur_seq: npt.NDArray[np.int8],
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
        stats._hit_single()
        improved = True
        scan_count = idx + 1
        break

    used = scan_count
    stats.single_evals += used
    return improved, cur_e, used


def _tabu_phase(
    cur_seq: npt.NDArray[np.int8],
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
    result, evaluations = _tabu_walk(cur_seq, tracker, cur_e, rng, cfg)
    stats.tabu_time_s += time.perf_counter() - t0
    stats.tabu_walks += 1
    stats.tabu_evals += evaluations
    if q_start == 1:
        stats.tabu_walks_q1 += 1
    elif q_start == 2:
        stats.tabu_walks_q2 += 1
    else:
        stats.tabu_walks_q3plus += 1
    if result is not None:
        stats.energy_saved_tabu += prev_e - result
        stats.tabu_hits += 1
        if q_start == 1:
            stats.tabu_hits_q1 += 1
        elif q_start == 2:
            stats.tabu_hits_q2 += 1
        else:
            stats.tabu_hits_q3plus += 1
        stats._hit_other()
        return True, result, evaluations
    return False, cur_e, evaluations


def _targeted_kick(
    cur_seq: npt.NDArray[np.int8],
    tracker: Tracker,
    rng: np.random.Generator,
    q_now: int,
    budget: int,
    best_seq: npt.NDArray[np.int8],
    best_e: int,
    stats: SearchStats,
) -> tuple[bool, npt.NDArray[np.int8], int, npt.NDArray[np.int8], int, int]:
    """Targeted escape: 1 neg-flip/Seq + Quench. Nur wenige Kombos testen."""
    n_cols = cur_seq.shape[1]
    m = (n_cols - 1) // 2
    assert tracker._delta is not None and tracker._u is not None
    _d = tracker._delta[:, :m]
    k = int(np.argmax(np.abs(tracker._u)))
    target = -tracker._u[k]

    neg: list[list[int]] = [[] for _ in range(4)]
    for s in range(4):
        for c in range(n_cols):
            if _d[s * n_cols + c, k] == target:
                neg[s].append(c)
    if not all(neg):
        return False, cur_seq, tracker.energy(), best_seq, best_e, 0

    max_p = 5
    n_arr = [np.array(ng[:max_p], dtype=np.int64) for ng in neg]
    quench_budget = min(10000, budget // 4)
    total_used = 0

    # Base-Tracker: einmal builden, dann per _adopt kopieren (3.3x schneller)
    base_tr = Tracker()
    base_tr.build(cur_seq)

    for i0 in range(len(n_arr[0])):
        for i1 in range(len(n_arr[1])):
            for i2 in range(len(n_arr[2])):
                for i3 in range(len(n_arr[3])):
                    c0 = int(n_arr[0][i0]) % n_cols
                    c1 = int(n_arr[1][i1]) % n_cols
                    c2 = int(n_arr[2][i2]) % n_cols
                    c3 = int(n_arr[3][i3]) % n_cols
                    cand = cur_seq.copy()
                    cand[0, c0] *= -1
                    cand[1, c1] *= -1
                    cand[2, c2] *= -1
                    cand[3, c3] *= -1
                    t2 = Tracker()
                    t2._n = base_tr._n
                    t2._seqs = cand
                    t2._u = base_tr._u.copy()
                    t2._q = base_tr._q
                    t2._delta = base_tr._delta.copy()
                    t2._norm2 = base_tr._norm2.copy()
                    t2._e = base_tr._e
                    t2._update_cols = base_tr._update_cols
                    t2._update_lags = base_tr._update_lags
                    t2._update_signs = base_tr._update_signs
                    t2.accept(cand, 0, c0)
                    t2.accept(cand, 1, c1)
                    t2.accept(cand, 2, c2)
                    t2.accept(cand, 3, c3)
                    sol, be, used, _ = search(
                        cand,
                        t2,
                        rng,
                        steps=quench_budget,
                        config=SolverConfig(targeted_escape=False),
                    )
                    total_used += used
                    stats.kicks += 1
                    stats.kick_evals += 1
                    if be < best_e:
                        best_seq, best_e = sol.copy(), be
                    if be == 0:
                        return True, sol, 0, best_seq, best_e, total_used

    return False, cur_seq, tracker.energy(), best_seq, best_e, total_used


def _random_kick(
    cur_seq: npt.NDArray[np.int8],
    tracker: Tracker,
    rng: np.random.Generator,
) -> int:
    """4-bit random kick. Returns new energy."""
    n_seqs, n_cols = cur_seq.shape
    cols = np.asarray(rng.integers(0, n_cols, size=n_seqs), dtype=np.intp)
    cur_e = 0
    for s in range(n_seqs):
        cur_e = tracker.accept(cur_seq, s, int(cols[s]))
    return cur_e


# --- Main Solver --------------------------------------------------------------


def search(
    seqs: npt.NDArray[np.int8],
    tracker: Tracker,
    rng: np.random.Generator,
    *,
    steps: int,
    config: SolverConfig | None = None,
) -> tuple[npt.NDArray[np.int8], int, int, SearchStats]:
    """Iterated local search. Returns (best_seq, best_energy, evals, stats)."""
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
    lowest_q_seen = cur_e // q_scale  # niedrigstes Q bisher
    times_at_lowest = 0  # wie oft schon beim niedrigsten Q gestuckt
    early_escape_done = False

    while steps > 0 and best_e > 0:
        # Phase 1: Greedy descent
        t_phase = time.perf_counter()
        improved, cur_e, used = _greedy_descent(
            cur_seq, tracker, positions, cur_e, steps, q_scale, cfg, stats
        )
        steps -= used
        stats.single_time_s += time.perf_counter() - t_phase

        if steps <= 0:
            best_seq, best_e = _update_best(cur_seq, cur_e, best_seq, best_e)
            break

        # Phase 2: Tabu (if stuck)
        if not improved and cfg.tabu:
            improved, cur_e, _ = _tabu_phase(cur_seq, tracker, cur_e, rng, cfg, stats)

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
                improved, cur_seq, cur_e, best_seq, best_e, kick_used = _targeted_kick(
                    cur_seq, tracker, rng, q_now, total_budget, best_seq, best_e, stats
                )
                steps -= kick_used
            else:
                cur_e = _random_kick(cur_seq, tracker, rng)
                improved = True
                steps -= 1
                stats.kick_evals += 1

            stats.kicks += 1
            stats.energy_saved_kicks += prev_e - cur_e
            stats.kick_time_s += time.perf_counter() - t_kick
            if not improved:
                stats._hit_other()

        best_seq, best_e = _update_best(cur_seq, cur_e, best_seq, best_e)

    return best_seq, best_e, total_budget - max(steps, 0), stats


# --- Tabu Walk Helper ---------------------------------------------------------


def _tabu_walk(
    cur_seq: npt.NDArray[np.int8],
    tracker: Tracker,
    cur_e: int,
    rng: np.random.Generator,
    config: SolverConfig,
) -> tuple[int | None, int]:
    """Explore uphill single flips and retain only the best visited state."""
    if config.tabu_steps <= 0:
        return None, 0

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
    best_seq, best_delta, best_norm2, best_u, best_q, evaluations = tabu_walk_kernel(
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
    best_e = 64 * n_cols * best_q
    if best_e >= cur_e:
        return None, evaluations

    cur_seq[...] = best_seq
    tracker._adopt(best_seq, best_u, best_q, best_delta, best_norm2)
    return best_e, evaluations
