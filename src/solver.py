"""Iterated local search — singles scan + pair rescue + kick."""

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
TRACE_EVAL_INTERVAL = 100_000
_jit: Any = njit


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


@dataclass
class SolverConfig:
    """Feature flags for ablation testing.  Default: proven combination."""

    pairs: bool = True
    kick: bool = True
    tabu: bool = True
    tabu_steps: int = 200
    tabu_tenure: float = 5.0
    tabu_decay: float = 0.7
    tabu_noise: float = 0.0


@dataclass(frozen=True)
class TraceSnapshot:
    """Exact solver state captured for trajectory analysis."""

    phase: str
    reason: str
    budget_used: int
    evaluations: int
    q: int
    best_q: int
    sequences: npt.NDArray[np.int8]
    residual: npt.NDArray[np.int32]


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
    """Hit counters + streak histogram + phase timing."""

    __slots__ = (
        "_streak",
        "energy_saved_kicks",
        "energy_saved_pairs",
        "energy_saved_singles",
        "energy_saved_tabu",
        "kick_evals",
        "kick_time_s",
        "kicks",
        "pairs",
        "rebuild_time_s",
        "rescue_evals",
        "rescue_time_s",
        "single_evals",
        "single_time_s",
        "singles",
        "singles_streaks",
        "tabu_evals",
        "tabu_hits",
        "tabu_time_s",
        "tabu_walks",
    )

    def __init__(self) -> None:
        self.singles: int = 0
        self.pairs: int = 0
        self.kicks: int = 0
        self.single_evals: int = 0
        self.rescue_evals: int = 0
        self.kick_evals: int = 0
        self.tabu_evals: int = 0
        self.tabu_hits: int = 0
        self.tabu_walks: int = 0
        self.singles_streaks: list[int] = []
        self._streak: int = 0
        self.energy_saved_singles: int = 0
        self.energy_saved_pairs: int = 0
        self.energy_saved_kicks: int = 0
        self.energy_saved_tabu: int = 0
        self.single_time_s: float = 0.0
        self.rescue_time_s: float = 0.0
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
            "pairs": self.pairs,
            "kicks": self.kicks,
            "e_singles": self.energy_saved_singles,
            "e_pairs": self.energy_saved_pairs,
            "e_kicks": self.energy_saved_kicks,
            "e_tabu": self.energy_saved_tabu,
            "single_evals": self.single_evals,
            "rescue_evals": self.rescue_evals,
            "kick_evals": self.kick_evals,
            "tabu_evals": self.tabu_evals,
            "tabu_hits": self.tabu_hits,
            "tabu_walks": self.tabu_walks,
            "single_time_s": self.single_time_s,
            "rescue_time_s": self.rescue_time_s,
            "kick_time_s": self.kick_time_s,
            "rebuild_time_s": self.rebuild_time_s,
            "tabu_time_s": self.tabu_time_s,
        }

    def display(self) -> str:
        """Compact one-line summary for CLI output."""
        self._flush()
        parts = [
            f"S={self.singles}({fmt_e(self.energy_saved_singles)})",
            f"P={self.pairs}({fmt_e(self.energy_saved_pairs)})",
            f"TB={self.tabu_hits}/{self.tabu_walks}({fmt_e(self.energy_saved_tabu)})",
            f"K={self.kicks}({fmt_e(self.energy_saved_kicks)})",
            f"evals={self.single_evals}/{self.rescue_evals}/{self.tabu_evals}/{self.kick_evals}",
            f"t={self.single_time_s:.1f}s/{self.rescue_time_s:.1f}s/{self.tabu_time_s:.1f}s/{self.kick_time_s:.1f}s",
        ]
        if self.singles_streaks:
            s = self.singles_streaks
            parts.append(f"strk={sum(s) / len(s):.1f}/{max(s)}")
        else:
            parts.append("strk=-")
        return " ".join(parts)


def search(
    seqs: npt.NDArray[np.int8],
    tracker: Tracker,
    rng: np.random.Generator,
    *,
    steps: int,
    config: SolverConfig | None = None,
    trace: list[TraceSnapshot] | None = None,
) -> tuple[npt.NDArray[np.int8], int, int, SearchStats]:
    """Iterated local search. Returns (best_seq, best_energy, evals, stats)."""
    cfg = config if config is not None else SolverConfig()
    n_seqs, n_cols = seqs.shape
    positions = _positions(n_seqs, n_cols)
    B = len(positions)
    stats = SearchStats()

    cur_seq = seqs.copy()
    t0 = time.perf_counter()
    tracker.build(cur_seq)
    stats.rebuild_time_s += time.perf_counter() - t0
    cur_e = tracker.energy()
    steps -= 1
    total_budget = steps
    best_seq = cur_seq.copy()
    best_e = cur_e
    next_trace_eval = TRACE_EVAL_INTERVAL

    if trace is not None:
        assert tracker._u is not None
        trace.append(
            TraceSnapshot(
                phase="initial",
                reason="initial",
                budget_used=0,
                evaluations=0,
                q=tracker._q,
                best_q=tracker._q,
                sequences=cur_seq.copy(),
                residual=tracker._u.copy(),
            )
        )

    K = min(B - 1, max(16, int(B**0.5 * 3)))

    singles_e = np.empty(B)
    while steps > 0 and best_e > 0:
        phase = "plateau"
        # Phase 1: greedy singles scan
        t_phase = time.perf_counter()
        singles_e.fill(np.inf)
        scan_count = min(B, steps)
        improved = False
        for start in range(0, scan_count, SINGLE_BATCH_SIZE):
            stop = min(start + SINGLE_BATCH_SIZE, scan_count)
            energies = tracker.flip_batch(start, stop)
            singles_e[start:stop] = energies
            improving = np.flatnonzero(energies < cur_e)
            if improving.size:
                idx = start + int(improving[0])
                s, c = positions[idx]
                prev_e = cur_e
                cur_e = tracker.accept(cur_seq, s, c)
                stats.energy_saved_singles += prev_e - cur_e
                stats._hit_single()
                improved = True
                phase = "single"
                scan_count = idx + 1
                break
        steps -= scan_count
        stats.single_evals += scan_count
        stats.single_time_s += time.perf_counter() - t_phase

        if steps <= 0:
            if trace is not None and cur_e < best_e:
                assert tracker._u is not None
                evaluations = (
                    stats.single_evals + stats.rescue_evals + stats.tabu_evals + stats.kick_evals
                )
                trace.append(
                    TraceSnapshot(
                        phase=phase,
                        reason="best",
                        budget_used=total_budget,
                        evaluations=evaluations,
                        q=tracker._q,
                        best_q=tracker._q,
                        sequences=cur_seq.copy(),
                        residual=tracker._u.copy(),
                    )
                )
            best_seq, best_e = _update_best(cur_seq, cur_e, best_seq, best_e)
            break

        if not improved:
            t_rescue = time.perf_counter()
            top = np.asarray(np.argpartition(singles_e, K - 1)[:K], dtype=np.intp)
            top_candidates: list[tuple[int, int]] = [positions[int(t)] for t in top]

            if cfg.pairs:
                prev_e = cur_e
                result, evaluations = _rescue(cur_seq, tracker, top_candidates, cur_e)
                stats.rescue_evals += evaluations
                if result is not None:
                    stats.energy_saved_pairs += prev_e - result
                    cur_e = result
                    improved = True
                    phase = "pair"
                    stats.pairs += 1
                    stats._hit_other()

            stats.rescue_time_s += time.perf_counter() - t_rescue

        if not improved and cfg.tabu:
            t_tabu = time.perf_counter()
            prev_e = cur_e
            result, evaluations = _tabu_walk(cur_seq, tracker, cur_e, rng, cfg)
            stats.tabu_time_s += time.perf_counter() - t_tabu
            stats.tabu_walks += 1
            stats.tabu_evals += evaluations
            if result is not None:
                stats.energy_saved_tabu += prev_e - result
                cur_e = result
                improved = True
                phase = "tabu"
                stats.tabu_hits += 1
                stats._hit_other()

        # Phase 3: Kick — always accept
        if not improved and steps > 0 and cur_e > 0 and cfg.kick:
            t_kick = time.perf_counter()
            cols = np.asarray(rng.integers(0, n_cols, size=n_seqs), dtype=np.intp)
            prev_e = cur_e
            for s in range(n_seqs):
                cur_e = tracker.accept(cur_seq, s, int(cols[s]))
            steps -= 1
            stats.kicks += 1
            stats.kick_evals += 1
            stats.energy_saved_kicks += prev_e - cur_e
            stats.kick_time_s += time.perf_counter() - t_kick
            stats._hit_other()
            phase = "kick"

        if trace is not None:
            evaluations = (
                stats.single_evals + stats.rescue_evals + stats.tabu_evals + stats.kick_evals
            )
            is_best = cur_e < best_e
            is_sample = evaluations >= next_trace_eval
            if is_best or is_sample:
                assert tracker._u is not None
                trace.append(
                    TraceSnapshot(
                        phase=phase,
                        reason="best" if is_best else "sample",
                        budget_used=total_budget - max(steps, 0),
                        evaluations=evaluations,
                        q=tracker._q,
                        best_q=min(cur_e, best_e) // (64 * n_cols),
                        sequences=cur_seq.copy(),
                        residual=tracker._u.copy(),
                    )
                )
            while evaluations >= next_trace_eval:
                next_trace_eval += TRACE_EVAL_INTERVAL

        best_seq, best_e = _update_best(cur_seq, cur_e, best_seq, best_e)

    return best_seq, best_e, total_budget - max(steps, 0), stats


def _rescue(
    cur_seq: npt.NDArray[np.int8],
    tracker: Tracker,
    candidates: list[tuple[int, int]],
    cur_e: int,
) -> tuple[int | None, int]:
    """Try pair combinations. Returns improved energy or None."""
    energies = tracker.pair_energies(candidates)
    rows, cols = np.triu_indices(len(candidates), 1)
    improving = np.flatnonzero(energies[rows, cols] < cur_e)
    if improving.size == 0:
        return None, len(rows)
    index = int(improving[0])
    combo = (candidates[int(rows[index])], candidates[int(cols[index])])
    for s, c in combo:
        tracker.accept(cur_seq, s, c)
    return tracker.energy(), index + 1


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
