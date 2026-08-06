"""Iterated local search — singles scan + pair rescue + kick."""

from __future__ import annotations

import time
from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations
from math import comb

import numpy as np
import numpy.typing as npt

from tracker import Tracker

SINGLE_BATCH_SIZE = 64


@dataclass
class SolverConfig:
    """Feature flags for ablation testing.  Default: proven combination."""

    pairs: bool = True
    triples: bool = False
    kick: bool = True
    restart: bool = False


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
    """Hit counters + streak histogram.  One place for all solver diagnostics."""

    __slots__ = (
        "_streak",
        "energy_saved_kicks",
        "energy_saved_pairs",
        "energy_saved_singles",
        "energy_saved_triples",
        "kick_evals",
        "kick_time_s",
        "kicks",
        "pairs",
        "rebuild_time_s",
        "rescue_evals",
        "rescue_time_s",
        "restarts",
        "single_evals",
        "single_time_s",
        "singles",
        "singles_streaks",
        "triples",
    )

    def __init__(self) -> None:
        self.singles: int = 0
        self.pairs: int = 0
        self.triples: int = 0
        self.kicks: int = 0
        self.restarts: int = 0
        self.single_evals: int = 0
        self.rescue_evals: int = 0
        self.kick_evals: int = 0
        self.singles_streaks: list[int] = []
        self._streak: int = 0
        self.energy_saved_singles: int = 0
        self.energy_saved_pairs: int = 0
        self.energy_saved_triples: int = 0
        self.energy_saved_kicks: int = 0
        self.single_time_s: float = 0.0
        self.rescue_time_s: float = 0.0
        self.kick_time_s: float = 0.0
        self.rebuild_time_s: float = 0.0

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
            "triples": self.triples,
            "kicks": self.kicks,
            "restarts": self.restarts,
            "e_singles": self.energy_saved_singles,
            "e_pairs": self.energy_saved_pairs,
            "e_triples": self.energy_saved_triples,
            "e_kicks": self.energy_saved_kicks,
            "single_evals": self.single_evals,
            "rescue_evals": self.rescue_evals,
            "kick_evals": self.kick_evals,
            "single_time_s": self.single_time_s,
            "rescue_time_s": self.rescue_time_s,
            "kick_time_s": self.kick_time_s,
            "rebuild_time_s": self.rebuild_time_s,
        }

    def display(self) -> str:
        """Compact one-line summary for CLI output."""
        self._flush()
        parts = [
            f"S={self.singles}({_fmt_e(self.energy_saved_singles)})",
            f"P={self.pairs}({_fmt_e(self.energy_saved_pairs)})",
            f"T={self.triples}({_fmt_e(self.energy_saved_triples)})",
            f"K={self.kicks}({_fmt_e(self.energy_saved_kicks)})",
            f"R={self.restarts}",
            f"evals={self.single_evals}/{self.rescue_evals}/{self.kick_evals}",
            f"t={self.single_time_s:.1f}s/{self.rescue_time_s:.1f}s/{self.kick_time_s:.1f}s",
        ]
        if self.singles_streaks:
            s = self.singles_streaks
            parts.append(f"strk={sum(s) / len(s):.1f}/{max(s)}")
        else:
            parts.append("strk=-")
        return " ".join(parts)


def _fmt_e(e: int) -> str:
    """Compact energy formatting: 1234 -> '1.2k'."""
    if e == 0:
        return "0"
    if e >= 1_000_000:
        return f"{e / 1_000_000:.1f}M"
    if e >= 1000:
        return f"{e / 1000:.0f}k"
    return str(e)


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

    K = min(B - 1, max(16, int(B**0.5 * 3)))
    KickStrikeMax = 3  # ponytail: constant, tune later

    singles_e = np.empty(B)
    top: npt.NDArray[np.int64] = np.empty(0, dtype=np.int64)
    top_candidates: list[tuple[int, int]] = []
    kick_strike: int = 0
    while steps > 0 and best_e > 0:
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
                scan_count = idx + 1
                break
        steps -= scan_count
        stats.single_evals += scan_count
        stats.single_time_s += time.perf_counter() - t_phase

        if steps <= 0:
            break

        if not improved:
            t_rescue = time.perf_counter()
            top = np.argpartition(singles_e, K - 1)[:K]
            top_candidates = [positions[t] for t in top]

            # 2-bit rescue
            if cfg.pairs:
                prev_e = cur_e
                result, evaluations = _rescue(cur_seq, tracker, top_candidates, 2, cur_e)
                stats.rescue_evals += evaluations
                if result is not None:
                    stats.energy_saved_pairs += prev_e - result
                    cur_e = result
                    improved = True
                    stats.pairs += 1
                    stats._hit_other()

            if not improved and cfg.triples:
                triple_top = np.argpartition(singles_e, min(K, 10) - 1)[: min(K, 10)]
                triple_candidates = [positions[t] for t in triple_top]
                prev_e = cur_e
                result, evaluations = _rescue(cur_seq, tracker, triple_candidates, 3, cur_e)
                stats.rescue_evals += evaluations
                if result is not None:
                    stats.energy_saved_triples += prev_e - result
                    cur_e = result
                    improved = True
                    stats.triples += 1
                    stats._hit_other()

            stats.rescue_time_s += time.perf_counter() - t_rescue

        # Phase 3: Kick — always accept, reset strike on success
        if not improved and steps > 0 and cur_e > 0 and cfg.kick:
            t_kick = time.perf_counter()
            cols = rng.integers(0, n_cols, size=n_seqs)
            prev_e = cur_e
            for s in range(n_seqs):
                cur_e = tracker.accept(cur_seq, s, int(cols[s]))
            kick_strike += 1
            if kick_strike >= KickStrikeMax and cfg.restart:
                for s in range(n_seqs):
                    cur_seq[s] = rng.choice(np.array([-1, 1], dtype=np.int8), size=n_cols)
                t0 = time.perf_counter()
                tracker.build(cur_seq)
                stats.rebuild_time_s += time.perf_counter() - t0
                cur_e = tracker.energy()
                kick_strike = 0
                stats.restarts += 1
                stats._hit_other()
            steps -= 1
            stats.kicks += 1
            stats.kick_evals += 1
            stats.energy_saved_kicks += prev_e - cur_e
            stats.kick_time_s += time.perf_counter() - t_kick
            stats._hit_other()

        best_seq, best_e = _update_best(cur_seq, cur_e, best_seq, best_e)

    return best_seq, best_e, total_budget - max(steps, 0), stats


def _rescue(
    cur_seq: npt.NDArray[np.int8],
    tracker: Tracker,
    candidates: list[tuple[int, int]],
    width: int,
    cur_e: int,
) -> tuple[int | None, int]:
    """Try width-bit combinations. Returns improved energy or None."""
    if width == 2:
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

    for evaluations, combo in enumerate(combinations(candidates, width), 1):
        e = tracker.combo_energy(list(combo))
        if e < cur_e:
            for s, c in combo:
                tracker.accept(cur_seq, s, c)
            return tracker.energy(), evaluations
    return None, comb(len(candidates), width) if len(candidates) >= width else 0
