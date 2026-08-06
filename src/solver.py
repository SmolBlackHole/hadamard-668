"""Iterated local search — singles scan + pair rescue + kick."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import combinations

import numpy as np
import numpy.typing as npt

from tracker import Tracker


@dataclass
class SolverConfig:
    """Feature flags for ablation testing.  Default: proven combination."""

    pairs: bool = True
    triples: bool = False
    kick: bool = True
    restart: bool = False
    rescue_mode: bool = False


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
        "kicks",
        "pairs",
        "restarts",
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
        self.singles_streaks: list[int] = []
        self._streak: int = 0
        self.energy_saved_singles: int = 0
        self.energy_saved_pairs: int = 0
        self.energy_saved_triples: int = 0
        self.energy_saved_kicks: int = 0

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
    _rows_band, _cols_band = tracker._rows_band, tracker._cols_band
    tracker.build(cur_seq, band_rows=_rows_band, band_cols=_cols_band)
    cur_e = tracker.energy()
    steps -= 1
    total_budget = steps
    iters: int = 0

    best_seq = cur_seq.copy()
    best_e = cur_e

    K = min(B - 1, max(16, int(B**0.5 * 3)))
    KickStrikeMax = 3  # ponytail: constant, tune later

    singles_e = np.empty(B)
    top: npt.NDArray[np.int64] = np.empty(0, dtype=np.int64)
    top_candidates: list[tuple[int, int]] = []
    kick_strike: int = 0
    while steps > 0 and best_e > 0:
        iters += 1

        # Phase 1: greedy singles scan
        singles_e.fill(np.inf)
        improved = False
        for idx, (s, c) in enumerate(positions):
            if steps <= 0:
                break
            e = tracker.flip(cur_seq, s, c)
            steps -= 1
            singles_e[idx] = e
            if e < cur_e:
                prev_e = cur_e
                cur_e = tracker.accept(cur_seq, s, c)
                stats.energy_saved_singles += prev_e - cur_e
                improved = True
                stats._hit_single()
                break

        if steps <= 0:
            break

        if not improved:
            top = np.argpartition(singles_e, K)[:K]
            top_candidates = [positions[t] for t in top]

            # 2-bit rescue
            if cfg.pairs:
                prev_e = cur_e
                result = _rescue(cur_seq, tracker, top_candidates, 2, cur_e, mode="first")
                if result is not None:
                    stats.energy_saved_pairs += prev_e - result
                    cur_e = result
                    improved = True
                    stats.pairs += 1
                    stats._hit_other()

        # Phase 3: Kick — always accept, reset strike on success
        if not improved and steps > 0 and cur_e > 0 and cfg.kick:
            cols = rng.integers(0, n_cols, size=n_seqs)
            for s in range(n_seqs):
                cur_seq[s, int(cols[s])] *= -1
            tracker.build(cur_seq, band_rows=_rows_band, band_cols=_cols_band)
            cur_e = tracker.energy()
            kick_strike += 1
            if kick_strike >= KickStrikeMax and cfg.restart:
                for s in range(n_seqs):
                    cur_seq[s] = rng.choice(np.array([-1, 1], dtype=np.int8), size=n_cols)
                tracker.build(cur_seq, band_rows=_rows_band, band_cols=_cols_band)
                cur_e = tracker.energy()
                kick_strike = 0
                stats.restarts += 1
                stats._hit_other()
            steps -= 1
            stats.kicks += 1
            stats._hit_other()

        best_seq, best_e = _update_best(cur_seq, cur_e, best_seq, best_e)

    return best_seq, best_e, total_budget - max(steps, 0), stats


def _rescue(
    cur_seq: npt.NDArray[np.int8],
    tracker: Tracker,
    candidates: list[tuple[int, int]],
    width: int,
    cur_e: int,
    *,
    mode: str = "first",
    narrowed: list[tuple[int, int]] | None = None,
) -> int | None:
    """Try width-bit combinations. Returns improved energy or None."""
    band_rows = tracker._rows_band
    band_cols = tracker._cols_band
    native = hasattr(tracker, "_combo_delta_native")
    if not native and (not band_rows or not band_cols):
        return None

    pool = narrowed if narrowed is not None else candidates
    best_e = cur_e
    best_combo: tuple | None = None

    for combo in combinations(pool, width):
        if native:
            e = tracker._combo_delta_native(list(combo))  # type: ignore[reportGeneralTypeIssues]
        else:
            e = tracker._combo_delta([(band_rows[s][c], band_cols[s][c]) for s, c in combo])  # type: ignore[reportGeneralTypeIssues]
        if e < cur_e:
            if mode == "first":
                for s, c in combo:
                    cur_seq[s, c] *= -1
                tracker.build(cur_seq, band_rows=band_rows, band_cols=band_cols)
                return tracker.energy()
            if e < best_e:
                best_e = e
                best_combo = combo

    if best_combo is None:
        return None

    for s, c in best_combo:
        cur_seq[s, c] *= -1
    tracker.build(cur_seq, band_rows=band_rows, band_cols=band_cols)
    return tracker.energy()
