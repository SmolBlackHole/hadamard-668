"""Navigator baseline — solver.py + SeqCache."""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass
from functools import lru_cache
from itertools import product
from typing import Any

import numpy as np
import numpy.typing as npt
from numba import njit  # pyright: ignore[reportMissingImports]

from .benchmark_stats import fmt_e
from .tracker import Tracker

SINGLE_BATCH_SIZE = 64
_jit: Any = njit


# ---------------------------------------------------------------------------
# SeqCache
# ---------------------------------------------------------------------------


class SeqCache:
    """O(1) lookup: bytes(seq) -> Q. Speichert alle besuchten Zustaende."""

    def __init__(self, capacity: int = 50000) -> None:
        self._data: dict[bytes, int] = {}
        self._cap = capacity
        self.hits = 0
        self.misses = 0

    def put(self, seqs: npt.NDArray[np.int8], q: int) -> None:
        if len(self._data) >= self._cap:
            self._data.pop(next(iter(self._data)))
        self._data[seqs.tobytes()] = q

    def get(self, seqs: npt.NDArray[np.int8]) -> int | None:
        v = self._data.get(seqs.tobytes())
        if v is not None:
            self.hits += 1
        else:
            self.misses += 1
        return v

    def __len__(self) -> int:
        return len(self._data)

    def display(self) -> str:
        return f"cache={len(self._data)} hits={self.hits} miss={self.misses}"


class BasinCache:
    """O(1) lookup: basin_fingerprint -> best_Q bei letztem targeted_kick."""

    def __init__(self) -> None:
        self._data: dict[str, int] = {}
        self.hits = 0
        self.skips = 0  # how many targeted_kicks were skipped

    def seen(self, fp: str, current_q: int) -> bool:
        """True if we already kicked from this basin at same or better Q."""
        prev = self._data.get(fp)
        if prev is not None and prev <= current_q:
            self.hits += 1
            return True
        return False

    def record(self, fp: str, best_q: int) -> None:
        prev = self._data.get(fp)
        if prev is None or best_q < prev:
            self._data[fp] = best_q

    def display(self) -> str:
        return f"basins={len(self._data)} basin_hits={self.hits} basin_skips={self.skips}"


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
    tabu_steps: int = 400
    tabu_tenure: float = 5.0
    tabu_decay: float = 0.7
    tabu_noise: float = 0.2
    geo_weight: float = 0.0
    targeted_escape: bool = True
    escape_policy: str = "legacy625"
    escape_quench_steps: int = 10_000
    qwindow_high: int = 9


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


def _support_lag625_candidates(
    delta: npt.NDArray[np.int8], u: npt.NDArray[np.int32], n: int
) -> list[tuple[int, int, int, int]]:
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


class SearchStats:
    __slots__ = (
        "_basin_display",
        "_cache_display",
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
        self._cache_display: str = ""
        self._basin_display: str = ""

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
        if self._cache_display:
            parts.append(self._cache_display)
        if self._basin_display:
            parts.append(self._basin_display)
        return " ".join(parts)


# --- Greedy ----------------------------------------------------------------


def _basin_fp(seqs: npt.NDArray[np.int8]) -> str:
    """blake2b fingerprint des Greedy-Minimums (Basin-ID)."""
    return hashlib.blake2b(seqs.tobytes(), digest_size=16).hexdigest()


def _greedy_descent(
    cur_seq: npt.NDArray[np.int8],
    tracker: Tracker,
    positions: tuple[tuple[int, int], ...],
    cur_e: int,
    steps: int,
    q_scale: int,
    cfg: SolverConfig,
    stats: SearchStats,
    seq_cache: SeqCache | None = None,
) -> tuple[bool, int, int]:
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
        if seq_cache is not None:
            seq_cache.put(cur_seq, tracker._q)
        stats.energy_saved_singles += prev_e - cur_e
        stats._hit_single()
        improved = True
        scan_count = idx + 1
        break

    used = scan_count
    stats.single_evals += used
    return improved, cur_e, used


# --- Tabu -------------------------------------------------------------------


def _tabu_phase(
    cur_seq: npt.NDArray[np.int8],
    tracker: Tracker,
    cur_e: int,
    rng: np.random.Generator,
    cfg: SolverConfig,
    stats: SearchStats,
) -> tuple[bool, int, int]:
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


# --- Targeted Kick (flat, no recursion) -------------------------------------


def _flat_ils(
    seqs: npt.NDArray[np.int8],
    rng: np.random.Generator,
    budget: int,
    cfg: SolverConfig,
) -> tuple[npt.NDArray[np.int8], int, int]:
    """Flat ILS: Greedy(free) -> Tabu(budget) -> RandomKick(1) -> repeat.

    No recursion. Same pipeline as the original solver, just flat.
    Greedy is free. Tabu steps and kick steps count against budget.
    Returns (best_state, best_Q, budget_used).
    """
    n = seqs.shape[1]
    q_scale = 64 * n
    positions = _positions(4, n)
    noop = SolverConfig(tabu=False, kick=False, targeted_escape=False)

    cur = seqs.copy()
    t = Tracker()
    t.build(cur)
    cur_e = t.energy()
    best_seq, best_e = cur.copy(), cur_e
    used = 0

    # Scan all 4*n positions per Greedy pass
    greedy_scan = 4 * n

    while used < budget and best_e > 0:
        # Greedy (free)
        for _ in range(2000 // SINGLE_BATCH_SIZE + 1):
            improved, cur_e, _ = _greedy_descent(
                cur,
                t,
                positions,
                cur_e,
                greedy_scan,
                q_scale,
                noop,
                SearchStats(),
                None,
            )
            if not improved:
                break
        if cur_e < best_e:
            best_seq, best_e = cur.copy(), cur_e
        if best_e == 0:
            break

        # Tabu (free)
        if cur_e > 0:
            tabu_cfg = SolverConfig(
                tabu=True,
                kick=False,
                targeted_escape=False,
                tabu_steps=cfg.tabu_steps,
                tabu_noise=cfg.tabu_noise,
                tabu_tenure=cfg.tabu_tenure,
                tabu_decay=cfg.tabu_decay,
            )
            improved, cur_e, _ = _tabu_phase(cur, t, cur_e, rng, tabu_cfg, SearchStats())
            if cur_e < best_e:
                best_seq, best_e = cur.copy(), cur_e
            if best_e == 0:
                break

        # Random kick (1 step, budgeted)
        if cur_e > 0 and used < budget:
            cur_e = _random_kick(cur, t, rng)
            used += 1

    return best_seq, best_e // q_scale, used


def _targeted_kick(
    cur_seq: npt.NDArray[np.int8],
    tracker: Tracker,
    rng: np.random.Generator,
    _q_now: int,
    budget: int,
    best_seq: npt.NDArray[np.int8],
    best_e: int,
    cfg: SolverConfig,
    stats: SearchStats,
    seq_cache: SeqCache | None = None,
) -> tuple[bool, npt.NDArray[np.int8], int, npt.NDArray[np.int8], int, int]:
    """Flat targeted kick: probe all candidates with Greedy, sort by Q, then process
    sequentially. Each candidate gets a fair budget share so ALL 625 get tested.

    No recursion, no round-robin complexity. Greedy is free, Tabu+Kick count.
    """
    n = cur_seq.shape[1]
    assert tracker._delta is not None and tracker._u is not None
    candidates = _support_lag625_candidates(tracker._delta, tracker._u, n)
    if not candidates:
        return False, cur_seq, tracker.energy(), best_seq, best_e, 0

    # Phase 1: Probe with Greedy (free)
    scored: list[tuple[int, int, int, int, int]] = []
    for c0, c1, c2, c3 in candidates:
        cand = cur_seq.copy()
        cand[0, c0 % n] *= -1
        cand[1, c1 % n] *= -1
        cand[2, c2 % n] *= -1
        cand[3, c3 % n] *= -1
        _, probe_q, _ = _flat_ils(cand, rng, 0, cfg)  # Greedy only
        scored.append((probe_q, c0 % n, c1 % n, c2 % n, c3 % n))

    scored.sort(key=lambda x: x[0])

    # Phase 2: Process candidates, fair budget share each
    total_used = 0
    improved = False
    per_candidate = min(5000, max(200, budget // 40))

    for _probe_q, c0, c1, c2, c3 in scored:
        if total_used >= budget:
            break

        cand = cur_seq.copy()
        cand[0, c0] *= -1
        cand[1, c1] *= -1
        cand[2, c2] *= -1
        cand[3, c3] *= -1

        remaining = min(per_candidate, budget - total_used)
        quenched, q_result, used = _flat_ils(cand, rng, remaining, cfg)
        total_used += used
        stats.kicks += 1
        stats.kick_evals += 1

        result_e = q_result * (64 * n)
        if seq_cache is not None:
            seq_cache.put(quenched, q_result)
        if result_e < best_e:
            best_seq, best_e = quenched.copy(), result_e
            improved = True
        if best_e == 0:
            return True, quenched, 0, best_seq, best_e, total_used

    return improved, cur_seq, tracker.energy(), best_seq, best_e, total_used


def _random_kick(
    cur_seq: npt.NDArray[np.int8],
    tracker: Tracker,
    rng: np.random.Generator,
) -> int:
    n_seqs, n_cols = cur_seq.shape
    n_flips = 2
    cols = np.asarray(rng.integers(0, n_cols, size=n_seqs), dtype=np.intp)
    cur_e = 0
    for s in rng.choice(n_seqs, size=min(n_flips, n_seqs), replace=False):
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
    cfg = config if config is not None else SolverConfig()
    n_seqs, n_cols = seqs.shape
    positions = _positions(n_seqs, n_cols)
    stats = SearchStats()
    q_scale = 64 * n_cols
    seq_cache = SeqCache()
    basin_cache = BasinCache()

    cur_seq = seqs.copy()
    t0 = time.perf_counter()
    tracker.build(cur_seq)
    stats.rebuild_time_s += time.perf_counter() - t0
    seq_cache.put(cur_seq, tracker._q)
    cur_e = tracker.energy()
    steps -= 1

    total_budget = steps
    best_seq, best_e = cur_seq.copy(), cur_e
    lowest_q_seen = cur_e // q_scale
    times_at_lowest = 0

    while steps > 0 and best_e > 0:
        t_phase = time.perf_counter()
        improved, cur_e, used = _greedy_descent(
            cur_seq,
            tracker,
            positions,
            cur_e,
            steps,
            q_scale,
            cfg,
            stats,
            seq_cache,
        )
        steps -= used
        stats.single_time_s += time.perf_counter() - t_phase

        if steps <= 0:
            best_seq, best_e = _update_best(cur_seq, cur_e, best_seq, best_e)
            break

        q_now = cur_e // q_scale
        if cfg.qwindow_high > 0 and q_now <= cfg.qwindow_high:
            improved = False

        if not improved and cfg.tabu:
            improved, cur_e, _ = _tabu_phase(cur_seq, tracker, cur_e, rng, cfg, stats)
            best_seq, best_e = _update_best(cur_seq, cur_e, best_seq, best_e)

        if not improved and steps > 0 and cur_e > 0 and cfg.kick:
            t_kick = time.perf_counter()
            prev_e = cur_e
            q_now = cur_e // q_scale

            if q_now < lowest_q_seen:
                lowest_q_seen = q_now
                times_at_lowest = 0
            elif q_now == lowest_q_seen:
                times_at_lowest += 1

            if cfg.targeted_escape and q_now == lowest_q_seen and times_at_lowest >= 2:
                times_at_lowest = 0
                # Basin cache: skip if we already kicked from this basin at same Q
                basin_fp = _basin_fp(cur_seq)
                if basin_cache.seen(basin_fp, q_now):
                    basin_cache.skips += 1
                    cur_e = _random_kick(cur_seq, tracker, rng)
                    improved = True
                    steps -= 1
                    stats.kick_evals += 1
                else:
                    improved, cur_seq, cur_e, best_seq, best_e, kick_used = _targeted_kick(
                        cur_seq, tracker, rng, q_now, steps,
                        best_seq, best_e, cfg, stats, seq_cache,
                    )
                    steps -= kick_used
                    basin_cache.record(basin_fp, best_e // q_scale)
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

    stats._cache_display = seq_cache.display()
    stats._basin_display = basin_cache.display()
    return best_seq, best_e, total_budget - steps, stats


# --- Tabu Walk Helper ---------------------------------------------------------


def _tabu_walk(
    cur_seq: npt.NDArray[np.int8],
    tracker: Tracker,
    cur_e: int,
    rng: np.random.Generator,
    config: SolverConfig,
) -> tuple[int | None, int]:
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
