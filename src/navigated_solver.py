"""Navigated ILS: support-lag escapes + ILS-tabu-quench.

Loop:
  1. Greedy descent (dynamic: stops when stuck, not fixed steps)
  2. Analyse violated lags -> escape candidates (one column per sequence)
  3. Probe each candidate with greedy depth -> sort by best Q reached
  4. For best candidates: full ILS-tabu-quench -> greedy polish
  5. If deeper basin found, repeat from 1.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

import numpy as np
import numpy.typing as npt

from .solver import SolverConfig, search as ils_search
from .tracker import Tracker

Int8Array = npt.NDArray[np.int8]


# ---------------------------------------------------------------------------
# Dynamic greedy: stops when Q stops decreasing
# ---------------------------------------------------------------------------


def _greedy_min(
    seqs: Int8Array,
    max_steps: int = 800,
    target_q: int | None = None,
) -> tuple[Int8Array, int]:
    """Greedy best-improvement descent.

    Dynamic range: stops when no improving move exists (qs[best] >= current Q).
    Optional early exit if Q drops below target_q.
    """
    cur = seqs.copy()
    t = Tracker()
    t.build(cur)
    n = seqs.shape[1]
    steps = 0
    while steps < max_steps and t._q > 0:
        qs = t.flip_qs()
        best = int(np.argmin(qs))
        if qs[best] >= t._q:  # stuck -> local minimum
            break
        s, c = divmod(best, n)
        t.accept(cur, s, c)
        steps += 1
        if target_q is not None and t._q <= target_q:
            break
    return cur, t.energy()


# ---------------------------------------------------------------------------
# Probe: run greedy and return best Q reached + final energy
# ---------------------------------------------------------------------------


def _probe_depth(seqs: Int8Array, max_steps: int = 80) -> int:
    """Run greedy descent and return best Q reached."""
    _, e = _greedy_min(seqs, max_steps)
    n = seqs.shape[1]
    return e // (64 * n)


# ---------------------------------------------------------------------------
# Support-lag escape candidates
# ---------------------------------------------------------------------------


def _support_lag_escapes(
    tracker: Tracker,
    max_per_lag: int = 5,
    max_total: int = 200,
) -> list[tuple[int, int, int, int]]:
    """For each violated lag k: find columns c where delta[s*n+c, k] == -u[k].

    Returns list of (c0,c1,c2,c3) spread equally across all violated lags.
    """
    n = tracker._n
    d_, u_ = tracker._delta, tracker._u
    if d_ is None or u_ is None:
        return []

    per_lag: list[list[tuple[int, int, int, int]]] = []
    for k in np.flatnonzero(u_):
        tgt = -u_[k]
        cols = [[c for c in range(n) if d_[s * n + c, k] == tgt][:max_per_lag] for s in range(4)]
        if all(cols):
            per_lag.append(list(product(*cols)))
    if not per_lag:
        return []

    base, extra = divmod(max_total, len(per_lag))
    selected: list[tuple[int, int, int, int]] = []
    for i, cands in enumerate(per_lag):
        take = min(len(cands), base + (i < extra))
        for j in range(take):
            selected.append(cands[j * len(cands) // max(take, 1)])
    return selected


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass
class NavigatedConfig:
    escape_quench_steps: int = 5_000
    max_escape_candidates: int = 100
    tabu_steps: int = 200
    max_cycles: int = 20
    probe_steps: int = 80  # dynamic: greedy stops when stuck
    probe_budget: int = 30  # how many candidates to probe per cycle


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def navigated_search(
    seqs: Int8Array,
    rng: np.random.Generator,
    config: NavigatedConfig | None = None,
) -> tuple[Int8Array, int, dict]:
    cfg = config or NavigatedConfig()
    n = seqs.shape[1]
    n64 = 64 * n

    # ILS config for quench: tabu-only, no internal kicks
    ils_cfg = SolverConfig(
        tabu=True,
        kick=False,
        targeted_escape=False,
        tabu_steps=cfg.tabu_steps,
        qwindow_high=9,
    )

    cur, cur_e = _greedy_min(seqs, max_steps=800)
    best, best_e = cur.copy(), cur_e

    attempts = 0
    successes = 0
    probes_done = 0

    for _ in range(cfg.max_cycles):
        if best_e == 0:
            break

        # 1. Generate escape candidates
        t = Tracker()
        t.build(cur)
        escapes = _support_lag_escapes(t, max_total=cfg.max_escape_candidates)

        if not escapes:
            break

        # 2. Probe: quick greedy on each candidate, sort by depth
        probes_done += min(len(escapes), cfg.probe_budget)
        scored: list[tuple[int, tuple[int, int, int, int]]] = []
        for c0, c1, c2, c3 in escapes[: cfg.probe_budget]:
            candidate = cur.copy()
            candidate[0, c0 % n] *= -1
            candidate[1, c1 % n] *= -1
            candidate[2, c2 % n] *= -1
            candidate[3, c3 % n] *= -1
            depth_q = _probe_depth(candidate, cfg.probe_steps)
            scored.append((depth_q, (c0 % n, c1 % n, c2 % n, c3 % n)))

        # Sort: lowest probe Q first (most promising)
        scored.sort(key=lambda x: x[0])

        # 3. For best candidates: full ILS-quench -> greedy polish
        improved = False
        for probe_q, (c0, c1, c2, c3) in scored:
            candidate = cur.copy()
            candidate[0, c0] *= -1
            candidate[1, c1] *= -1
            candidate[2, c2] *= -1
            candidate[3, c3] *= -1

            # Full ILS-tabu quench
            t2 = Tracker()
            t2.build(candidate)
            q_seq, q_e, _, _ = ils_search(
                candidate.copy(),
                t2,
                rng,
                steps=cfg.escape_quench_steps,
                config=ils_cfg,
            )
            # Greedy polish
            q_seq, q_e = _greedy_min(q_seq, max_steps=800)
            attempts += 1

            if q_e < best_e:
                best, best_e = q_seq.copy(), q_e
                successes += 1
                improved = True
                if best_e == 0:
                    break

            # Stop after first successful escape (re-analyze at new basin)
            if improved:
                break

        if not improved:
            break

        cur = best.copy()
        cur_e = best_e

    return (
        best,
        best_e,
        {
            "attempts": attempts,
            "successes": successes,
            "probes": probes_done,
            "best_q": best_e // n64,
        },
    )
