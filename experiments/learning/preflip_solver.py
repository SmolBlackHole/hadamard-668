"""Instrumented solver: captures the IMMEDIATE state BEFORE the solving flip.

General theorem (algebraically checked): a single flip solves (Q=0) iff its delta
vector d equals -u exactly.  This module re-runs the frozen dcfacc5 pipeline
(solver_snapshot) but records, at the exact moment a flip drives Q to 0, the
pre-flip state:

    u_pre, d_solve, Q_pre, supp(u_pre), flip (s, c), solve path.

The solving flip normally happens inside the Tabu kernel (numba), so we copy
the kernel with a pre-flip snapshot.  The frozen snapshot itself is untouched.
"""
from __future__ import annotations

import base64
import sys
import time
from pathlib import Path

import numpy as np
from numba import njit  # pyright: ignore[reportMissingImports]

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from solver_snapshot.solver import (  # noqa: E402
    SearchStats,
    SolverConfig,
    _positions,
    _update_best,
)
from solver_snapshot.tracker import Tracker  # noqa: E402

import common  # noqa: E402

Q_SCALE = 64  # energy per q-unit at any n: E = 64*n*Q, so Q = E // (64n)


# --- Tabu kernel with pre-flip capture ---------------------------------------


@njit(cache=True)
def tabu_walk_kernel_preflip(
    seqs, delta, norm2, u, q,
    update_cols, update_lags, update_signs,
    noise, tenure, decay,
    pre_pos, pre_seq, pre_u, pre_delta,
    q_traj, q_min_pre, deficit_hist,
):
    """Tabu walk; when a step drives q to 0, snapshot the state BEFORE it.

    Returns (best_seq, best_delta, best_norm2, best_u, best_q, used, solved).
    On a solving step, pre_pos = [s, c] of the solving flip, pre_seq = the
    state before the flip, pre_u = u before the flip, pre_delta = the delta
    row of the solving flip before it was applied.  q_traj records the q
    value after every applied flip; q_min_pre gets the running minimum q
    seen before the solving flip (entry q included).

    deficit_hist[step] = C-deficit of the state at walk position step:
    the minimum Q reachable by ONE flip from that state (min over all
    flips of q + delta_q).  0 means the state is on the Cancellation
    Manifold C.
    """
    n_seqs, n_cols = seqs.shape
    n_lags = u.size
    tabu = np.zeros((n_seqs, n_cols), dtype=np.float64)
    best_seq = np.empty_like(seqs)
    best_delta = np.empty_like(delta)
    best_norm2 = np.empty_like(norm2)
    best_u = np.empty_like(u)
    best_q = q
    used = 0
    solved = 0

    for step in range(noise.shape[0]):
        best_score = np.inf
        index = 0
        best_raw_q = q
        for candidate in range(n_seqs * n_cols):
            delta_q = int(norm2[candidate])
            for lag in range(n_lags):
                delta_q += 2 * int(delta[candidate, lag]) * int(u[lag])
            q_after = q + delta_q
            if q_after < best_raw_q:
                best_raw_q = q_after
            s = candidate // n_cols
            c = candidate - s * n_cols
            score = q_after * (1.0 + tabu[s, c] + noise[step, s, c])
            if score < best_score:
                best_score = score
                index = candidate
        deficit_hist[step] = best_raw_q

        s = index // n_cols
        c = index - s * n_cols
        delta_q = int(norm2[index])
        for lag in range(n_lags):
            delta_q += 2 * int(delta[index, lag]) * int(u[lag])

        if q + delta_q == 0 and solved == 0:
            # THIS flip solves: snapshot the pre-flip state.
            pre_pos[0] = s
            pre_pos[1] = c
            for i in range(n_seqs):
                for j in range(n_cols):
                    pre_seq[i, j] = seqs[i, j]
            for lag in range(n_lags):
                pre_u[lag] = u[lag]
                pre_delta[lag] = delta[index, lag]
            q_min_pre[0] = best_q
            solved = 1

        q += delta_q
        q_traj[step] = q
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

    return best_seq, best_delta, best_norm2, best_u, best_q, used, solved


def _tabu_walk_pre(
    cur_seq: np.ndarray,
    tracker: Tracker,
    cur_e: int,
    rng: np.random.Generator,
    config: SolverConfig,
) -> tuple[int | None, int, dict | None]:
    """Tabu walk with pre-flip capture. Returns (best_e, evals, pre_info)."""
    if config.tabu_steps <= 0:
        return None, 0, None

    n_seqs, n_cols = cur_seq.shape
    assert (
        tracker._delta is not None
        and tracker._norm2 is not None
        and tracker._u is not None
        and tracker._update_cols is not None
        and tracker._update_lags is not None
        and tracker._update_signs is not None
    )
    m = tracker._u.size
    noise = config.tabu_noise * rng.random((config.tabu_steps, n_seqs, n_cols))
    pre_pos = np.full(2, -1, dtype=np.int64)
    pre_seq = np.zeros_like(cur_seq)
    pre_u = np.zeros(m, dtype=np.int32)
    pre_delta = np.zeros(m, dtype=np.int8)
    q_traj = np.zeros(config.tabu_steps, dtype=np.int64)
    q_min_pre = np.zeros(1, dtype=np.int64)
    deficit_hist = np.zeros(config.tabu_steps, dtype=np.int64)

    best_seq, best_delta, best_norm2, best_u, best_q, evals, solved = (
        tabu_walk_kernel_preflip(
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
            pre_pos, pre_seq, pre_u, pre_delta,
            q_traj, q_min_pre, deficit_hist,
        )
    )
    best_e = 64 * n_cols * best_q
    walk_info = {
        "entry_q": cur_e // (64 * n_cols),
        "end_q": best_e // (64 * n_cols),
        "used": int(evals),
        "solved": bool(solved),
        "deficit": deficit_hist[: int(evals)].tolist(),
    }
    if best_e < cur_e:
        walk_info["min_q"] = int(best_q)
    pre_info = None
    if solved:
        pre_info = {
            "s": int(pre_pos[0]),
            "c": int(pre_pos[1]),
            "pre_seq": pre_seq.copy(),
            "pre_u": pre_u.copy(),
            "pre_d": pre_delta.copy(),
            "q_pre": int(np.dot(pre_u, pre_u)),
            "walk_entry_q": cur_e // (64 * n_cols),
            "flip_in_walk": int(evals),
            "q_min_before_solve": int(q_min_pre[0]),
            "q_traj": q_traj[: int(evals)].copy(),
        }
    if best_e >= cur_e:
        return None, evals, pre_info, walk_info

    cur_seq[...] = best_seq
    tracker._adopt(best_seq, best_u, best_q, best_delta, best_norm2)
    return best_e, evals, pre_info, walk_info


def _tabu_phase_pre(
    cur_seq: np.ndarray,
    tracker: Tracker,
    cur_e: int,
    rng: np.random.Generator,
    cfg: SolverConfig,
    stats: SearchStats,
    walk_log: list,
) -> tuple[bool, int, int, dict | None]:
    """Tabu phase with pre-flip capture. Returns (improved, new_e, evals, pre)."""
    n_cols = cur_seq.shape[1]
    t0 = time.perf_counter()
    q_start = cur_e // (64 * n_cols)
    prev_e = cur_e
    result, evaluations, pre, walk_info = _tabu_walk_pre(cur_seq, tracker, cur_e, rng, cfg)
    stats.tabu_time_s += time.perf_counter() - t0
    stats.tabu_walks += 1
    stats.tabu_evals += evaluations
    if q_start == 1:
        stats.tabu_walks_q1 += 1
    elif q_start == 2:
        stats.tabu_walks_q2 += 1
    else:
        stats.tabu_walks_q3plus += 1
    if walk_log is not None:
        walk_log.append(walk_info)
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
        return True, result, evaluations, pre
    return False, cur_e, evaluations, pre


# --- Greedy descent / random kick with accepted-flip capture -----------------


def _greedy_descent_pre(
    cur_seq: np.ndarray,
    tracker: Tracker,
    positions: tuple[tuple[int, int], ...],
    cur_e: int,
    steps: int,
    q_scale: int,
    cfg: SolverConfig,
    stats: SearchStats,
) -> tuple[bool, int, int, tuple[int, int] | None]:
    """Greedy descent. Also returns the accepted flip (s, c) or None."""
    B = len(positions)
    scan_count = min(B, steps)
    improved = False
    accepted: tuple[int, int] | None = None

    for start in range(0, scan_count, 64):
        stop = min(start + 64, scan_count)
        energies = tracker.flip_batch(start, stop)
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
        accepted = (s, c)
        scan_count = idx + 1
        break

    used = scan_count
    stats.single_evals += used
    return improved, cur_e, used, accepted


def _random_kick_pre(
    cur_seq: np.ndarray,
    tracker: Tracker,
    rng: np.random.Generator,
) -> tuple[int, list[int]]:
    """4-bit random kick. Returns (new_energy, flipped_columns)."""
    n_seqs, n_cols = cur_seq.shape
    cols = [int(c) for c in rng.integers(0, n_cols, size=n_seqs)]
    cur_e = 0
    for s in range(n_seqs):
        cur_e = tracker.accept(cur_seq, s, cols[s])
    return cur_e, cols


# --- Solving-event capture ----------------------------------------------------


def _record_solve(
    solve_rec: dict,
    path: str,
    pre_seq: np.ndarray,
    flip: tuple[int, int],
    q_pre: int,
    extra: dict | None = None,
) -> None:
    """Fill solve_rec with the pre-flip state + independent d computation."""
    if not solve_rec:
        s, c = flip
        u_pre = common.u_of(pre_seq)
        d = _delta_of(pre_seq, s, c)
        solve_rec.update(
            path=path,
            s=s, c=c,
            q_pre=int(q_pre),
            u_pre=u_pre.tolist(),
            supp=[int(i) for i in np.flatnonzero(u_pre)],
            supp_size=int((u_pre != 0).sum()),
            d=d.tolist(),
            d_equals_minus_u=bool(np.array_equal(d, -u_pre)),
            pre_b64=base64.b64encode(pre_seq.tobytes()).decode(),
        )
        if extra:
            solve_rec.update(extra)


def _delta_of(seqs: np.ndarray, s: int, c: int) -> np.ndarray:
    """Exact delta vector of single flip (s, c) via the closed formula."""
    n = seqs.shape[1]
    m = (n - 1) // 2
    v = seqs.astype(np.int16)
    xc = int(v[s, c])
    f = (c + np.arange(1, m + 1)) % n
    ff = np.where(c + np.arange(1, m + 1) < n, 1, -1)
    b = (c - np.arange(1, m + 1)) % n
    bf = np.where(b + np.arange(1, m + 1) < n, 1, -1)
    d = -2 * xc * (ff * v[s, f] + bf * v[s, b]) // 4
    return d.astype(np.int32)


# --- Targeted escape (quench driver) with solve threading --------------------


def targeted_kick_preflip(
    cur_seq: np.ndarray,
    tracker: Tracker,
    rng: np.random.Generator,
    q_now: int,
    budget: int,
    best_seq: np.ndarray,
    best_e: int,
    stats: SearchStats,
    log: dict,
    solve_rec: dict,
    walk_log: list,
) -> tuple[bool, np.ndarray, int, np.ndarray, int, int]:
    """Replica of the dcfacc5 _targeted_kick with solve_rec threading."""
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
        log.update(k=k, target=target, neg=neg, aborted="empty")
        return False, cur_seq, tracker.energy(), best_seq, best_e, 0

    max_p = 5
    n_arr = [np.array(ng[:max_p], dtype=np.int64) for ng in neg]
    quench_budget = min(10000, budget // 4)
    total_used = 0

    base_tr = Tracker()
    base_tr.build(cur_seq)
    log.update(k=k, target=target, n_arr=n_arr, q_start=tracker._q,
               quench_budget=quench_budget)
    n_combo = 0

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

                    inner_log: dict = {}
                    sol, be, used, _ = search_preflip(
                        cand, t2, rng, steps=quench_budget,
                        config=SolverConfig(targeted_escape=False),
                        log=inner_log, solve_rec=solve_rec, walk_log=walk_log,
                    )
                    total_used += used
                    stats.kicks += 1
                    stats.kick_evals += 1
                    n_combo += 1
                    if be == 0:
                        return True, sol, 0, best_seq, best_e, total_used
                    if be < best_e:
                        best_seq, best_e = sol.copy(), be

    return False, cur_seq, tracker.energy(), best_seq, best_e, total_used


# --- Search loop --------------------------------------------------------------


def search_preflip(
    seqs: np.ndarray,
    tracker: Tracker | None,
    rng: np.random.Generator,
    *,
    steps: int,
    config: SolverConfig | None = None,
    log: dict | None = None,
    solve_rec: dict | None = None,
    walk_log: list | None = None,
) -> tuple[np.ndarray, int, int, SearchStats]:
    """dcfacc5 search loop; when Q reaches 0, fill solve_rec with the
    immediate pre-flip state (u_pre, d_solve, Q_pre, supp).  walk_log
    collects every tabu walk's deficit trajectory (C-proximity per step)."""
    cfg = config if config is not None else SolverConfig()
    n_seqs, n_cols = seqs.shape
    positions = _positions(n_seqs, n_cols)
    stats = SearchStats()
    q_scale = 64 * n_cols
    sr = solve_rec if solve_rec is not None else {}
    wl = walk_log if walk_log is not None else None

    cur_seq = seqs.copy()
    if tracker is None:
        tracker = Tracker()
    t0 = time.perf_counter()
    tracker.build(cur_seq)
    stats.rebuild_time_s += time.perf_counter() - t0
    cur_e = tracker.energy()
    steps -= 1

    total_budget = steps
    best_seq, best_e = cur_seq.copy(), cur_e
    lowest_q_seen = cur_e // q_scale
    times_at_lowest = 0
    early_escape_done = False

    if log is None:
        log = {}
    log.setdefault("q0", cur_e // q_scale)
    log.setdefault("events", [])
    solved_step: int | None = None

    while steps > 0 and best_e > 0:
        t_phase = time.perf_counter()
        improved, cur_e, used, accepted = _greedy_descent_pre(
            cur_seq, tracker, positions, cur_e, steps, q_scale, cfg, stats
        )
        steps -= used
        stats.single_time_s += time.perf_counter() - t_phase
        log["events"].append({"type": "descent", "q": cur_e // q_scale,
                              "steps_left": steps})

        if cur_e == 0 and accepted is not None:
            # greedy descent just solved: reconstruct pre state by flipping back
            s, c = accepted
            pre_seq = cur_seq.copy()
            pre_seq[s, c] *= -1
            _record_solve(sr, "greedy", pre_seq, (s, c),
                          q_pre=int(np.dot(common.u_of(pre_seq), common.u_of(pre_seq))))
            solved_step = total_budget - steps
            best_seq, best_e = cur_seq.copy(), cur_e
            break

        if steps <= 0:
            best_seq, best_e = _update_best(cur_seq, cur_e, best_seq, best_e)
            break

        if not improved and cfg.tabu:
            improved, cur_e, _, pre = _tabu_phase_pre(
                cur_seq, tracker, cur_e, rng, cfg, stats, wl
            )
            best_seq, best_e = _update_best(cur_seq, cur_e, best_seq, best_e)
            log["events"].append({"type": "tabu", "q": cur_e // q_scale,
                                  "steps_left": steps})
            if cur_e == 0 and pre is not None:
                s, c = pre["s"], pre["c"]
                pre_seq = cur_seq.copy()
                pre_seq[s, c] *= -1
                _record_solve(
                    sr, "tabu", pre_seq, (s, c), pre["q_pre"],
                    extra={"walk_entry_q": pre["walk_entry_q"],
                           "flip_in_walk": pre["flip_in_walk"],
                           "q_min_before_solve": pre["q_min_before_solve"],
                           "q_traj": [int(v) for v in pre["q_traj"]],
                           "tabu_pre_seq_match": bool(np.array_equal(
                               pre_seq, pre["pre_seq"]))},
                )
                solved_step = total_budget - steps
                best_seq, best_e = cur_seq.copy(), cur_e
                break

        if not improved and steps > 0 and cur_e > 0 and cfg.kick:
            t_kick = time.perf_counter()
            prev_e = cur_e
            q_now = cur_e // q_scale

            if q_now < lowest_q_seen:
                lowest_q_seen = q_now
                times_at_lowest = 0
            elif q_now == lowest_q_seen:
                times_at_lowest += 1

            ev = {"type": "kick", "q_before": q_now, "steps_left": steps}
            if (
                cfg.targeted_escape
                and not early_escape_done
                and q_now == lowest_q_seen
                and times_at_lowest >= 2
            ):
                early_escape_done = True
                ev["kind"] = "targeted"
                ev["escape"] = {}
                improved, cur_seq, cur_e, best_seq, best_e, kick_used = targeted_kick_preflip(
                    cur_seq, tracker, rng, q_now, total_budget, best_seq, best_e, stats,
                    ev["escape"], sr, wl,
                )
                steps -= kick_used
                ev["success"] = bool(improved) and cur_e == 0
            else:
                cur_e, cols = _random_kick_pre(cur_seq, tracker, rng)
                improved = True
                steps -= 1
                stats.kick_evals += 1
                ev["kind"] = "random"
                ev["success"] = False
                if cur_e == 0:
                    # 4-flip kick solved (not a single-flip d=-u event)
                    pre_seq = cur_seq.copy()
                    for s in range(n_seqs):
                        pre_seq[s, cols[s]] *= -1
                    _record_solve(sr, "kick4", pre_seq, (0, cols[0]),
                                  q_pre=int(np.dot(common.u_of(pre_seq),
                                                   common.u_of(pre_seq))),
                                  extra={"cols": cols})
                    solved_step = total_budget - steps
                    best_seq, best_e = cur_seq.copy(), cur_e
                    log["events"].append(ev)
                    break

            stats.kicks += 1
            stats.energy_saved_kicks += prev_e - cur_e
            stats.kick_time_s += time.perf_counter() - t_kick
            ev["q_after"] = cur_e // q_scale
            log["events"].append(ev)
            if cur_e == 0:
                solved_step = total_budget - steps
                best_seq, best_e = cur_seq.copy(), cur_e
                break
            if not improved:
                stats._hit_other()

        best_seq, best_e = _update_best(cur_seq, cur_e, best_seq, best_e)

    if solved_step is None and best_e == 0:
        solved_step = total_budget - steps
    log["solved_step"] = solved_step
    log["lowest_q_seen"] = lowest_q_seen
    log["times_at_lowest"] = times_at_lowest
    return best_seq, best_e, total_budget - max(steps, 0), stats


def random_seqs(n: int, rng: np.random.Generator) -> np.ndarray:
    return rng.choice(np.array([-1, 1], dtype=np.int8), size=(4, n))
