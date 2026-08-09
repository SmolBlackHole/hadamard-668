"""Variant H: re-seeded noisy tabu walks at every stuck point.

Replica of search_preflip (dcfacc5 pipeline) with a pluggable walk policy:

  policy "base":    1 deterministic walk (noise 0) per stuck point  [solver default]
  policy "seed":    M=4 walks with noise a=0.3 per stuck point, the BEST
                    outcome is adopted (solve on any hit)
  policy "cap":     M=4 noisy walks with altitude cap q_max=9 (lateral search
                    inside the hazard window 4..9)
  policy "base400": baseline policy with 4x step budget (time-equivalent
                    control for "seed": same total walk count)

Rationale (q14/q15): the solver's tabu walk is deterministic (tabu_noise=0);
record-low Q=1 states are C-poor (0 hits in 184k re-seeded steps), while the
~1.6% per-walk hits come from transient stuck states at Q 2..13.  Re-seeding
multiplies the chances from every stuck point.

Usage: python variant_h.py <policy> <start_seed> <count> [worker_id]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

from preflip_solver import (  # noqa: E402
    _greedy_descent_pre,
    _random_kick_pre,
    random_seqs,
    tabu_walk_kernel_preflip,
    targeted_kick_preflip,
)
from solver_snapshot.solver import SearchStats, SolverConfig, _positions, _update_best  # noqa: E402
from solver_snapshot.tracker import Tracker  # noqa: E402

HERE = Path(__file__).resolve().parent


def _tabu_walks_multi(
    cur_seq: np.ndarray,
    tracker: Tracker,
    cur_e: int,
    rng: np.random.Generator,
    config: SolverConfig,
    m_walks: int,
    noise_a: float,
    q_max: int,
) -> tuple[int, int, list[dict]]:
    """m_walks tabu walks from the SAME state (fresh noise each).

    The best walk outcome is adopted as the new current state.  q_max < 0
    disables the altitude cap (no cap in the original kernel; here we just
    ignore q_max for "base"/"seed" and use a separate capped kernel).
    """
    n_seqs, n_cols = cur_seq.shape
    m = tracker._u.size
    infos: list[dict] = []
    entry_e = cur_e  # best energy reached by the multi-walk set
    total_evals = 0
    for _ in range(m_walks):
        # ALWAYS consume the rng (the baseline calls rng.random even with
        # tabu_noise=0; skipping it desynchronizes the kick stream)
        noise = noise_a * rng.random((config.tabu_steps, n_seqs, n_cols))
        pre_pos = np.full(2, -1, dtype=np.int64)
        pre_seq = np.zeros_like(cur_seq)
        pre_u = np.zeros(m, dtype=np.int32)
        pre_d = np.zeros(m, dtype=np.int8)
        q_traj = np.zeros(config.tabu_steps, dtype=np.int64)
        q_min_pre = np.zeros(1, dtype=np.int64)
        deficit_hist = np.zeros(config.tabu_steps, dtype=np.int64)
        best_seq, best_delta, best_norm2, best_u, best_q, evals, solved = (
            tabu_walk_kernel_preflip(
                cur_seq.copy(), tracker._delta.copy(), tracker._norm2.copy(),
                tracker._u.copy(), tracker._q,
                tracker._update_cols, tracker._update_lags, tracker._update_signs,
                noise, config.tabu_tenure, config.tabu_decay,
                pre_pos, pre_seq, pre_u, pre_d, q_traj, q_min_pre, deficit_hist,
            )
        )
        walk_e = 64 * n_cols * best_q
        total_evals += evals
        infos.append({"entry_q": cur_e // (64 * n_cols), "used": int(evals),
                      "solved": bool(solved), "end_q": int(best_q)})
        # adopt only improving walks: the kernel's best_* arrays are
        # uninitialized for a non-improving walk (same rule as the baseline)
        if walk_e < entry_e:
            entry_e = walk_e
            cur_seq[...] = best_seq
            tracker._adopt(best_seq, best_u, best_q, best_delta, best_norm2)
        if solved:
            break  # a hit solves the run
    return entry_e, total_evals, infos


def search_variant_h(
    seqs: np.ndarray,
    rng: np.random.Generator,
    *,
    steps: int = 100000,
    policy: str = "base",
    m_walks: int = 4,
    noise_a: float = 0.3,
) -> dict:
    cfg = SolverConfig(targeted_escape=True)
    n_seqs, n_cols = seqs.shape
    positions = _positions(n_seqs, n_cols)
    stats = SearchStats()
    q_scale = 64 * n_cols

    cur_seq = seqs.copy()
    tracker = Tracker()
    t0 = time.perf_counter()
    tracker.build(cur_seq)
    cur_e = tracker.energy()
    steps -= 1
    best_seq, best_e = cur_seq.copy(), cur_e
    lowest_q_seen = cur_e // q_scale
    times_at_lowest = 0
    early_escape_done = False
    n_walks = 0
    n_solve_walks = 0
    events: list[dict] = []

    while steps > 0 and best_e > 0:
        improved, cur_e, used, accepted = _greedy_descent_pre(
            cur_seq, tracker, positions, cur_e, steps, q_scale, cfg, stats)
        steps -= used
        if cur_e == 0 and accepted is not None:
            best_seq, best_e = cur_seq.copy(), cur_e
            break
        if steps <= 0:
            best_seq, best_e = _update_best(cur_seq, cur_e, best_seq, best_e)
            break
        if not improved and cfg.tabu:
            n_walks += 1
            e_before = cur_e
            if policy in ("base", "base400"):
                mw, na = 1, 0.0  # base400 = more budget, same deterministic walk
            elif policy == "target":
                # targeted re-seed: noisy multi-walks only at transient stuck
                # points (entry Q 3..12, the C-rich band); the record-low
                # Q<=2 traps get the cheap baseline walk (0/1814 hits there)
                mw, na = (m_walks, noise_a) if 3 <= e_before // q_scale <= 12 \
                    else (1, 0.0)
            else:
                mw, na = m_walks, noise_a
            result, evals, infos = _tabu_walks_multi(
                cur_seq, tracker, cur_e, rng, cfg, mw, na, q_max=-1)
            n_solve_walks += sum(1 for i in infos if i["solved"])
            cur_e = tracker.energy()
            improved = cur_e < e_before  # walk improved -> no kick (as baseline)
            best_seq, best_e = _update_best(cur_seq, cur_e, best_seq, best_e)
            events.append({"type": "tabu", "q": cur_e // q_scale,
                           "steps_left": steps})
            if cur_e == 0:
                best_seq, best_e = cur_seq.copy(), 0
                break
        if not improved and steps > 0 and cur_e > 0 and cfg.kick:
            q_now = cur_e // q_scale
            if q_now < lowest_q_seen:
                lowest_q_seen = q_now
                times_at_lowest = 0
            elif q_now == lowest_q_seen:
                times_at_lowest += 1
            if (cfg.targeted_escape and not early_escape_done
                    and q_now == lowest_q_seen and times_at_lowest >= 2):
                early_escape_done = True
                ev = {"type": "kick", "kind": "targeted", "q_before": q_now,
                      "escape": {}}
                events.append(ev)
                sr: dict = {}
                improved2, cur_seq, cur_e, best_seq, best_e, kick_used = (
                    targeted_kick_preflip(
                        cur_seq, tracker, rng, q_now, steps, best_seq, best_e,
                        stats, ev["escape"], sr, None)
                )
                steps -= kick_used
                if cur_e == 0:
                    best_seq, best_e = cur_seq.copy(), 0
                    break
            else:
                cur_e, cols = _random_kick_pre(cur_seq, tracker, rng)
                if cur_e == 0:
                    best_seq, best_e = cur_seq.copy(), 0
                    break
            steps -= 1
            stats.kicks += 1
        best_seq, best_e = _update_best(cur_seq, cur_e, best_seq, best_e)

    return {"solved": best_e == 0, "best_q": best_e // q_scale,
            "lowest_q": lowest_q_seen, "time_s": round(time.perf_counter() - t0, 2),
            "n_walks": n_walks, "n_solve_walks": n_solve_walks,
            "events": events}


def main() -> None:
    policy = sys.argv[1] if len(sys.argv) > 1 else "base"
    start = int(sys.argv[2]) if len(sys.argv) > 2 else 6400
    count = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    worker = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    n = int(sys.argv[5]) if len(sys.argv) > 5 else 44
    f = HERE / f"variant_h_{policy}_n{n}.jsonl"
    done = set()
    if f.exists():
        for line in f.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(line)["seed"])
            except Exception:
                pass
    seeds = [s for s in range(start, start + count) if s not in done]
    budget = 400000 if policy == "base400" else 100000
    print(f"variant {policy} (budget {budget}): {len(seeds)} runs (worker {worker})",
          flush=True)
    n_solved = 0
    for i, seed in enumerate(seeds):
        rng = np.random.default_rng(seed)
        seqs = random_seqs(n, rng)
        r = search_variant_h(seqs, rng, steps=budget, policy=policy,
                             m_walks=4, noise_a=0.3)
        r["seed"] = seed
        r["policy"] = policy
        n_solved += int(r["solved"])
        with f.open("a", encoding="utf-8") as fh:
            # numpy int32/int64 in the escape event log are not JSON-safe
            fh.write(json.dumps(r, default=lambda o: int(o)
                                 if isinstance(o, np.integer)
                                 else float(o) if isinstance(o, np.floating)
                                 else str(o)) + "\n")
        if (i + 1) % 5 == 0:
            print(f"  worker{worker}: {i+1}/{len(seeds)} done, solved {n_solved} "
                  f"(seed {seed}: solved={r['solved']}, {r['time_s']:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
