"""QWindow solver: steered ILS that keeps the search INSIDE the C-rich band.

Design from the d2_shell_hazard findings (d2_shell_hazard.md):

  1. C-states live at Q_pre 4..9 (97.8% of the 198 hits); the per-step
     hazard h(q) scales as 2^(-q): h(4..6) ~ 3e-2 vs h(9) ~ 2e-3, and the
     record-low Q<=2 states are C-poor (0 hits in 1.814 walks).
  2. The baseline solver wastes its time: greedy grinds to the Q~3 floor
     (C-empty), 4-flip kicks throw the state to Q 50-150 (C-absent), and
     the deterministic walk (tabu_noise=0) spends 65% of its steps at Q
     7..9 where h is low.
  3. h(q,t) shows structure accumulation: at FIXED q, the hit rate rises
     with walk position (fresh territory at the same Q-shell).

QWindow therefore:
  - stops the greedy descent at Q <= WINDOW_HIGH (instead of the floor),
    so every walk starts inside the band,
  - runs the walk with an altitude cap (lateral search inside the band)
    and fresh noise per walk (stochastic trajectories),
  - kicks with 1-2 flips only, so the state stays near the band.

Budget: descent flips and kick flips consume steps; walks are budget-free
(as in the baseline).  Reuses the dcfacc5 Tracker (exact cache) and a
capped pre-flip-snapshot tabu kernel.

Usage:  python qwindow_solver.py <n> <start_seed> <count> [worker_id]
"""
from __future__ import annotations

import base64
import json
import sys
import time
from pathlib import Path

import numpy as np
from numba import njit  # pyright: ignore[reportMissingImports]

sys.path.insert(0, str(Path(__file__).resolve().parent))

from preflip_solver import random_seqs  # noqa: E402
from solver_snapshot.tracker import Tracker  # noqa: E402

HERE = Path(__file__).resolve().parent

WINDOW_HIGH = 9
WALK_STEPS = 200
TENURE = 5.0
DECAY = 0.7
NOISE_A = 0.2


@njit(cache=True)
def tabu_walk_kernel_qwin(
    seqs, delta, norm2, u, q,
    update_cols, update_lags, update_signs,
    noise, tenure, decay, q_max,
    pre_pos, pre_seq, pre_u, pre_delta,
):
    """Tabu walk with altitude cap q_max and pre-flip solve snapshot.

    Moves that leave Q > q_max are rejected (score inf).  On a solving
    step (q_after == 0) the pre-flip state is snapshotted.  Returns
    (best_seq, best_delta, best_norm2, best_u, best_q, used, solved).
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
        for candidate in range(n_seqs * n_cols):
            delta_q = int(norm2[candidate])
            for lag in range(n_lags):
                delta_q += 2 * int(delta[candidate, lag]) * int(u[lag])
            q_after = q + delta_q
            if q_max >= 0 and q_after > q_max:
                continue
            s = candidate // n_cols
            c = candidate - s * n_cols
            score = q_after * (1.0 + tabu[s, c] + noise[step, s, c])
            if score < best_score:
                best_score = score
                index = candidate

        s = index // n_cols
        c = index - s * n_cols
        delta_q = int(norm2[index])
        for lag in range(n_lags):
            delta_q += 2 * int(delta[index, lag]) * int(u[lag])

        if q + delta_q == 0 and solved == 0:
            pre_pos[0] = s
            pre_pos[1] = c
            for i in range(n_seqs):
                for j in range(n_cols):
                    pre_seq[i, j] = seqs[i, j]
            for lag in range(n_lags):
                pre_u[lag] = u[lag]
                pre_delta[lag] = delta[index, lag]
            solved = 1

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

    return best_seq, best_delta, best_norm2, best_u, best_q, used, solved


def _run_walk(cur_seq, tracker, rng, n, m, noise_a, q_max):
    """One capped noisy walk; returns (solved, q_pre or None, pre_seq or None)."""
    noise = noise_a * rng.random((WALK_STEPS, 4, n))
    pre_pos = np.full(2, -1, dtype=np.int64)
    pre_seq = np.zeros_like(cur_seq)
    pre_u = np.zeros(m, dtype=np.int32)
    pre_d = np.zeros(m, dtype=np.int8)
    best_seq, best_delta, best_norm2, best_u, best_q, evals, solved = (
        tabu_walk_kernel_qwin(
            cur_seq.copy(), tracker._delta.copy(), tracker._norm2.copy(),
            tracker._u.copy(), tracker._q,
            tracker._update_cols, tracker._update_lags, tracker._update_signs,
            noise, TENURE, DECAY, q_max,
            pre_pos, pre_seq, pre_u, pre_d,
        )
    )
    q_pre = None
    pre_out = None
    if solved:
        q_pre = int(np.dot(pre_u, pre_u))
        pre_out = pre_seq.copy()
    if best_q < tracker._q:
        cur_seq[...] = best_seq
        tracker._adopt(best_seq, best_u, best_q, best_delta, best_norm2)
    return bool(solved), q_pre, pre_out


def search_qwindow(
    seqs: np.ndarray,
    rng: np.random.Generator,
    *,
    steps: int = 100000,
    window_high: int = WINDOW_HIGH,
    noise_a: float = NOISE_A,
    q_max: int = WINDOW_HIGH,
) -> dict:
    """QWindow search loop.  Returns a summary dict."""
    n = seqs.shape[1]
    m = (n - 1) // 2
    tracker = Tracker()
    t0 = time.perf_counter()
    tracker.build(seqs)
    cur = seqs.copy()
    q = tracker._q
    budget = steps
    best_q = q
    n_walks = 0
    solved = False
    q_pre = None
    pre_b64 = None
    events: list[dict] = []

    while budget > 0 and best_q > 0:
        # --- descent only while ABOVE the window (stop before the floor) --
        improved = True
        while improved and q > window_high and budget > 0:
            qs_all = tracker.flip_qs()
            imp = np.flatnonzero(qs_all < q)
            if imp.size:
                idx = int(imp[0])
                s, c = divmod(idx, n)
                tracker.accept(cur, s, c)
                q = tracker._q
                budget -= 1
                if q < best_q:
                    best_q = q
                if q == 0:
                    solved = True
                    break
            else:
                improved = False
        if solved:
            break

        # --- walk inside the window (capped, noisy) -------------------------
        n_walks += 1
        hit, q_pre, pre_out = _run_walk(cur, tracker, rng, n, m, noise_a, q_max)
        q = tracker._q
        if q < best_q:
            best_q = q
        if hit:
            solved = True
            pre_b64 = base64.b64encode(pre_out.tobytes()).decode() \
                if pre_out is not None else None
            break

        # --- small kick: 1 flip in-window, 2 flips below the window ---------
        k = 2 if q < 4 else 1
        for _ in range(k):
            if budget <= 0:
                break
            s = int(rng.integers(0, 4))
            c = int(rng.integers(0, n))
            tracker.accept(cur, s, c)
            q = tracker._q
            budget -= 1
            if q < best_q:
                best_q = q
            if q == 0:
                solved = True
                break
        if solved:
            break
        events.append({"type": "cycle", "q_after_kick": int(q),
                       "budget_left": int(budget)})

    return {"solved": solved, "best_q": int(best_q),
            "q_pre": q_pre, "pre_b64": pre_b64, "n_walks": n_walks,
            "time_s": round(time.perf_counter() - t0, 2),
            "budget_used": int(steps - budget),
            "events": events[-20:]}


def main() -> None:
    n = int(sys.argv[1])
    start = int(sys.argv[2])
    count = int(sys.argv[3])
    worker = int(sys.argv[4]) if len(sys.argv) > 4 else 0
    f = HERE / f"qwindow_n{n}.jsonl"
    done = set()
    if f.exists():
        for line in f.read_text(encoding="utf-8").splitlines():
            try:
                done.add(json.loads(line)["seed"])
            except Exception:
                pass
    seeds = [s for s in range(start, start + count) if s not in done]
    print(f"qwindow n={n}: {len(seeds)} runs (worker {worker})", flush=True)
    n_solved = 0
    for i, seed in enumerate(seeds):
        rng = np.random.default_rng(seed)
        seqs = random_seqs(n, rng)
        r = search_qwindow(seqs, rng)
        r["seed"] = seed
        r["n"] = n
        n_solved += int(r["solved"])
        with f.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(r) + "\n")
        if (i + 1) % 5 == 0:
            print(f"  worker{worker}: {i+1}/{len(seeds)} done, solved {n_solved} "
                  f"(seed {seed}: solved={r['solved']}, {r['time_s']:.1f}s, "
                  f"walks={r['n_walks']})", flush=True)


if __name__ == "__main__":
    main()
