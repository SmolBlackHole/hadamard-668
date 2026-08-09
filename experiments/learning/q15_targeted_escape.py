"""Q15: Targeted-escape levers — re-seeded walks, unconditional C-hazard,
construction, and the Q-window variant.

KEY OBSERVATION: the solver's tabu walk is DETERMINISTIC (tabu_noise=0.0,
score = q_after * (1 + tabu + 0)).  Its randomness comes solely from the
distribution of stuck states at which it is invoked.  The lever question is
therefore: does re-seeding the walk from the SAME stuck state with fresh
noise multiply the C-hit probability?

Part A: d == -u at random states (n=44/50): null-model check, expect ~0.
Part B: C-neighborhood construction: flipping one bit of a KNOWN solution
        yields a state in C (verify d(x) == -u(x) numerically).
Part C: re-seed experiment (n=44 escape states + n=50 low-Q states):
        M independent walks per state, noise amplitude a in {0.0, 0.3}.
        Measures, unconditional (no selection bias):
          - walk-level hit rate from a fixed state,
          - per-step hazard h(q) = P(hit next step | state Q = q),
          - P(u ternary | Q=q) on walk states and h(q | u ternary),
          - positional hazard (hits vs walk position).
Part D: Q-window variant: altitude cap Q_max (lateral search inside the
        hazard window) vs baseline.  Pure driver, solver untouched.

Usage: python q15_targeted_escape.py
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
import common  # noqa: E402
from preflip_solver import random_seqs  # noqa: E402
from solver_snapshot.tracker import Tracker  # noqa: E402

HERE = Path(__file__).resolve().parent


def delta_all(seqs: np.ndarray) -> np.ndarray:
    n = seqs.shape[1]
    m = (n - 1) // 2
    c = np.arange(n)[:, None]
    t = np.arange(1, m + 1)[None, :]
    forward = (c + t) % n
    backward = (c - t) % n
    fwd_sign = np.where(c + t < n, 1, -1)
    bwd_sign = np.where(backward + t < n, 1, -1)
    v = seqs.astype(np.int16)
    d = -2 * v[:, :, None] * (fwd_sign[None, :, :] * v[:, forward] +
                              bwd_sign[None, :, :] * v[:, backward])
    return (d // 4).reshape(4 * n, m)


# --- Part A: d == -u on random states -----------------------------------------


def random_state_c_count(n: int, k: int, seed: int = 5) -> dict:
    rng = np.random.default_rng(seed)
    hits = 0
    per_state = []
    for _ in range(k):
        s = random_seqs(n, rng)
        u = common.u_of(s)
        D = delta_all(s)
        cnt = int(np.sum(np.all(D == -u[None, :], axis=1)))
        hits += cnt
        per_state.append(cnt)
    return {"n": n, "states": k, "flips_total": k * 4 * n, "d_eq_minus_u": hits,
            "max_per_state": max(per_state)}


# --- Part B: C-neighborhood of solutions --------------------------------------


def verify_solution_neighborhood(n: int, n_solutions: int, seed: int = 5) -> dict:
    d = json.load(open(common.DATA / "solutions.json", encoding="utf-8"))["gs4"]
    runs = [r for r in d.get(str(n), []) if r.get("solved")][:n_solutions]
    ok = 0
    checked = 0
    qpres: list[int] = []
    t0 = time.perf_counter()
    for r in runs:
        seqs = np.frombuffer(base64.b64decode(r["seqs_b64"]), dtype=np.int8).reshape(4, n)
        rng = np.random.default_rng(seed + r["seed"])
        s, c = int(rng.integers(0, 4)), int(rng.integers(0, n))
        x = seqs.copy()
        x[s, c] *= -1
        # the flipped row's delta on x must equal -u(x)
        ux = common.u_of(x)
        dx = delta_all(x)[s * n + c]
        ok += int(np.array_equal(dx, -ux))
        checked += 1
        qpres.append(int((ux * ux).sum()))
    return {"n": n, "checked": checked, "d_eq_minus_u": ok,
            "q_pre_of_flipped": sorted(set(qpres)),
            "time_s": round(time.perf_counter() - t0, 1)}


# --- Part C/D: walk kernel with u-history, noise amplitude, altitude cap ------


@njit(cache=True)
def tabu_walk_kernel_hist(
    seqs, delta, norm2, u, q,
    update_cols, update_lags, update_signs,
    noise, tenure, decay, q_max,
    u_hist, q_traj, deficit_hist,
):
    """Tabu walk; records u and q after every step.  Hit (deficit 0) is
    captured per step.  q_max < 0 disables the altitude cap; otherwise moves
    that leave Q > q_max are rejected (score inf)."""
    n_seqs, n_cols = seqs.shape
    n_lags = u.size
    tabu = np.zeros((n_seqs, n_cols), dtype=np.float64)
    used = 0
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
            if q_max >= 0 and q_after > q_max:
                continue
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
        q += delta_q
        q_traj[step] = q
        for lag in range(n_lags):
            u[lag] += delta[index, lag]
        u_hist[step, :] = u[:]

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
    return used


def run_walks(state: np.ndarray, n_walks: int, steps: int, noise_a: float,
              q_max: int, rng: np.random.Generator) -> dict:
    """Run n_walks tabu walks from one state.  Returns per-step aggregates."""
    tr = Tracker()
    tr.build(state)
    m = tr._u.size
    n_seqs, n_cols = state.shape
    steps_at: dict[int, int] = {}
    hits_at: dict[int, int] = {}
    tern_at: dict[int, int] = {}
    n_hits = 0
    t_at_risk: list[int] = []
    t_hits: list[int] = []
    for _ in range(n_walks):
        noise = noise_a * rng.random((steps, n_seqs, n_cols))
        u_hist = np.zeros((steps, m), dtype=np.int8)
        q_traj = np.zeros(steps, dtype=np.int64)
        deficit = np.zeros(steps, dtype=np.int64)
        used = tabu_walk_kernel_hist(
            state.copy(), tr._delta.copy(), tr._norm2.copy(), tr._u.copy(), tr._q,
            tr._update_cols, tr._update_lags, tr._update_signs,
            noise, 5.0, 0.7, q_max, u_hist, q_traj, deficit,
        )
        for k in range(used):
            qk = int(q_traj[k])
            steps_at[qk] = steps_at.get(qk, 0) + 1
            u_tern = int(np.all(np.abs(u_hist[k]) <= 1))
            tern_at[qk] = tern_at.get(qk, 0) + u_tern
            if deficit[k] == 0:
                hits_at[qk] = hits_at.get(qk, 0) + 1
                n_hits += 1
                t_hits.append(k + 1)
                break  # walk solved; stop
            t_at_risk.append(k + 1)
    return {"steps_at": steps_at, "hits_at": hits_at, "tern_at": tern_at,
            "n_hits": n_hits, "t_at_risk": t_at_risk, "t_hits": t_hits}


def load_escape_states_n44(limit: int = 200) -> list[tuple[np.ndarray, int]]:
    rows = [json.loads(l) for l in (HERE / "collect_n44_spec.jsonl")
            .read_text(encoding="utf-8").splitlines()]
    out = []
    for r in rows:
        if "esc_b64" not in r:
            continue
        seqs = np.frombuffer(base64.b64decode(r["esc_b64"]), dtype=np.int8).reshape(4, 44)
        u = np.array(r["esc_u"], dtype=np.int64)
        out.append((seqs, int(np.dot(u, u))))
        if len(out) >= limit:
            break
    return out


def sample_low_q_n50(limit: int = 150, max_q: int = 13, seed: int = 11) -> list[tuple[np.ndarray, int]]:
    """Descent + kicks to collect n=50 states with Q <= max_q (walk regime)."""
    rng = np.random.default_rng(seed)
    out: list[tuple[np.ndarray, int]] = []
    while len(out) < limit:
        seqs = random_seqs(50, rng)
        tr = Tracker()
        tr.build(seqs)
        cur = seqs.copy()
        q = tr._q
        for _ in range(6000):
            qs_all = tr.flip_qs()
            improving = np.flatnonzero(qs_all < q)
            if improving.size:
                idx = int(improving[0])
                s, c = divmod(idx, 50)
                tr.accept(cur, s, c)
                q = tr._q
                if q <= max_q and len(out) < limit:
                    out.append((cur.copy(), q))
                continue
            cols = rng.integers(0, 50, size=4)
            for s in range(4):
                tr.accept(cur, s, int(cols[s]))
            q = tr._q
        if not out:
            print("  n=50: no low-Q states", flush=True)
    return out


def hazard_from(per_state: list[dict], label: str) -> dict:
    """Aggregate per-step hazard over all walks of all states."""
    steps_at: dict[int, int] = {}
    hits_at: dict[int, int] = {}
    tern_at: dict[int, int] = {}
    tot_hits = 0
    t_ar: list[int] = []
    t_hits: list[int] = []
    for ps in per_state:
        for k, v in ps["steps_at"].items():
            steps_at[k] = steps_at.get(k, 0) + v
        for k, v in ps["hits_at"].items():
            hits_at[k] = hits_at.get(k, 0) + v
        for k, v in ps["tern_at"].items():
            tern_at[k] = tern_at.get(k, 0) + v
        tot_hits += ps["n_hits"]
        t_ar.extend(ps["t_at_risk"])
        t_hits.extend(ps["t_hits"])
    qs = sorted(set(steps_at) | set(hits_at))
    haz = {}
    for q in qs:
        st = steps_at.get(q, 0)
        hi = hits_at.get(q, 0)
        te = tern_at.get(q, 0)
        haz[str(q)] = {"steps": st, "hits": hi, "h_step": hi / st if st else None,
                       "tern_share": te / st if st else None,
                       "h_tern": hi / te if te else None}
    out = {"label": label, "n_hits": tot_hits,
           "steps_total": sum(steps_at.values()),
           "hazard_by_q": haz}
    if t_ar:
        ar = np.bincount(t_ar)
        th = np.bincount(t_hits, minlength=ar.size)
        band = {}
        for lo in (1, 41, 81, 121, 161):
            hi = lo + 39
            a = int(ar[lo - 1 : hi].sum()) if hi <= ar.size else int(ar[lo - 1 :].sum())
            h = int(th[lo - 1 : hi].sum()) if hi <= ar.size else int(th[lo - 1 :].sum())
            band[f"{lo}-{min(hi, ar.size)}"] = {"at_risk": a, "hits": h,
                                                "h": h / a if a else None}
        out["positional_hazard_bands"] = band
    return out


def main() -> None:
    out: dict = {"time0": time.strftime("%Y-%m-%d %H:%M:%S")}
    print("Part A: d == -u at random states ...", flush=True)
    out["random_c_count"] = [random_state_c_count(n, 300) for n in (44, 50)]

    print("Part B: solution neighborhood in C ...", flush=True)
    out["neighborhood_c"] = [verify_solution_neighborhood(n, 20) for n in (44, 50)]

    print("Part C: re-seed experiment n=44 (escape states) ...", flush=True)
    states44 = load_escape_states_n44(200)
    out["n44_states"] = {"n": len(states44),
                         "q_hist": {str(k): int(np.sum([q == k for _, q in states44]))
                                    for k in range(6)}}
    rng = np.random.default_rng(42)
    for amp in (0.0, 0.3):
        ps = []
        for i, (st, _) in enumerate(states44):
            ps.append(run_walks(st, n_walks=10 if amp else 1, steps=200,
                                noise_a=amp, q_max=-1,
                                rng=np.random.default_rng(1000 + i)))
        out[f"n44_reseed_a{amp}"] = hazard_from(ps, f"n=44 re-seed noise={amp}")

    print("Part C: re-seed experiment n=50 (low-Q states) ...", flush=True)
    states50 = sample_low_q_n50(150)
    out["n50_states"] = {"n": len(states50)}
    rng = np.random.default_rng(43)
    for amp in (0.0, 0.3):
        ps = []
        for i, (st, _) in enumerate(states50):
            ps.append(run_walks(st, n_walks=10 if amp else 1, steps=200,
                                noise_a=amp, q_max=-1,
                                rng=np.random.default_rng(2000 + i)))
        out[f"n50_reseed_a{amp}"] = hazard_from(ps, f"n=50 re-seed noise={amp}")

    print("Part D: Q-window variant (n=44, cap 9 vs none, noise 0.3) ...", flush=True)
    for cap in (9, -1):
        ps = []
        for i, (st, _) in enumerate(states44[:100]):
            ps.append(run_walks(st, n_walks=10, steps=200, noise_a=0.3,
                                q_max=cap, rng=np.random.default_rng(3000 + i)))
        out[f"n44_cap{cap}"] = hazard_from(ps, f"n=44 Q-window cap={cap}")

    print(json.dumps(out, indent=1, default=str))
    (HERE / "q15_results.json").write_text(json.dumps(out, indent=1, default=str),
                                           encoding="utf-8")


if __name__ == "__main__":
    main()
