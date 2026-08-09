"""Q13: ||d||^2 statistics — where does the solution shell live?

The identity Q_pre = ||d_solve||^2 (136/136 validated) means: the Q-level of
the pre-solve state equals the squared norm of the solving flip's delta
vector, i.e. its number of "broken lags".  The distribution of ||d_i||^2
over all 4n single flips therefore locates the solution shell.

Sources:
  1. Random states (n=44/50/52): ||d||^2 ~ Bin(m, 1/2) theory check.
  2. One-flip neighborhoods of SOLUTIONS (data/solutions.json): flipping one
     bit of a solution yields a state in C with Q_pre = ||d||^2 of that row
     (u(x) = d_sol(y), d(x) = -d_sol(y), hence d(x) = -u(x)).  This is the
     DIRECT empirical answer: on which Q-shell do the solutions lie?
  3. Escape states (n=44 collect_n44_spec.jsonl, n=52 all_seeds_escape.json):
     ||d||^2 statistics of the flips available at real low-Q states.
  4. n=50 low-Q states via descent + random kicks (own sampler).
  5. Identity check on all 136 preflip solutions: Q_pre == ||d_solve||^2
     == |supp(u_pre)|.

Usage: python q13_d2_stats.py
"""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from solver_snapshot.tracker import Tracker  # noqa: E402

HERE = Path(__file__).resolve().parent
DATA = common.DATA


def delta_all(seqs: np.ndarray) -> np.ndarray:
    """Full single-flip delta matrix (4n, m) for a state."""
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


def d2(seqs: np.ndarray) -> np.ndarray:
    """||d_i||^2 for all 4n flips (int)."""
    D = delta_all(seqs).astype(np.int32)
    return np.asarray((D * D).sum(axis=1), dtype=np.int64)


def norm_stats(arr: np.ndarray, title: str) -> dict:
    """Histogram + key percentiles of ||d||^2 values."""
    mx = int(arr.max())
    hist = {str(k): int((arr == k).sum()) for k in range(mx + 1)}
    out = {"title": title, "n": int(arr.size), "mean": float(arr.mean()),
           "median": float(np.median(arr)),
           "p25": float(np.percentile(arr, 25)),
           "p75": float(np.percentile(arr, 75)),
           "min": int(arr.min()), "max": int(arr.max()),
           "hist": hist}
    out["P_le_11"] = float((arr <= 11).mean())
    out["P_le_9"] = float((arr <= 9).mean())
    out["P_4_9"] = float(((arr >= 4) & (arr <= 9)).mean())
    return out


# --- 1. random states ---------------------------------------------------------


def random_state_stats(n: int, k: int, seed: int = 1234) -> dict:
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(k):
        s = rng.choice(np.array([-1, 1], dtype=np.int8), size=(4, n))
        vals.append(d2(s))
    arr = np.concatenate(vals)
    return norm_stats(arr, f"random states n={n} (k={k})")


# --- 2. solution neighborhoods ------------------------------------------------


def load_solutions() -> dict[int, list[np.ndarray]]:
    d = json.load(open(DATA / "solutions.json", encoding="utf-8"))["gs4"]
    out: dict[int, list[np.ndarray]] = {}
    for n, runs in d.items():
        n = int(n)
        if n not in (44, 48, 50, 52):
            continue
        arr = []
        for r in runs:
            if not r.get("solved"):
                continue
            seqs = np.frombuffer(base64.b64decode(r["seqs_b64"]), dtype=np.int8)
            arr.append(seqs.reshape(4, n))
        out[n] = arr
    return out


def solution_shell() -> dict:
    """||d||^2 distribution over all 4n flips at solutions (u=0)."""
    sols = load_solutions()
    out: dict = {}
    for n, states in sols.items():
        vals = np.concatenate([d2(s) for s in states])
        out[str(n)] = norm_stats(vals, f"1-flip neighborhood of solutions n={n} "
                                        f"({len(states)} solutions, {len(states)*4*n} flips)")
    return out


# --- 3. escape states ---------------------------------------------------------


def escape_state_stats() -> dict:
    out: dict = {}
    # n=44
    rows = [json.loads(l) for l in (HERE / "collect_n44_spec.jsonl")
            .read_text(encoding="utf-8").splitlines()]
    vals = []
    qs = []
    shell_ok = 0   # state has a flip with ||d||^2 == Q
    n_state = 0
    for r in rows:
        if "esc_b64" not in r:
            continue
        seqs = np.frombuffer(base64.b64decode(r["esc_b64"]), dtype=np.int8).reshape(4, 44)
        u = np.array(r["esc_u"], dtype=np.int64)
        q = int(np.dot(u, u))
        v = d2(seqs)
        vals.append(v)
        qs.append(q)
        n_state += 1
        shell_ok += int((v == q).any())
    arr = np.concatenate(vals)
    qa = np.array(qs)
    out["n44_escape"] = norm_stats(arr, f"escape states n=44 (n_state={n_state})")
    out["n44_escape_state_q"] = {"mean": float(qa.mean()),
                                 "median": float(np.median(qa)),
                                 "hist": {str(k): int((qa == k).sum())
                                          for k in range(int(qa.min()), int(qa.max()) + 1)}}
    out["n44_escape_has_shell_flip"] = f"{shell_ok}/{n_state}"
    # correlation state Q vs mean/min ||d||^2 over flips
    mn = np.array([v.min() for v in vals])
    out["n44_escape_corr_Q_vs_min_norm2"] = float(np.corrcoef(qa, mn)[0, 1]) \
        if len(np.unique(mn)) > 1 else None
    # n=52
    d52 = json.load(open(HERE / "all_seeds_escape.json", encoding="utf-8"))
    vals52 = []
    qs52 = []
    n52 = 0
    for k, v in d52.items():
        if "seqs_at_escape_b64" not in v:
            continue
        seqs = np.frombuffer(base64.b64decode(v["seqs_at_escape_b64"]), dtype=np.int8).reshape(4, 52)
        u = np.array(json.loads(v["u_at_escape"]) if isinstance(v["u_at_escape"], str)
                     else v["u_at_escape"], dtype=np.int64)
        qs52.append(int(np.dot(u, u)))
        vals52.append(d2(seqs))
        n52 += 1
    arr52 = np.concatenate(vals52)
    out["n52_escape"] = norm_stats(arr52, f"escape states n=52 (n_state={n52})")
    out["n52_escape_state_q"] = {"mean": float(np.mean(qs52)),
                                 "median": float(np.median(qs52))}
    return out


# --- 4. n=50 low-Q sampler ----------------------------------------------------


def sample_low_q_states(n: int, n_states: int, seed: int = 42,
                        max_q: int = 30) -> tuple[list[np.ndarray], np.ndarray]:
    """Descent + random 4-flip kicks; collect states with Q <= max_q.

    NOTE: tracker.accept() returns the ENERGY (64*n*Q), not Q.  Always read
    tracker._q afterwards (same discipline as q12_steepest_descent).
    """
    rng = np.random.default_rng(seed)
    states: list[np.ndarray] = []
    qs = []
    while len(states) < n_states:
        seqs = rng.choice(np.array([-1, 1], dtype=np.int8), size=(4, n))
        tracker = Tracker()
        tracker.build(seqs)
        cur = seqs.copy()
        q = tracker._q
        for _ in range(4000):
            qs_all = tracker.flip_qs()
            improving = np.flatnonzero(qs_all < q)
            if improving.size:
                idx = int(improving[0])
                s, c = divmod(idx, n)
                tracker.accept(cur, s, c)
                q = tracker._q
                if q <= max_q and len(states) < n_states:
                    states.append(cur.copy())
                    qs.append(q)
                continue
            # stuck: kick
            cols = rng.integers(0, n, size=4)
            for s in range(4):
                tracker.accept(cur, s, int(cols[s]))
            q = tracker._q
        if not states:
            print(f"  n={n}: no low-Q states found", flush=True)
    return states, np.array(qs)


def low_q_state_stats() -> dict:
    out: dict = {}
    for n in (44, 50):
        states, qs = sample_low_q_states(n, n_states=300, max_q=30)
        vals = np.concatenate([d2(s) for s in states])
        out[f"n{n}_lowq"] = norm_stats(vals, f"descent+kick low-Q states n={n} "
                                             f"(n_state={len(states)}, Q<=30)")
        out[f"n{n}_lowq_state_q"] = {"mean": float(qs.mean()),
                                     "median": float(np.median(qs)),
                                     "hist": {str(k): int((qs == k).sum())
                                              for k in range(int(qs.min()), int(qs.max()) + 1)}}
        # shell condition: fraction of states with a flip whose ||d||^2 == Q
        ok = sum(int((d2(s) == int(np.dot(common.u_of(s), common.u_of(s)))).any())
                 for s in states)
        out[f"n{n}_lowq_has_shell_flip"] = f"{ok}/{len(states)}"
    return out


# --- 5. identity check on the 136 solutions -----------------------------------


def identity_check() -> dict:
    rows = [json.loads(l) for l in (HERE / "preflip_n44.jsonl")
            .read_text(encoding="utf-8").splitlines()]
    sols = [r for r in rows if r.get("solved") and r.get("pre")]
    ok_q = 0
    ok_d = 0
    qvals = []
    for r in sols:
        p = r["pre"]
        dvec = np.array(p["d"], dtype=np.int64)
        n2 = int((dvec * dvec).sum())
        qvals.append(n2)
        ok_q += int(p["q_pre"] == n2 == p["supp_size"])
        ok_d += int(p["d_equals_minus_u"])
    out = {"n_solutions": len(sols),
           "identity_q_pre_eq_norm2_eq_suppsize": f"{ok_q}/{len(sols)}",
           "d_equals_minus_u": f"{ok_d}/{len(sols)}"}
    arr = np.array(qvals)
    out["d_solve_norm2_hist"] = {str(k): int((arr == k).sum())
                                 for k in range(int(arr.min()), int(arr.max()) + 1)}
    return out


def main() -> None:
    out: dict = {}
    print("random states ...", flush=True)
    out["random"] = {str(n): random_state_stats(n, 300) for n in (44, 50, 52)}
    print("solution neighborhoods ...", flush=True)
    out["solution_shell"] = solution_shell()
    print("escape states ...", flush=True)
    out["escape"] = escape_state_stats()
    print("low-Q sampler (n=44/50, descent+kicks) ...", flush=True)
    out["lowq"] = low_q_state_stats()
    print("identity check ...", flush=True)
    out["identity"] = identity_check()
    print(json.dumps(out, indent=1, default=str))
    (HERE / "q13_results.json").write_text(json.dumps(out, indent=1, default=str),
                                           encoding="utf-8")


if __name__ == "__main__":
    main()
